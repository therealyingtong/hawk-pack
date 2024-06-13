// Converted from Python to Rust.
use std::{
    cmp::{max, min},
    collections::HashSet,
};
mod layer;
mod queue;

use crate::VectorStore;
use layer::Layer;

use self::queue::{FurthestQueue, NearestQueue};

struct Params {
    ef: usize,
    M: usize,
    Mmax: usize,
    Mmax0: usize,
    m_L: f64,
}

pub struct HSNW<V: VectorStore> {
    params: Params,
    store: V,
    layers: Vec<Layer<V>>,
    entry_point: Option<V::VectorRef>,
}

impl<V: VectorStore> HSNW<V> {
    pub fn new(store: V) -> Self {
        HSNW {
            params: Params {
                ef: 32,
                M: 32,
                Mmax: 32,
                Mmax0: 32,
                m_L: 0.3,
            },
            store,
            layers: vec![],
            entry_point: None,
        }
    }

    fn connect_bidir(&mut self, q: &V::VectorRef, mut neighbors: FurthestQueue<V>, lc: usize) {
        neighbors.trim_to_k_nearest(self.params.M);
        let neighbors = neighbors;

        let max_links = if lc == 0 {
            self.params.Mmax0
        } else {
            self.params.Mmax
        };

        let mut layer = &mut self.layers[lc];

        // Connect n -> q.
        for (n, nq) in neighbors.iter() {
            layer.connect(&mut self.store, n.clone(), q.clone(), nq.clone(), max_links);
        }

        // Connect q -> n.
        layer.set_neighbors(q.clone(), neighbors);
    }

    fn select_layer(&self) -> usize {
        let random = rand::random::<f64>();
        (-random.ln() * self.params.m_L) as usize
    }

    fn ef_for_layer(&self, lc: usize) -> usize {
        // Note: the original HNSW paper uses a different ef parameter depending on:
        // - bottom layer versus higher layers,
        // - search versus insertion,
        // - during insertion, mutated versus non-mutated layers,
        // - the requested K nearest neighbors.
        // Here, we treat search and insertion the same way and we use the highest parameter everywhere.
        self.params.ef
    }

    fn search_init(&mut self, query: &V::QueryRef) -> FurthestQueue<V> {
        let mut W = FurthestQueue::<V>::new();
        if let Some(entry_point) = &self.entry_point {
            let distance = self.store.eval_distance(query, entry_point);
            W.insert(&mut self.store, entry_point.clone(), distance);
        }
        W
    }

    /// Mutate W into the ef nearest neighbors of q_vec in the given layer.
    fn search_layer(&mut self, q: &V::QueryRef, W: &mut FurthestQueue<V>, ef: usize, lc: usize) {
        let layer = &self.layers[lc];

        // v: The set of already visited vectors.
        let mut v = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // C: The set of vectors to visit, ordered by increasing distance to the query.
        let mut C = NearestQueue::from_furthest_queue(W);

        // fq: The current furthest distance in W.
        let (_, mut fq) = W.get_furthest().expect("W cannot be empty").clone();

        while C.len() > 0 {
            let (c, cq) = C.pop_nearest().expect("C cannot be empty").clone();

            // If the nearest distance to C is greater than the furthest distance in W, then we can stop.
            if self.store.less_than(&fq, &cq) {
                break;
            }

            // Visit all neighbors of c.
            if let Some(c_neighbors) = layer.get_neighbors(&c) {
                for (e, _ec) in c_neighbors.iter() {
                    // Visit any node at most once.
                    if !v.insert(e.clone()) {
                        continue;
                    }

                    let eq = self.store.eval_distance(q, e);

                    if W.len() == ef {
                        // When W is full, we decide whether to replace the furthest element.
                        if self.store.less_than(&eq, &fq) {
                            // Make room for the new better candidate…
                            W.pop_furthest();
                        } else {
                            // …or ignore the candidate and do not continue on this path.
                            continue;
                        }
                    }

                    // Track the new candidate in C so we will continue this path later.
                    C.insert(&mut self.store, e.clone(), eq.clone());

                    // Track the new candidate as a potential k-nearest.
                    W.insert(&mut self.store, e.clone(), eq);

                    // fq stays the furthest distance in W.
                    (_, fq) = W.get_furthest().expect("W cannot be empty").clone();
                }
            }
        }
    }

    pub fn search_to_insert(&mut self, query: &V::QueryRef) -> Vec<FurthestQueue<V>> {
        let mut links = vec![];

        let mut W = self.search_init(&query);

        // From the top layer down to layer 0.
        for lc in (0..self.layers.len()).rev() {
            let ef = self.ef_for_layer(lc);
            self.search_layer(&query, &mut W, ef, lc);

            links.push(W.clone());
        }

        links.reverse(); // We inserted top-down, so reverse to match the layer indices (bottom=0).
        links
    }

    pub fn insert_from_search_results(
        &mut self,
        query: &V::QueryRef,
        links: Vec<FurthestQueue<V>>,
    ) {
        assert_eq!(links.len(), self.layers.len());

        // Insert the new vector into the store.
        let inserted_vector = self.store.insert(&query);

        // Choose a maximum layer for the new vector. It may be greater than the current number of layers.
        let l = self.select_layer();

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_links) in links.into_iter().enumerate().take(l + 1) {
            self.connect_bidir(&inserted_vector, layer_links, lc);
        }

        // If the new vector goes into a layer higher than ever seen before, then it becomes the new entry point of the graph.
        if l >= self.layers.len() {
            self.entry_point = Some(inserted_vector);
            while l >= self.layers.len() {
                self.layers.push(Layer::new());
            }
        }
    }

    pub fn is_match(&self, neighbors: &[FurthestQueue<V>]) -> bool {
        neighbors
            .first()
            .and_then(|bottom_layer| bottom_layer.get_nearest())
            .map(|(_, smallest_distance)| self.store.is_match(smallest_distance))
            .unwrap_or(false) // Empty database.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_db() {
        let store = crate::MockVectorStore::new();
        let mut db = HSNW::new(store);

        let queries = (0..100)
            .map(|i| {
                let raw_query = vec![i];
                let query = db.store.prepare_query(&raw_query);
                query
            })
            .collect::<Vec<_>>();

        // Insert the codes.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(&query);
            assert!(!db.is_match(&neighbors));
            db.insert_from_search_results(&query, neighbors);
        }

        // Search for the same codes and find matches.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(&query);
            println!("{:?}", neighbors);
            assert!(db.is_match(&neighbors));
        }
    }
}
