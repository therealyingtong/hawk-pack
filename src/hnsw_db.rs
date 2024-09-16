// Converted from Python to Rust.
use std::collections::HashSet;
mod queue;
use aes_prng::AesRng;
pub use queue::{FurthestQueue, FurthestQueueV, NearestQueue, NearestQueueV};
use rand::{Rng, RngCore, SeedableRng};
pub mod coroutine;

use crate::{graph_store::EntryPoint, GraphStore, VectorStore};

#[allow(non_snake_case)]
#[derive(PartialEq, Clone)]
struct Params {
    ef: usize,
    M: usize,
    Mmax: usize,
    Mmax0: usize,
    m_L: f64,
}

/// An implementation of the HNSW algorithm.
///
/// Operations on vectors are delegated to a VectorStore.
/// Operations on the graph are delegate to a GraphStore.
#[derive(Clone)]
pub struct HawkSearcher<V: VectorStore, G: GraphStore<V>> {
    params: Params,
    pub vector_store: V,
    pub graph_store: G,
    rng: AesRng,
}

impl<V: VectorStore, G: GraphStore<V>> HawkSearcher<V, G> {
    pub fn m_max(&self) -> usize {
        self.params.Mmax
    }

    pub fn m_max_0(&self) -> usize {
        self.params.Mmax0
    }

    pub fn new<R: RngCore>(vector_store: V, graph_store: G, rng: &mut R) -> Self {
        let rng = AesRng::from_rng(rng).unwrap();
        HawkSearcher {
            params: Params {
                ef: 32,
                M: 32,
                Mmax: 32,
                Mmax0: 32,
                m_L: 0.3,
            },
            vector_store,
            graph_store,
            rng,
        }
    }

    async fn connect_bidir(
        &mut self,
        q: &V::VectorRef,
        mut neighbors: FurthestQueueV<V>,
        lc: usize,
    ) {
        neighbors.trim_to_k_nearest(self.params.M);
        let neighbors = neighbors;

        let max_links = if lc == 0 {
            self.params.Mmax0
        } else {
            self.params.Mmax
        };

        // Connect all n -> q.
        for (n, nq) in neighbors.iter() {
            let mut links = self.graph_store.get_links(n, lc).await;
            links
                .insert(&self.vector_store, q.clone(), nq.clone())
                .await;
            links.trim_to_k_nearest(max_links);
            self.graph_store.set_links(n.clone(), links, lc).await;
        }

        // Connect q -> all n.
        self.graph_store.set_links(q.clone(), neighbors, lc).await;
    }

    pub fn select_layer(&mut self) -> usize {
        let random = self.rng.gen::<f64>();
        (-random.ln() * self.params.m_L) as usize
    }

    fn ef_for_layer(&self, _lc: usize) -> usize {
        // Note: the original HNSW paper uses a different ef parameter depending on:
        // - bottom layer versus higher layers,
        // - search versus insertion,
        // - during insertion, mutated versus non-mutated layers,
        // - the requested K nearest neighbors.
        // Here, we treat search and insertion the same way and we use the highest parameter everywhere.
        self.params.ef
    }

    #[allow(non_snake_case)]
    async fn search_init(&self, query: &V::QueryRef) -> (FurthestQueueV<V>, usize) {
        if let Some(entry_point) = self.graph_store.get_entry_point().await {
            let entry_vector = entry_point.vector_ref;
            let distance = self.vector_store.eval_distance(query, &entry_vector).await;

            let mut W = FurthestQueueV::<V>::new();
            W.insert(&self.vector_store, entry_vector, distance).await;

            (W, entry_point.layer_count)
        } else {
            (FurthestQueue::new(), 0)
        }
    }

    /// Mutate W into the ef nearest neighbors of q_vec in the given layer.
    #[allow(non_snake_case)]
    async fn search_layer(&self, q: &V::QueryRef, W: &mut FurthestQueueV<V>, ef: usize, lc: usize) {
        // v: The set of already visited vectors.
        let mut v = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // C: The set of vectors to visit, ordered by increasing distance to the query.
        let mut C = NearestQueue::from_furthest_queue(W);

        // fq: The current furthest distance in W.
        let (_, mut fq) = W.get_furthest().expect("W cannot be empty").clone();

        while C.len() > 0 {
            let (c, cq) = C.pop_nearest().expect("C cannot be empty").clone();

            // If the nearest distance to C is greater than the furthest distance in W, then we can stop.
            if self.vector_store.less_than(&fq, &cq).await {
                break;
            }

            // Visit all neighbors of c.
            let c_links = self.graph_store.get_links(&c, lc).await;

            // Evaluate the distances of the neighbors to the query, as a batch.
            let c_links = {
                let e_batch = c_links
                    .iter()
                    .map(|(e, _ec)| e.clone())
                    .filter(|e| {
                        // Visit any node at most once.
                        v.insert(e.clone())
                    })
                    .collect::<Vec<_>>();

                let distances = self.vector_store.eval_distance_batch(q, &e_batch).await;

                e_batch
                    .into_iter()
                    .zip(distances.into_iter())
                    .collect::<Vec<_>>()
            };

            for (e, eq) in c_links.into_iter() {
                if W.len() == ef {
                    // When W is full, we decide whether to replace the furthest element.
                    if self.vector_store.less_than(&eq, &fq).await {
                        // Make room for the new better candidate…
                        W.pop_furthest();
                    } else {
                        // …or ignore the candidate and do not continue on this path.
                        continue;
                    }
                }

                // Track the new candidate in C so we will continue this path later.
                C.insert(&self.vector_store, e.clone(), eq.clone()).await;

                // Track the new candidate as a potential k-nearest.
                W.insert(&self.vector_store, e, eq).await;

                // fq stays the furthest distance in W.
                (_, fq) = W.get_furthest().expect("W cannot be empty").clone();
            }
        }
    }

    #[allow(non_snake_case)]
    pub async fn search_to_insert(&self, query: &V::QueryRef) -> Vec<FurthestQueueV<V>> {
        let mut links = vec![];

        let (mut W, layer_count) = self.search_init(query).await;

        // From the top layer down to layer 0.
        for lc in (0..layer_count).rev() {
            let ef = self.ef_for_layer(lc);
            self.search_layer(query, &mut W, ef, lc).await;

            links.push(W.clone());
        }

        links.reverse(); // We inserted top-down, so reverse to match the layer indices (bottom=0).
        links
    }

    pub async fn insert_from_search_results(
        &mut self,
        inserted_vector: V::VectorRef,
        links: Vec<FurthestQueueV<V>>,
    ) {
        let layer_count = links.len();

        // Choose a maximum layer for the new vector. It may be greater than the current number of layers.
        let l = self.select_layer();

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_links) in links.into_iter().enumerate().take(l + 1) {
            self.connect_bidir(&inserted_vector, layer_links, lc).await;
        }

        // If the new vector goes into a layer higher than ever seen before, then it becomes the new entry point of the graph.
        if l >= layer_count {
            self.graph_store
                .set_entry_point(EntryPoint {
                    vector_ref: inserted_vector,
                    layer_count: l + 1,
                })
                .await;
        }
    }

    pub async fn is_match(&self, neighbors: &[FurthestQueueV<V>]) -> bool {
        match neighbors
            .first()
            .and_then(|bottom_layer| bottom_layer.get_nearest())
        {
            None => false, // Empty database.
            Some((_, smallest_distance)) => self.vector_store.is_match(smallest_distance).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::lazy_memory_store::LazyMemoryStore;
    use crate::graph_store::graph_mem::GraphMem;
    use tokio;

    #[tokio::test]
    async fn test_hnsw_db() {
        let vector_store = LazyMemoryStore::new();
        let graph_store = GraphMem::new();
        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut db = HawkSearcher::new(vector_store, graph_store, &mut rng);

        let queries = (0..100)
            .map(|raw_query| db.vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(query).await;
            assert!(!db.is_match(&neighbors).await);
            // Insert the new vector into the store.
            let inserted = db.vector_store.insert(query).await;
            db.insert_from_search_results(inserted, neighbors).await;
        }

        // Search for the same codes and find matches.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(query).await;
            assert!(db.is_match(&neighbors).await);
        }
    }
}
