use std::collections::HashMap;

use crate::VectorStore;

use super::queue::FurthestQueue;

pub struct Layer<V: VectorStore> {
    /// Map a base vector to its neighbors, including the distance base-neighbor.
    neighbors: HashMap<V::VectorRef, FurthestQueue<V>>,
}

impl<V: VectorStore> Layer<V> {
    pub fn new() -> Self {
        Layer {
            neighbors: HashMap::new(),
        }
    }

    pub fn get_neighbors(&self, from: &V::VectorRef) -> Option<&FurthestQueue<V>> {
        self.neighbors.get(from)
    }

    pub fn set_neighbors(&mut self, from: V::VectorRef, neighbors: FurthestQueue<V>) {
        self.neighbors.insert(from, neighbors);
    }

    pub fn connect(
        &mut self,
        store: &mut V,
        from: V::VectorRef,
        to: V::VectorRef,
        distance: V::DistanceRef,
        max_links: usize,
    ) {
        let mut neighbors = self
            .neighbors
            .entry(from)
            .or_insert_with(FurthestQueue::new);
        neighbors.insert(store, to, distance);
        neighbors.trim_to_k_nearest(max_links);
    }
}
