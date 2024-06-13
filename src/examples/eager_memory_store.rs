use crate::VectorStore;

/// Example implementation of a vector store - Eager variant.
///
/// A distance is computed eagerly on request in `eval_distance`, and `DistanceRef` points to an actual distance.
#[derive(Debug)]
pub struct EagerMemoryStore {
    vectors: Vec<u64>,

    pending_queries: Vec<u64>,

    distances: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VectorRef(usize);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct QueryRef(usize);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DistanceRef(usize);

impl EagerMemoryStore {
    pub fn new() -> Self {
        EagerMemoryStore {
            vectors: vec![],
            pending_queries: vec![],
            distances: vec![],
        }
    }
}

impl EagerMemoryStore {
    pub fn prepare_query(&mut self, raw_query: u64) -> <Self as VectorStore>::QueryRef {
        self.pending_queries.push(raw_query);
        QueryRef(self.pending_queries.len() - 1)
    }
}

impl VectorStore for EagerMemoryStore {
    type QueryRef = QueryRef;
    type VectorRef = VectorRef;
    type DistanceRef = DistanceRef;

    fn insert(&mut self, query_ref: &Self::QueryRef) -> Self::VectorRef {
        let query = self.pending_queries[query_ref.0];
        // Here the query can be removed (not implemented).

        self.vectors.push(query);
        VectorRef(self.vectors.len() - 1)
    }

    fn eval_distance(
        &mut self,
        query_ref: &Self::QueryRef,
        vector_ref: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Hamming distance
        let query = self.pending_queries[query_ref.0];
        let vector = self.vectors[vector_ref.0];
        let distance = (query ^ vector).count_ones() as u32;

        self.distances.push(distance);
        DistanceRef(self.distances.len() - 1)
    }

    fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        self.distances[distance.0] == 0
    }

    fn less_than(&self, distance1: &Self::DistanceRef, distance2: &Self::DistanceRef) -> bool {
        self.distances[distance1.0] < self.distances[distance2.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_distance() {
        let mut store = EagerMemoryStore::new();

        let query = store.prepare_query(11);
        let vector = store.insert(&query);
        let distance = store.eval_distance(&query, &vector);
        assert!(store.is_match(&distance));

        let other_query = store.prepare_query(12);
        let other_vector = store.insert(&other_query);
        let other_distance = store.eval_distance(&query, &other_vector);
        assert!(!store.is_match(&other_distance));
    }
}
