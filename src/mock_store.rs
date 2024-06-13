use super::{Code, VectorStore};

// Example implementation of a vector store.
#[derive(Debug)]
pub struct MockVectorStore {
    vectors: Vec<u64>,
}

impl MockVectorStore {
    pub fn new() -> Self {
        MockVectorStore { vectors: vec![] }
    }
}

impl VectorStore for MockVectorStore {
    type QueryRef = u64; // Implementation-specific encoding of queries and vectors.
    type VectorRef = usize; // Vector ID.
    type DistanceRef = u32; // Implementation-specific distance metric.

    fn prepare_query(&mut self, raw_query: &Code) -> Self::QueryRef {
        raw_query
            .0
            .iter()
            .rev()
            .fold(0, |acc, &x| acc * 256 + x as u64)
    }

    fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.vectors.push(*query);
        self.vectors.len() - 1
    }

    fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Hamming distance
        let vector = self.vectors[*vector];
        (query ^ vector).count_ones() as u32
    }

    fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        *distance == 0
    }

    fn less_than(&self, distance1: &Self::DistanceRef, distance2: &Self::DistanceRef) -> bool {
        distance1 < distance2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_distance() {
        let mut store = MockVectorStore::new();

        let query = 11;
        let vector = store.insert(&query);
        let distance = store.eval_distance(&query, &vector);
        assert_eq!(distance, 0);

        let different_vector = store.insert(&12);
        let distance = store.eval_distance(&query, &different_vector);
        assert!(distance > 0);
    }
}
