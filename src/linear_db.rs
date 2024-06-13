use crate::VectorStore;

pub struct LinearDb<V: VectorStore> {
    store: V,
    vectors: Vec<V::VectorRef>,
}

impl<V: VectorStore> LinearDb<V> {
    pub fn new(store: V) -> Self {
        LinearDb {
            store,
            vectors: vec![],
        }
    }

    pub fn insert(&mut self, raw_query: &crate::Code) -> bool {
        let query = self.store.prepare_query(raw_query);

        if self.exists(&query) {
            return false;
        }

        let vector = self.store.insert(&query);
        self.vectors.push(vector);
        true
    }

    fn exists(&mut self, query: &V::QueryRef) -> bool {
        for vector in &self.vectors {
            let distance = self.store.eval_distance(&query, vector);
            if self.store.is_match(&distance) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_db() {
        let store = crate::MockVectorStore::new();
        let mut db = LinearDb::new(store);

        let query = crate::Code(vec![1, 2, 3]);

        assert!(db.insert(&query));
        assert!(!db.insert(&query));
    }
}
