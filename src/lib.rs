pub mod graph_store;
pub mod hnsw_db;

pub mod examples;
mod linear_db;

use std::fmt::Debug;
use std::hash::Hash;

pub use graph_store::GraphStore;
use serde::Serialize;

// The operations exposed by a vector store, sufficient for a search algorithm.
pub trait VectorStore: Debug {
    /// Opaque reference to a query.
    ///
    /// Example: a preprocessed representation optimized for distance evaluations.
    type QueryRef: Clone
        + Debug
        + PartialEq
        + Eq
        + Hash
        + Sync
        + Serialize
        + for<'de> serde::Deserialize<'de>;

    /// Opaque reference to a stored vector.
    ///
    /// Example: a vector ID.
    type VectorRef: Clone
        + Debug
        + PartialEq
        + Eq
        + Hash
        + Sync
        + Serialize
        + for<'de> serde::Deserialize<'de>;

    /// Opaque reference to a distance metric.
    ///
    /// Example: an encrypted distance.
    type DistanceRef: Clone
        + Debug
        + PartialEq
        + Eq
        + Hash
        + Sync
        + Serialize
        + for<'de> serde::Deserialize<'de>;

    /// Persist a query as a new vector in the store, and return a reference to it.
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef;

    /// Evaluate the distance between a query and a vector.
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef;

    /// Check whether a distance is a match, meaning the query is considered equivalent to a previously inserted vector.
    async fn is_match(&self, distance: &Self::DistanceRef) -> bool;

    /// Compare two distances.
    async fn less_than(&self, distance1: &Self::DistanceRef, distance2: &Self::DistanceRef)
        -> bool;

    /// Find the insertion index for a target distance to maintain order in a list of ascending distances.
    async fn search_sorted(
        &self,
        distances: &[Self::DistanceRef],
        target: &Self::DistanceRef,
    ) -> usize {
        let mut left = 0;
        let mut right = distances.len();

        while left < right {
            let mid = left + (right - left) / 2;

            match self.less_than(&distances[mid], target).await {
                true => left = mid + 1,
                false => right = mid,
            }
        }
        left
    }
}
