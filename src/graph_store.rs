use crate::hnsw_db::FurthestQueue;
use crate::VectorStore;
use std::fmt::Debug;

pub mod graph_mem;
mod graph_pg;

pub trait GraphStore<V: VectorStore> {
    fn get_entry_point(&self) -> Option<EntryPoint<V::VectorRef>>;

    fn set_entry_point(&mut self, entry_point: EntryPoint<V::VectorRef>);

    fn get_links(&self, base: &<V as VectorStore>::VectorRef, lc: usize) -> FurthestQueue<V>;

    fn set_links(&mut self, base: V::VectorRef, links: FurthestQueue<V>, lc: usize);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint<VectorRef: Clone + Debug> {
    pub vector_ref: VectorRef,
    pub layer_count: usize,
}
