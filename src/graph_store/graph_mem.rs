use super::{EntryPoint, GraphStore};
use crate::{
    hnsw_db::{FurthestQueue, FurthestQueueV},
    VectorStore,
};
use std::collections::HashMap;

#[derive(Default, Clone)]
pub struct GraphMem<V: VectorStore> {
    entry_point: Option<EntryPoint<V::VectorRef>>,
    layers: Vec<Layer<V>>,
}

impl<V: VectorStore> GraphMem<V> {
    pub fn new() -> Self {
        GraphMem {
            entry_point: None,
            layers: vec![],
        }
    }
}

// Plain converter for a Graph structure that has the same distance ref and vector ref
// Needed when switching from a PlaintextStore to a secret shared VectorStore.
impl<V: VectorStore> GraphMem<V> {
    #[cfg(test)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[cfg(test)]
    pub fn layer(&self, layer: usize) -> &Layer<V> {
        &self.layers[layer]
    }

    pub fn from_another<U>(graph: GraphMem<U>) -> Self
    where
        U: VectorStore,
        U: VectorStore<DistanceRef = V::DistanceRef>,
        U: VectorStore<VectorRef = V::VectorRef>,
    {
        let new_entry = graph.entry_point as Option<EntryPoint<V::VectorRef>>;
        let layers: Vec<_> = graph
            .layers
            .into_iter()
            .map(|v| {
                let links: HashMap<_, _> = v.links.into_iter().collect();
                Layer::<V> { links }
            })
            .collect();
        GraphMem::<V> {
            entry_point: new_entry,
            layers,
        }
    }
}

impl<V: VectorStore> GraphStore<V> for GraphMem<V> {
    async fn get_entry_point(&self) -> Option<EntryPoint<V::VectorRef>> {
        self.entry_point.clone()
    }

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V::VectorRef>) {
        if let Some(previous) = self.entry_point.as_ref() {
            assert!(
                previous.layer_count < entry_point.layer_count,
                "A new entry point should be on a higher layer than before."
            );
        }

        while entry_point.layer_count > self.layers.len() {
            self.layers.push(Layer::new());
        }

        self.entry_point = Some(entry_point);
    }

    async fn get_links(
        &self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> FurthestQueueV<V> {
        let layer = &self.layers[lc];
        if let Some(links) = layer.get_links(base) {
            links.clone()
        } else {
            FurthestQueue::new()
        }
    }

    async fn set_links(&mut self, base: V::VectorRef, links: FurthestQueueV<V>, lc: usize) {
        let layer = &mut self.layers[lc];
        layer.set_links(base, links);
    }
}

#[derive(PartialEq, Eq, Default, Clone)]
pub struct Layer<V: VectorStore> {
    /// Map a base vector to its neighbors, including the distance base-neighbor.
    links: HashMap<V::VectorRef, FurthestQueueV<V>>,
}

impl<V: VectorStore> Layer<V> {
    #[cfg(test)]
    pub fn nodes(&self) -> Vec<&V::VectorRef> {
        self.links.keys().collect()
    }

    fn new() -> Self {
        Layer {
            links: HashMap::new(),
        }
    }

    fn get_links(&self, from: &V::VectorRef) -> Option<&FurthestQueueV<V>> {
        self.links.get(from)
    }

    fn set_links(&mut self, from: V::VectorRef, links: FurthestQueueV<V>) {
        self.links.insert(from, links);
    }
}
