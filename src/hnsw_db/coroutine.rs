use crate::{
    graph_store::EntryPoint,
    hnsw_db::{FurthestQueue, HawkSearcher},
    GraphStore, Ref, VectorStore,
};
use serde::Serialize;
use std::fmt::Debug;
use std::hash::Hash;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

// TODO: handle channel errors.

/// Coroutine-based implementation of HNSW.
///
/// In this implementation, the HNSW algorithm is split into multiple asynchronous
/// operations that are executed in a coroutine-like fashion. The `HawkStream` struct
/// holds the coroutine state and interactions.
///
/// To perform a search, create a `HawkRun` with a query and call `next` until it
/// returns `None`. Each call to `next` will return an `Op` that represents the
/// operation that the search routine is waiting for. The caller is responsible for
/// executing the operation and sending the result back to the routine.
pub fn search_to_insert_stream<Q, V, D>(query: Q) -> HawkStream<Q, V, D>
where
    Q: Ref + Send + 'static,
    V: Ref + Send + 'static,
    D: Ref + Send + 'static,
{
    let (tx, rx) = mpsc::channel(1);
    tokio::spawn(async move {
        let mut hawk = HawkSearcher::new(
            OpsCollector { ops: tx.clone() },
            OpsCollector { ops: tx.clone() },
        );
        let result = hawk.search_to_insert(&query).await;
        let result = result.into_iter().map(|queue| queue.into()).collect();
        tx.send(Op::SearchResult { query, result }).await.unwrap();
    });
    ReceiverStream::new(rx)
}

pub type HawkStream<Q, V, D> = ReceiverStream<Op<Q, V, D>>;

pub enum Op<Query, Vector, Distance> {
    // VectorStore operations.
    EvalDistanceBatch {
        query: Query,
        vectors: Vec<Vector>,
        reply: oneshot::Sender<Vec<Distance>>,
    },
    LessThanBatch {
        distance: Distance,
        distances: Vec<Distance>,
        reply: oneshot::Sender<Vec<bool>>,
    },

    // GraphStore operations.
    GetEntryPoint {
        reply: oneshot::Sender<Option<EntryPoint<Vector>>>,
    },
    SetEntryPoint {
        entry_point: EntryPoint<Vector>,
    },
    GetLinks {
        base: Vector,
        lc: usize,
        reply: oneshot::Sender<Vec<(Vector, Distance)>>, // FurthestQueue
    },
    SetLinks {
        base: Vector,
        links: Vec<(Vector, Distance)>, // FurthestQueue
        lc: usize,
    },

    // Result.
    SearchResult {
        query: Query,
        result: Vec<Vec<(Vector, Distance)>>, // FurthestQueue
    },
}

#[derive(Clone, Debug)]
struct OpsCollector<Q, V, D> {
    ops: mpsc::Sender<Op<Q, V, D>>,
}

impl<Q: Ref, V: Ref, D: Ref> VectorStore for OpsCollector<Q, V, D> {
    type QueryRef = Q;
    type VectorRef = V;
    type DistanceRef = D;

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        todo!()
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        self.eval_distance_batch(query, &[vector.clone()])
            .await
            .pop()
            .unwrap()
    }

    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Vec<Self::DistanceRef> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::EvalDistanceBatch {
            query: query.clone(),
            vectors: vectors.to_vec(),
            reply,
        };

        self.ops.send(op).await.unwrap();
        get_reply.await.unwrap()
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        self.less_than_batch(distance1, &[distance2.clone()])
            .await
            .pop()
            .unwrap()
    }

    async fn less_than_batch(
        &self,
        distance: &Self::DistanceRef,
        distances: &[Self::DistanceRef],
    ) -> Vec<bool> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::LessThanBatch {
            distance: distance.clone(),
            distances: distances.to_vec(),
            reply,
        };

        self.ops.send(op).await.unwrap();
        get_reply.await.unwrap()
    }

    async fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        todo!()
    }
}

impl<Q: Ref, V: Ref, D: Ref> GraphStore<OpsCollector<Q, V, D>> for OpsCollector<Q, V, D> {
    async fn get_entry_point(&self) -> Option<EntryPoint<V>> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::GetEntryPoint { reply };

        self.ops.send(op).await.unwrap();
        get_reply.await.unwrap()
    }

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V>) {
        let op = Op::SetEntryPoint { entry_point };
        self.ops.send(op).await.unwrap();
    }

    async fn get_links(&self, base: &V, lc: usize) -> FurthestQueue<Self> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::GetLinks {
            base: base.clone(),
            lc,
            reply,
        };

        self.ops.send(op).await.unwrap();
        let links = get_reply.await.unwrap();
        FurthestQueue::from_ascending_vec(links)
    }

    async fn set_links(&mut self, base: V, links: FurthestQueue<Self>, lc: usize) {
        let op = Op::SetLinks {
            base,
            links: links.into(),
            lc,
        };
        self.ops.send(op).await.unwrap();
    }
}
