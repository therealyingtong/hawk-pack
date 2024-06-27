use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::marker::PhantomData;
use tokio;
use tokio::runtime::Runtime;

use crate::{
    examples::lazy_memory_store::LazyMemoryStore, hnsw_db::FurthestQueue, GraphStore, VectorStore,
};

use super::EntryPoint;

const DB_URL: &str = "postgres://postgres:postgres@localhost/postgres";
const POOL_SIZE: u32 = 5;

const CREATE_TABLE_LINKS: &str = "
CREATE TABLE IF NOT EXISTS hawk_graph_links (
    source_ref text NOT NULL,
    links jsonb NOT NULL,
    CONSTRAINT hawk_graph_pkey PRIMARY KEY (source_ref)
)";

const CREATE_TABLE_ENTRY: &str = "
CREATE TABLE IF NOT EXISTS hawk_graph_entry (
    entry_point jsonb,
    id integer NOT NULL,
    CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (id)
)
";

const INSERT_EDGE: &str =
    "INSERT INTO `graph` (`source_ref`, `dest_ref`, `distance_ref`) VALUES ($1, $2, $3)";

pub struct GraphPg<V: VectorStore> {
    pool: sqlx::PgPool,
    phantom: PhantomData<V>,
}

impl<V: VectorStore> GraphPg<V> {
    pub async fn new() -> Result<Self, sqlx::Error> {
        let pool = PgPoolOptions::new()
            .max_connections(POOL_SIZE)
            .connect(DB_URL)
            .await?;

        sqlx::query(CREATE_TABLE_LINKS).execute(&pool).await?;
        sqlx::query(CREATE_TABLE_ENTRY).execute(&pool).await?;
        println!("Created tables");

        Ok(GraphPg {
            pool,
            phantom: PhantomData,
        })
    }
}

impl<V: VectorStore> GraphStore<V> for GraphPg<V> {
    async fn get_entry_point(&self) -> Option<EntryPoint<V::VectorRef>> {
        let r = sqlx::query("SELECT entry_point FROM hawk_graph_entry WHERE id = 0")
            .fetch_optional(&self.pool)
            .await
            .expect("Failed to fetch entry point");

        r.map(|row| {
            let x: sqlx::types::Json<EntryPoint<V::VectorRef>> = row.get("entry_point");
            let y: EntryPoint<V::VectorRef> = x.as_ref().clone();
            println!("Got entry point: {:?} {:?}", x, y);
            y
        })
    }

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V::VectorRef>) {
        sqlx::query(
            "
            INSERT INTO hawk_graph_entry (entry_point, id)
            VALUES ($1, 0) ON CONFLICT (id)
            DO UPDATE SET entry_point = EXCLUDED.entry_point
        ",
        )
        .bind(sqlx::types::Json(&entry_point))
        .execute(&self.pool)
        .await
        .expect("Failed to set entry point");
    }

    async fn get_links(&self, base: &<V as VectorStore>::VectorRef, lc: usize) -> FurthestQueue<V> {
        unimplemented!()
    }

    async fn set_links(&mut self, base: V::VectorRef, links: FurthestQueue<V>, lc: usize) {
        unimplemented!()
    }
}

#[tokio::test]
async fn test_db() {
    let mut graph = GraphPg::<LazyMemoryStore>::new().await.unwrap();
    let mut vector_store = LazyMemoryStore::new();

    let ep = graph.get_entry_point().await;
    println!("Entry point {:?}", ep);

    let ep2 = EntryPoint {
        vector_ref: vector_store.prepare_query(0),
        layer_count: ep.map(|e| e.layer_count).unwrap_or_default() + 1,
    };

    graph.set_entry_point(ep2.clone()).await;

    let ep3 = graph.get_entry_point().await.unwrap();
    println!("Entry point {:?}", ep3);
    assert_eq!(ep2, ep3);
}
