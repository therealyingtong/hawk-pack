use sqlx::postgres::PgPoolOptions;
use tokio;

const CREATE_TABLE: &str = "CREATE TABLE `graph` (`source_ref` TEXT, `dest_ref` TEXT, `distance_ref` TEXT, `rank` INT, PRIMARY KEY (source_ref, dest_ref))";

const CREATE_INDEX: &str = "CREATE UNIQUE INDEX `index_graph_links` ON `graph` (source_ref, rank)";

const INSERT_EDGE: &str =
    "INSERT INTO `graph` (`source_ref`, `dest_ref`, `distance_ref`) VALUES ($1, $2, $3)";

async fn setup_db() -> Result<(), sqlx::Error> {
    // Create a connection pool
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect("postgres://postgres:x@localhost/postgres")
        .await?;

    // Make a simple query to return the given parameter (use a question mark `?` instead of `$1` for MySQL/MariaDB)
    let row: (i64,) = sqlx::query_as("SELECT $1")
        .bind(150_i64)
        .fetch_one(&pool)
        .await?;

    assert_eq!(row.0, 150);

    Ok(())
}

#[tokio::test]
async fn test_db() {
    //setup_db().await.unwrap();
}
