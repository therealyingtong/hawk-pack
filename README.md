# Hawk Pack

This is a search engine for approximate nearest vectors (kNN).

All vector and scalar operations are delegated to an external system. The notions of vector and distance are fully abstracted.
The core question that the engine asks to the vector store is this:

> Given vector IDs `a`, `b`, `c`, which of `a` or `b` is closer to `c`?

From there, the algorithm HSNW is implemented to create a graph database along with sub-linear search and insertion procedures.

### Usage

```bash
cargo test
```

See the `trait VectorStore` for the interface that the external store must provide.
