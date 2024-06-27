# Hawk Pack

<p align="center">
<img src="https://github.com/Inversed-Tech/hawk-pack/assets/8718243/4a2b613f-2d07-4afe-9c1f-eda2f5a6f90b" width=50% alt="AI image of a Hawk Pack">
</p>

### Concept

This is a search engine for approximate nearest vectors (kNN).

All vector and scalar operations are delegated to an external system. The notions of vector and distance are fully abstracted.
The core question that the engine asks to the vector store is this:

> Given vector IDs `a`, `b`, `c`, which of `a` or `b` is closer to `c`?

From there, the algorithm [HNSW](https://arxiv.org/abs/1603.09320) is implemented to create a graph database along with sub-linear search and insertion procedures.

### Usage

```bash
docker-compose up -d

cargo test
```

See the `trait VectorStore` for the interface that the external store must provide. Check out the `examples` module.
