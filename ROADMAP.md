# Marquetry Roadmap

Marquetry is an independent deep learning framework that owns its whole stack — tensor,
autograd, and kernels. The pure-Python era ended with v0.3.0; from v0.4.0 the compute engine is
implemented in Rust, targeting practical performance, a lightweight footprint, and GPU support
across Apple, NVIDIA, and AMD hardware — with the browser as a long-term first-class target.

This document is the product-level direction. The normative engineering plan for the engine
migration — technology decisions, phase structure, exit criteria, test gates — is
[REPLACE_TO_RUST_ENGINE.md](./REPLACE_TO_RUST_ENGINE.md); where the two differ in detail, that
document wins.

## Releases

### v0.4.0 — Rust CPU engine

The compute engine (strided tensor, define-by-run autograd, CPU backend) moves to Rust behind
the existing Python API. **Full feature parity with v0.3 is the bar**: the entire v0.3 test
suite passes, `.mq` model archives and ONNX export keep working, and classic ML (decision
trees, random forest, SVM) continues as pure Python. Headline target: ≥10× v0.3 on CPU training
workloads.

### v0.5.0 — Apple Metal backend

The first GPU backend. The GPU parity, determinism, and capability test suites established here
govern every backend that follows.

### v0.6.0 — NVIDIA CUDA backend

cuBLAS-backed GEMM with custom CUDA kernels for the rest of the operator set.

### v0.7.0 — wgpu backend (AMD / Windows / Intel / Web)

One portable backend covers AMD, Windows, and Intel GPUs — and the browser: the core builds for
wasm32 and a training loop runs on WebGPU in a browser.

### v0.8.0 — Integrated optimization

With all four backends in place, the cross-backend performance push: kernel fusion guided by
profiling, convolution provider upgrades (cuDNN / MPSCNN), Metal GEMM re-evaluation, wgpu
autotuner maturation, allocator and stream tuning.

### v0.9.0 — Modern operator expansion

Broad growth of the operator library — contemporary activations, optimizers, normalization and
attention-era building blocks — with every addition entering through the same test gates as the
core.

### v0.10.0 — Stabilization

Hardening, edge-case fuzzing, API freeze, and documentation refresh on the road to 1.0.

### v1.0.0 — Stable release

The Rust engine at full maturity across CPU / Metal / CUDA / wgpu, with published benchmarks.

## Beyond 1.0

- **Marquetry on the Web** — building on the wasm32 + WebGPU foundation from v0.7.0, toward a
  browser-native and eventually graphical model-building experience.
- **Documentation platform refresh** — from static pages to a modern documentation site.
- **Tracked candidates** — mixed-precision training, DLPack interop, a native ROCm backend,
  crates.io publication of the core: see the Open Questions in REPLACE_TO_RUST_ENGINE.md.

Versions and ordering may evolve; this document tracks direction, not commitments.
