# REPLACE_TO_RUST_ENGINE.md

**Marquetry Rust Engine Migration Plan**

- Status: **Approved** (living document — updated as decisions evolve)
- Decided: 2026-06-11
- Owner: SHIMA0111
- Applies from: after v0.3.0 (the final pure-Python release)

---

## 1. Background & Goals

Marquetry v0.1–v0.3 was a pure-Python deep learning framework, built for clarity and education.
With v0.3.0 released, the project moves to its next stage: **replacing the compute engine with Rust**
while keeping the Python-facing API as the primary user surface.

### Goals

1. **Practical performance.** Unlike the pure-Python era, the Rust engine explicitly targets
   real-world speed and a lightweight footprint (fast startup, small dependency tree, low memory overhead).
2. **GPU support as a first-class citizen.** Apple Metal, NVIDIA CUDA, and AMD GPUs are all in scope.
3. **Own the stack.** The tensor representation, autograd, and operator layer are implemented by this
   project — not delegated to an existing tensor runtime.
4. **Web-ready core.** The long-term roadmap includes a browser version. The engine core must compile
   to `wasm32`, and the portable GPU backend must run on WebGPU.
5. **Keep the define-by-run developer experience** that Marquetry has always had.

### Non-Goals

- Building on top of existing tensor runtimes (**candle / burn / tch / tract are explicitly rejected** —
  they would reduce Marquetry to a wrapper, which contradicts the project philosophy).
- Hand-optimizing GEMM-class microkernels (see Guiding Principle #2).
- Distributed training, quantization, and training-scale LLM features (out of scope for v1.0).

---

## 2. Guiding Principles

1. **Own the structure, delegate the muscle.** We implement the tensor type, strides/broadcasting,
   autograd graph, operator dispatch, and backend abstraction ourselves. We do **not** hand-write
   GEMM-class optimized microkernels: matrix-multiply and decomposition primitives are delegated to
   specialist libraries (faer, cuBLAS, MPS, …). Nothing good comes from hand-rolling a BLAS.
   - *Accepted exception:* the portable WGSL backend has no vendor library layer, so we own its GEMM
     kernels. "Own" means owning a proper **kernel generator**, not writing one naive shader: the
     established WebGPU-ML pattern (burn/CubeCL, TensorFlow.js and ONNX Runtime WebGPU backends) of
     workgroup shared-memory tiling + register blocking, kernels specialized per (dtype, tile config,
     shape class) via generation/specialization constants, pipeline-cached, with lightweight
     autotuning over tile configs, and subgroup ops as progressive enhancement where available.
     Its performance ceiling vs native backends is accepted — it is the compatibility backend, not
     the peak-performance backend.
2. **Thin abstractions over fast-moving dependencies.** faer ships breaking releases every 4–5 months;
   wgpu quarterly. Every external kernel provider is wrapped behind a narrow internal trait so version
   bumps stay localized.
3. **Runtime dynamism, compile-time kernels.** Shapes and dtypes are runtime values (Python users
   expect this). Kernels are generic Rust monomorphized per dtype, bridged by an `enum DType` +
   dispatch macro. Type-level shapes (const generics, dfdx-style) are rejected — that road is dead.
4. **Data lives in Rust.** Python holds opaque handles. The FFI boundary passes scalars, shapes, and
   (only at the edges) zero-copy NumPy views. Per-op FFI calls are fine (~tens of ns, two orders of
   magnitude below per-op framework overhead); per-element conversion is the thing that kills.
5. **Free-threading-ready from day one.** No global mutable state without synchronization; the
   extension module builds with `gil_used = false` (PyO3 0.28+ default).
6. **Tests are the contract.** The existing Python test suite (32 files) defines behavioral parity for
   the migration. The Rust engine must pass it (modulo documented, intentional changes).
7. **AI-leveraged development.** Implementation throughput is amplified by AI assistance with human
   review by the maintainer. Plans below are therefore sequenced by dependency, not by effort estimates.

---

## 3. Technology Decisions

| Area | Decision | Rejected alternatives |
|---|---|---|
| Language | Rust (edition 2024, latest stable toolchain) | — |
| ND tensor | **Custom strided tensor** (runtime `shape: Vec<usize>` + `strides`, stride-0 broadcasting, zero-copy views) | ndarray (foundation role), nalgebra, mdarray (too young) |
| CPU GEMM / linalg | **faer** (0.24+, behind a thin internal trait) | hand-written GEMM, ndarray-linalg (LAPACK linking pain), nalgebra (no parallelism / no blocked kernels) |
| CPU parallelism | **rayon** (feature-gated; off for `wasm32`) | — |
| CPU SIMD (elementwise) | Autovectorization first; **pulp** where explicit SIMD pays off | std::simd (still nightly-only), wide (no runtime dispatch) |
| dtype model | `enum DType { F16, BF16, F32, F64, I32, I64, Bool }` + generic kernels + dispatch macro | dtype as public generic parameter (forces enum at the Python boundary anyway) |
| Autograd | **Define-by-run tape in Rust**, arena/index-based graph (`NodeId`), no lifetime-linked graph references | graph kept in Python (works, but Rust graph is required for the WASM/Web story) |
| GPU strategy | **Hybrid** (llama.cpp / MLX pattern): native Metal + native CUDA + portable wgpu | wgpu-only (perf ceiling), all-native incl. ROCm (no mature safe HIP wrapper in 2026), CubeCL (lockstep with burn; conflicts with own-the-stack) |
| Metal | **objc2-metal** (metal-rs is deprecated); MSL kernels embedded + runtime JIT + pipeline cache; GEMM via **MPS first**, evaluate MLX-derived MSL kernels if MPS limits us | metal-rs, mlx-rs |
| CUDA | **cudarc**; GEMM via **cuBLAS/cuBLASLt**; custom elementwise kernels as CUDA C → PTX embedded; conv may adopt cuDNN | Rust-CUDA / cuda-oxide (both alpha as of 2026) |
| AMD | **Via the wgpu (Vulkan) backend** — llama.cpp benchmarks show Vulkan ≈ ROCm-native on RDNA3 for many workloads. Native HIP backend reconsidered when the Rust ROCm ecosystem matures | rocm-rs / cubecl-hip-sys raw FFI (Linux-only, immature) |
| Portable GPU / Web | **wgpu** (WGSL kernels) — covers AMD, Windows, Intel, and browsers (WebGPU shipped in Chrome/Edge 113+, Safari 26, Firefox 141+ — Windows first, macOS from 145–147; Firefox Linux still pending) | Vulkan via ash + MoltenVK (loses coopmat on macOS; ash release stagnation) |
| Python bindings | **PyO3 (0.28+) + rust-numpy** (zero-copy NumPy views at the boundary) | HPy (dormant), rust-cpython (dead), UniFFI (no array story) |
| Packaging | **maturin**, mixed layout: pure-Python package `marquetry` + extension `marquetry._native`; **uv** for dev workflow | setuptools-rust |
| Wheels | abi3 (`cp311-abi3`) + `cp314t` free-threaded wheels. **abi3t** (PEP 803, targets Python 3.15) is a future upgrade, conditional on PyO3 0.29 / maturin 1.14 — both unreleased as of 2026-06; the strategy does not depend on it | per-version wheels only |
| Python support | **Python ≥ 3.11** (3.10 EOL 2026-10; abi3 buffer protocol needs 3.11+) | keeping 3.10 |
| Repo layout | **Monorepo**, Cargo workspace in this repository | separate engine repo (pydantic-core retreated from this in 2026) |
| Versioning | **Lockstep, single tag** (safetensors style); Rust crates not published to crates.io initially | independent rs-/py- versioning (polars style — only useful when the Rust API is a public product) |

### 3.1 Numerical core details

- **Storage:** ref-counted buffers (`Arc`) per device; tensors are (buffer, dtype, shape, strides, offset).
  Views (transpose / slice / broadcast / reshape-when-contiguous) are zero-copy.
- **Type promotion:** NumPy NEP 50 semantics (matching the v0.3 Python behavior on NumPy 2).
- **F16/BF16 are in the dtype enum and storage design from day one** because GPU performance depends
  on them. CPU kernels may initially compute via f32 upcast.
- **Operator set:** a small primitive core (elementwise unary/binary, reductions, matmul, index/gather,
  shape ops, compare/select) with composite ops (softmax, layer-norm, GELU, …) defined by composition
  first, fused later only where profiling justifies it.
- **Convolution:** im2col + GEMM everywhere initially (delegates the hard part to GEMM libraries,
  consistent with Principle #1). cuDNN (via cudarc) is the upgrade path on CUDA; MPSCNN on Metal.

### 3.2 Backend abstraction

Modeled on the proven candle shape (~30 storage methods + ~13 device methods), with enum dispatch
(not trait objects) across backends:

```text
marquetry-core
└── Backend (enum dispatch: Cpu | Metal | Cuda | Wgpu)
    ├── CPU    — faer GEMM + rayon + own elementwise/reduction kernels
    ├── Metal  — objc2-metal, MSL JIT + pipeline cache, MPS GEMM      (primary dev target)
    ├── CUDA   — cudarc, cuBLAS GEMM, CUDA C → PTX kernels
    └── Wgpu   — WGSL kernels; AMD / Windows / Intel / WebGPU (browser)
```

**Backend capability matrix (planned):**

| | F16 | BF16 | F32 | F64 | notes |
|---|---|---|---|---|---|
| CPU | ✓ (upcast at first) | ✓ (upcast at first) | ✓ | ✓ | faer GEMM f32/f64 |
| Metal | ✓ | ✓ | ✓ | ✗ (MSL has no f64) | f64 raises explicit unsupported-dtype error (PyTorch MPS precedent) |
| CUDA | ✓ | ✓ | ✓ | ✓ | |
| Wgpu | feature-detect (`shader-f16`) | ✗ | ✓ | ✗ (no f64 in WGSL) | browser limits apply (buffer sizes etc.) |

- **GPU memory management:** the industry-convergent pattern — ref-counted buffer handles,
  per-stream/command-buffer pools (size-bucketed caching allocator), and in-flight work retaining
  buffer references so user-side drop is just "return to pool".
- **Kernel compilation caching:** compiled pipelines/modules cached in the device handle behind an
  `RwLock` map (candle pattern), keyed by (source, entry point, specialization constants).

### 3.3 Python boundary

- `marquetry` stays a Python package; `marquetry._native` is the PyO3 extension. Public API
  (`Container`, functions, layers, optimizers, …) is preserved as closely as practical; the test
  suite defines parity. Intentional breaks are allowed only at the 1.0 boundary and must be documented.
- NumPy in/out via rust-numpy zero-copy views; everything else stays device-side behind handles.
- DLPack interop (PyTorch/JAX/CuPy exchange) is a post-1.0 add-on (e.g. `dlpark`).

### 3.4 Repository layout (target)

```text
Marquetry/
├── crates/
│   ├── marquetry-core/     # tensor, dtype, autograd, backends (pure Rust, wasm32-compatible)
│   ├── marquetry-kernels/  # MSL / CUDA C / WGSL kernel sources + build glue
│   └── marquetry-py/       # PyO3 bindings (cdylib → marquetry._native)
├── marquetry/              # pure-Python package (API layer, datasets, ONNX export, …)
├── tests/                  # Python test suite = parity contract
├── pyproject.toml          # maturin build backend, uv-managed dev env
└── REPLACE_TO_RUST_ENGINE.md
```

The pure-Python engine of v0.3 is preserved on the `v0.3` branch and the 0.3.x PyPI line; `main`
becomes the Rust-engine line.

---

## 4. Migration Phases

Sequenced by dependency. Each phase ends with its exit criteria green on CI.

**Phase 0 — Scaffolding.**
Cargo workspace, maturin + uv wiring, CI matrix (Rust tests; Python 3.11–3.14 incl. a free-threaded
job; wheel builds), benchmark harness skeleton.
*Exit: `pip install` of a dev wheel exposes `marquetry._native.hello()` on all platforms.*

**Phase 1 — CPU tensor core.**
Strided tensor + views + broadcasting, DType enum + dispatch macro, primitive op set on CPU
(faer GEMM, rayon), NEP 50 promotion.
*Exit: Rust-side unit tests; NumPy-checked property tests for every primitive op.*

**Phase 2 — Autograd in Rust.**
Arena/index tape, backward for all primitives, generation-ordered traversal (current engine
semantics), `no_backprop_mode` / `test_mode` equivalents.
*Exit: gradient checks (numerical vs analytical) for all ops; MLP trains on spiral dataset in pure Rust.*

**Phase 3 — Python API layer.**
`Container` and `functions/` re-bound onto `_native` handles; zero-copy NumPy boundary;
free-threading-safe handle semantics.
*Exit: existing tensor/function/autograd test files pass against the Rust engine.*

**Phase 4 — Layers, models, optimizers.**
`Layer`/`Model` parameter management, all 11 optimizers (SGD, MomentumSGD, Nesterov, AdaGrad,
AdaDelta, RMSProp, Adam, AdamW, AdaMax, Nadam, Lion — as Rust kernels; optimizer steps are
hot loops), Conv2D/pooling (im2col+GEMM), recurrent layers, normalization layers.
*Exit: full v0.3 test suite passes; Fashion-MNIST CNN + LSTM samples train end-to-end.*

**Phase 5 — Metal backend** *(first GPU target — primary dev machine).*
objc2-metal device/queue/pool plumbing, MSL elementwise/reduction kernels, MPS GEMM,
buffer pooling, pipeline cache, dtype capability gating (no f64).
*Exit: GPU parity tests vs CPU (tolerance-based) for all ops; CNN training on Metal beats CPU.*

**Phase 6 — CUDA backend.**
cudarc plumbing, cuBLAS GEMM, PTX kernel pipeline (build.rs / cudaforge), stream-ordered pooling.
*Exit: same parity suite green on CUDA; CI strategy documented (self-hosted or vendor CI).*

**Phase 7 — wgpu backend (AMD / Windows / Intel / Web).**
WGSL kernel set; GEMM via a kernel **generator** (workgroup tiling + register blocking, specialization
per dtype/tile config, autotuned — following burn's published multiplatform-matmul techniques and the
TFJS/ORT WebGPU kernel designs, not a naive shader); buffer-limit handling, readback ergonomics;
`wasm32` build of marquetry-core + browser smoke test.
*Exit: parity suite green on Vulkan (AMD) and Metal-via-wgpu; a demo training loop runs in a browser.*

**Phase 8 — Ecosystem features & 1.0.**
Model archive (`.mq`) format vN+1 over the Rust graph, ONNX export adapted (stays in Python,
introspecting the Rust graph), classic ML (trees / random forest / SVM) re-implemented in Rust,
docs refresh (per roadmap), benchmark publication.
*Exit: v1.0.0 — Rust engine at full feature parity, Metal + CUDA + wgpu shipped, wheels
(abi3 + cp31Xt, abi3t when toolchain allows) on PyPI.*

**Cross-cutting (all phases):** benchmark suite tracked in CI — target ≥10× v0.3 pure-Python on
CPU training workloads and same-order-of-magnitude as NumPy for elementwise ops; performance
regressions gate merges once baselines exist.

---

## 5. Rejected Alternatives — Decision Log

| Option | Verdict | Reason (as of 2026-06) |
|---|---|---|
| candle / burn / tch / tract as foundation | Rejected | Marquetry would become a wrapper; contradicts project philosophy |
| nalgebra as numerical base | Rejected | No internal parallelism, no blocked decompositions, geometry-oriented; not built for large dynamic matrices |
| ndarray as the tensor foundation | Rejected (kept as reference) | Healthy again (0.17.x), but owning the strided layer is the point of this project; faer interop lag (faer-ext) adds friction |
| dfdx-style const-generic shapes | Rejected | Incompatible with a dynamic Python frontend; the approach's flagship (dfdx) is still pre-alpha and dormant — last release 2023-07, last commit activity 2024-07, none since |
| CubeCL for GPU kernels | Rejected | Technically excellent (cuBLAS-level matmul) but single-vendor governance, lockstep with burn releases |
| Native ROCm/HIP backend now | Deferred | No mature safe Rust wrapper (Linux-only raw FFI); Vulkan path delivers comparable performance on consumer RDNA3 |
| ash + MoltenVK instead of wgpu | Rejected | Loses cooperative-matrix on macOS, ash crates.io stagnation, no browser story |
| Rust-written GPU kernels (Rust-CUDA, cuda-oxide, rust-gpu) | Deferred | All alpha in 2026; revisit as they mature |
| HPy / rust-cpython / UniFFI | Rejected | Dormant / deprecated / no array support — PyO3 is the unambiguous standard |
| Graph kept in Python + Rust kernels only | Rejected | Viable per FFI measurements, but a Rust-side graph is required for the WASM/Web roadmap and locks in the performance story |

---

## 6. Open Questions (tracked, not blocking)

1. **Metal GEMM:** MPS first vs MLX-derived MSL kernels — decide with benchmarks in Phase 5.
2. **Higher-order differentiation** (grad-of-grad): the tape design must not preclude it; scheduling TBD.
3. **CUDA CI:** self-hosted runner vs cloud GPU CI for Phase 6 exit criteria.
4. **Native HIP backend:** revisit once a cudarc-class safe wrapper exists for ROCm.
5. **abi3t adoption timing:** waiting on PyO3 0.29 + maturin 1.14 releases.
6. **wgpu cooperative-matrix:** experimental in wgpu v29 (Vulkan/Metal targets only, API unstable) —
   adopt in the WGSL GEMM generator once stabilized.
7. **Crates.io publication** of marquetry-core as a standalone Rust library: post-1.0 decision.

---

## 7. Key References

- faer — https://codeberg.org/sarah-quinones/faer (0.24, 2026-01)
- PyO3 free-threading guide — https://pyo3.rs/latest/free-threading.html
- PEP 803 (abi3t, accepted 2026-03) — https://peps.python.org/pep-0803/
- maturin mixed-layout projects — https://www.maturin.rs/project_layout
- objc2-metal — https://crates.io/crates/objc2-metal (metal-rs deprecation: https://github.com/gfx-rs/metal-rs)
- cudarc — https://github.com/coreylowman/cudarc
- llama.cpp Vulkan vs CUDA/ROCm benchmarks (2026-04) — https://knightli.com/en/2026/04/23/llama-cpp-gpu-benchmark-cuda-rocm-vulkan-scoreboard/
- WebGPU browser availability — https://web.dev/blog/webgpu-supported-major-browsers
- WebGPU dispatch overhead study — https://arxiv.org/abs/2604.02344
- Architecture precedents: candle backend traits, polars workspace layout, safetensors release flow,
  tinygrad runtime docs, MLX (Metal-native + CUDA addition, 2025-07)
