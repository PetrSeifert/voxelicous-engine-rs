## Project Overview

Voxelicous Engine is a Rust + Vulkan voxel game engine using **pure voxel ray tracing** - rays traverse SVO-DAG (Sparse Voxel Octree - Directed Acyclic Graph) directly with no mesh generation. Supports Windows, Linux, and macOS.

**Rendering Paths:**
- **Hardware Ray Tracing**: RTX GPUs via `VK_KHR_ray_tracing_pipeline` (10-100x faster)
- **Compute Ray Marching**: Fallback for all Vulkan 1.3 GPUs

## Build Commands

```bash
# Build entire workspace
cargo build --workspace

# Build release (with LTO)
cargo build --workspace --release

# Run specific applications
cargo run -p voxelicous-viewer        # Demo viewer
cargo run -p voxelicous-editor        # Editor
cargo run -p voxelicous-benchmark --release

# Feature flags
cargo build --features voxelicous-gpu/headless      # No windowing (tests/server)
cargo build --features voxelicous-nvidia/nvidia     # Enable RTX support
```

## Testing

```bash
# All tests
cargo test --workspace

# Single crate
cargo test -p voxelicous-voxel
cargo test -p voxelicous-gpu --features headless

# Benchmarks
cargo bench --workspace
```

## Visual debugging

```bash
# Capture frame 0 only
cargo run -p voxelicous-viewer -- -S

# Capture frames 0, 10, 20 and exit
cargo run -p voxelicous-viewer -- -S -f 0,10,20 --exit-after

# Capture frames 0-5 with custom output pattern
cargo run -p voxelicous-viewer -- -S -o frame_{}.png -f 0-5

# Capture every 10th frame from 0-100
cargo run -p voxelicous-viewer -- -S -f 0,10,20,30,40,50,60,70,80,90,100 -o orbit_{}.png --exit-after
```

## Linting

```bash
cargo fmt --all                    # Format code
cargo fmt --all -- --check         # Check formatting
cargo clippy --workspace           # Lint (configured: all, pedantic, nursery)
```

## Architecture

### Crate Dependency Graph

```
voxelicous-core (foundation: types, math, ECS re-exports)
├── voxelicous-voxel (SVO-DAG storage, compression, GPU format)
├── voxelicous-gpu (Vulkan abstraction via ash, memory via gpu-allocator)
│   └── voxelicous-render (ray tracing pipeline, compute fallback)
│       ├── voxelicous-nvidia (RTX acceleration, DLSS - feature-gated)
│       └── voxelicous-test (headless harness)
├── voxelicous-platform (windowing via winit)
├── voxelicous-world (chunk streaming, terrain generation)
├── voxelicous-physics (collision via rapier3d, raycasting)
├── voxelicous-audio (spatial audio via kira)
├── voxelicous-entity (ECS via hecs, Lua scripting via mlua)
└── voxelicous-shaders (GLSL/HLSL to SPIR-V)
```

## Workspace Lints

Clippy: `all`, `pedantic`, `nursery` warnings enabled. Allowed exceptions:
- `module_name_repetitions`
- `too_many_lines`
- `missing_errors_doc`
- `missing_panics_doc`

Rust: `unsafe_op_in_unsafe_fn = warn`
