## Project Overview

Voxelicous Engine is a Rust + Vulkan voxel game engine using **pure voxel ray marching** with no mesh generation. The current rendering path is clipmap-based compute ray marching. Supports Windows, Linux, and macOS.

**Rendering Paths:**
- **Compute Ray Marching (Clipmap)**: Active rework path used by the current viewer

Rework status and known issues are tracked in `clipmap_rework_status.md`.

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
├── voxelicous-voxel (clipmap storage/compression/GPU format)
├── voxelicous-gpu (Vulkan abstraction via ash, memory via gpu-allocator)
│   └── voxelicous-render (clipmap compute ray marching pipeline)
│       └── voxelicous-test (headless harness)
├── voxelicous-platform (windowing via winit)
├── voxelicous-world (clipmap streaming, terrain generation)
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
