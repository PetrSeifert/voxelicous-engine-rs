# Voxelicous Engine

A high-performance voxel game engine written in Rust, featuring **pure voxel ray tracing** that directly traverses Sparse Voxel Octree - Directed Acyclic Graphs (SVO-DAG) without mesh generation.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## 🌟 Key Features

- **Pure Voxel Ray Tracing**: Rays traverse SVO-DAG structures directly
- **Cross-Platform**: Windows, Linux, and macOS support
- **Modular Architecture**: Clean separation of concerns across multiple crates
- **Performance Focused**: Thin LTO, optimized release builds

## 📦 Applications

- **`voxelicous-viewer`**: Demo viewer for exploring voxel worlds
- **`voxelicous-profiler-tui`**: Terminal-based profiling interface

## 🏗️ Architecture

### Core Crates

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
└── voxelicous-shaders (GLSL/HLSL to SPIR-V compilation)
```

## 🚀 Getting Started

### Prerequisites

- **Rust**: 1.77 or later
- **Vulkan SDK**: Required for shader compilation ([Download](https://vulkan.lunarg.com/sdk/home))
  - On Windows: Install Vulkan SDK
  - On Linux/macOS: Install via package manager or build from source with cmake + ninja

### Building

```bash
# Clone the repository
git clone https://github.com/voxelicous/voxelicous-engine-rs.git
cd voxelicous-engine-rs

# Build entire workspace
cargo build --workspace

# Build optimized release (with LTO)
cargo build --workspace --release
```

### Running Applications

```bash
# Demo viewer
cargo run -p voxelicous-viewer

# Profiler TUI
cargo run -p voxelicous-profiler-tui
```

### Feature Flags

```bash
# Enable profiling features
cargo build --features voxelicous-app/profiling

# Headless mode (for testing/servers)
cargo build --features voxelicous-gpu/headless
```

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace

# Test specific crate
cargo test -p voxelicous-voxel

# Run benchmarks
cargo bench --workspace
```

## 📸 Visual Debugging

Capture frames for debugging and documentation:

```bash
# Capture frame 0 only
cargo run -p voxelicous-viewer -- -S

# Capture specific frames and exit
cargo run -p voxelicous-viewer -- -S -f 0,10,20 --exit-after

# Capture range with custom output
cargo run -p voxelicous-viewer -- -S -o frame_{}.png -f 0-5

# Capture every 10th frame (useful for orbits/animations)
cargo run -p voxelicous-viewer -- -S -f 0,10,20,30,40,50,60,70,80,90,100 -o orbit_{}.png --exit-after
```

## 🛠️ Development

### Code Quality

```bash
# Format code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check

# Lint code (all, pedantic, nursery warnings enabled)
cargo clippy --workspace
```

### Workspace Lints

- **Clippy**: `all`, `pedantic`, `nursery` warnings enabled
- **Rust**: `unsafe_op_in_unsafe_fn = warn`
- **Allowed exceptions**: `module_name_repetitions`, `too_many_lines`, `missing_errors_doc`, `missing_panics_doc`

## 📋 Dependencies

### Core Rendering
- **ash**: Vulkan bindings
- **gpu-allocator**: GPU memory management
- **shaderc**: Shader compilation to SPIR-V

### Math & Utils
- **glam**: High-performance math library
- **bytemuck**: Safe casting between types
- **rayon**: Parallel computation

### Game Systems
- **hecs**: Entity Component System
- **rapier3d**: Physics simulation
- **kira**: Spatial audio
- **noise**: Procedural generation

### Platform
- **winit**: Cross-platform windowing
- **tracing**: Structured logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Licensed under ([MIT](LICENSE))

## 🙏 Acknowledgments

- Built with [ash](https://github.com/ash-rs/ash) for Vulkan bindings
