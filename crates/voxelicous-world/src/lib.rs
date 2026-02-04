//! Clipmap world generation and streaming for the Voxelicous engine.

pub mod clipmap_streaming;
pub mod generation;

pub use clipmap_streaming::{ClipmapDirtyState, ClipmapStreamingController};
pub use generation::{TerrainConfig, TerrainGenerator};

/// World seed for procedural generation.
pub type WorldSeed = u64;
