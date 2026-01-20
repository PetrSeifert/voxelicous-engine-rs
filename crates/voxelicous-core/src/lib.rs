//! Core types, math, and traits for the Voxelicous engine.
//!
//! This crate provides the foundational types used throughout the engine:
//! - Voxel and block types
//! - Coordinate systems (world, chunk, local)
//! - Math utilities and SIMD helpers
//! - Common traits and error types

pub mod coords;
pub mod error;
pub mod math;
pub mod types;

pub use coords::{ChunkPos, LocalPos, WorldPos};
pub use error::{Error, Result};
pub use types::{BlockId, Material, Voxel};

/// Engine-wide constants
pub mod constants {
    /// Size of a chunk in voxels per axis
    pub const CHUNK_SIZE: usize = 32;
    /// Total voxels in a chunk (32^3)
    pub const CHUNK_SIZE_CUBED: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
    /// Bits needed to represent position within a chunk (5 bits for 0-31)
    pub const CHUNK_BITS: u32 = 5;
    /// Maximum octree depth for chunk-sized data
    pub const OCTREE_DEPTH: u32 = CHUNK_BITS;
}
