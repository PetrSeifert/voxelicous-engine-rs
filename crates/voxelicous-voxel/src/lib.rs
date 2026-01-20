//! Voxel data structures for the Voxelicous engine.
//!
//! This crate provides:
//! - Sparse Voxel Octree (SVO) implementation
//! - DAG compression for memory efficiency
//! - GPU-friendly buffer formats for ray traversal

pub mod dag;
pub mod svo;

pub use dag::SvoDag;
pub use svo::{OctreeNode, SparseVoxelOctree};

use voxelicous_core::types::BlockId;

/// Trait for voxel storage structures.
pub trait VoxelStorage {
    /// Get the voxel at the given position.
    fn get(&self, x: u32, y: u32, z: u32) -> BlockId;

    /// Set the voxel at the given position.
    fn set(&mut self, x: u32, y: u32, z: u32, block: BlockId);

    /// Get the depth/resolution of the structure.
    fn depth(&self) -> u32;

    /// Get the size in voxels per axis (2^depth).
    fn size(&self) -> u32 {
        1 << self.depth()
    }

    /// Check if the structure is empty (all air).
    fn is_empty(&self) -> bool;

    /// Get memory usage in bytes.
    fn memory_usage(&self) -> usize;
}

/// GPU-uploadable format for SVO-DAG.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuOctreeNode {
    /// Child pointers or leaf data.
    /// For branch nodes: indices to child nodes (0 = no child).
    /// For leaf nodes: material/block IDs packed.
    pub children: [u32; 8],
    /// Node flags and metadata.
    /// Bits 0-7: valid child mask
    /// Bits 8-15: leaf mask (which children are leaves)
    /// Bits 16-31: reserved
    pub flags: u32,
    /// Padding for alignment.
    pub _padding: [u32; 3],
}

impl Default for GpuOctreeNode {
    fn default() -> Self {
        Self {
            children: [0; 8],
            flags: 0,
            _padding: [0; 3],
        }
    }
}

impl GpuOctreeNode {
    /// Create a new empty node.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the valid child mask.
    pub fn valid_mask(&self) -> u8 {
        (self.flags & 0xFF) as u8
    }

    /// Get the leaf mask.
    pub fn leaf_mask(&self) -> u8 {
        ((self.flags >> 8) & 0xFF) as u8
    }

    /// Check if a child is valid.
    pub fn has_child(&self, index: usize) -> bool {
        (self.valid_mask() & (1 << index)) != 0
    }

    /// Check if a child is a leaf.
    pub fn is_leaf(&self, index: usize) -> bool {
        (self.leaf_mask() & (1 << index)) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_node_size() {
        // Ensure GPU node is 48 bytes (good alignment)
        assert_eq!(std::mem::size_of::<GpuOctreeNode>(), 48);
    }

    #[test]
    fn gpu_node_masks() {
        let mut node = GpuOctreeNode::new();
        node.flags = 0b1010_0101_1100_0011; // leaf mask = 0xA5, valid mask = 0xC3

        assert_eq!(node.valid_mask(), 0xC3);
        assert_eq!(node.leaf_mask(), 0xA5);
        assert!(node.has_child(0));
        assert!(node.has_child(1));
        assert!(!node.has_child(2));
    }
}
