//! Sparse Voxel Octree implementation.
//!
//! The SVO is the base data structure for voxel storage before DAG compression.

use crate::VoxelStorage;
use voxelicous_core::types::BlockId;

/// A node in the sparse voxel octree.
#[derive(Clone, Debug)]
pub enum OctreeNode {
    /// Empty node (all voxels are air).
    Empty,
    /// Leaf node with a single block type.
    Leaf(BlockId),
    /// Branch node with 8 children.
    Branch(Box<[OctreeNode; 8]>),
}

impl Default for OctreeNode {
    fn default() -> Self {
        Self::Empty
    }
}

impl OctreeNode {
    /// Create a new empty node.
    pub fn new() -> Self {
        Self::Empty
    }

    /// Create a branch node with all empty children.
    pub fn new_branch() -> Self {
        Self::Branch(Box::new([
            Self::Empty,
            Self::Empty,
            Self::Empty,
            Self::Empty,
            Self::Empty,
            Self::Empty,
            Self::Empty,
            Self::Empty,
        ]))
    }

    /// Check if the node is empty.
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    /// Check if the node is a branch.
    pub fn is_branch(&self) -> bool {
        matches!(self, Self::Branch(_))
    }

    /// Get the block ID if this is a leaf node.
    pub fn as_leaf(&self) -> Option<BlockId> {
        match self {
            Self::Leaf(id) => Some(*id),
            _ => None,
        }
    }

    /// Get children if this is a branch node.
    pub fn as_branch(&self) -> Option<&[OctreeNode; 8]> {
        match self {
            Self::Branch(children) => Some(children),
            _ => None,
        }
    }

    /// Get mutable children if this is a branch node.
    pub fn as_branch_mut(&mut self) -> Option<&mut [OctreeNode; 8]> {
        match self {
            Self::Branch(children) => Some(children),
            _ => None,
        }
    }

    /// Count total nodes in this subtree.
    pub fn count_nodes(&self) -> usize {
        match self {
            Self::Empty | Self::Leaf(_) => 1,
            Self::Branch(children) => {
                1 + children.iter().map(OctreeNode::count_nodes).sum::<usize>()
            }
        }
    }

    /// Calculate memory usage of this subtree.
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::Empty => std::mem::size_of::<Self>(),
            Self::Leaf(_) => std::mem::size_of::<Self>(),
            Self::Branch(children) => {
                std::mem::size_of::<Self>()
                    + std::mem::size_of::<[OctreeNode; 8]>()
                    + children.iter().map(OctreeNode::memory_usage).sum::<usize>()
            }
        }
    }
}

/// Sparse Voxel Octree for voxel storage.
#[derive(Clone, Debug)]
pub struct SparseVoxelOctree {
    root: OctreeNode,
    depth: u32,
}

impl SparseVoxelOctree {
    /// Create a new empty octree with the given depth.
    /// Size will be 2^depth voxels per axis.
    pub fn new(depth: u32) -> Self {
        Self {
            root: OctreeNode::Empty,
            depth,
        }
    }

    /// Get the root node.
    pub fn root(&self) -> &OctreeNode {
        &self.root
    }

    /// Get the child index for a position at a given level.
    fn child_index(x: u32, y: u32, z: u32, level: u32) -> usize {
        let bit = 1 << level;
        let xi = ((x & bit) != 0) as usize;
        let yi = ((y & bit) != 0) as usize;
        let zi = ((z & bit) != 0) as usize;
        xi | (yi << 1) | (zi << 2)
    }

    /// Get voxel at position recursively.
    fn get_recursive(node: &OctreeNode, x: u32, y: u32, z: u32, level: u32) -> BlockId {
        match node {
            OctreeNode::Empty => BlockId::AIR,
            OctreeNode::Leaf(id) => *id,
            OctreeNode::Branch(children) => {
                if level == 0 {
                    // Should not happen in well-formed octree
                    BlockId::AIR
                } else {
                    let idx = Self::child_index(x, y, z, level - 1);
                    Self::get_recursive(&children[idx], x, y, z, level - 1)
                }
            }
        }
    }

    /// Set voxel at position recursively.
    fn set_recursive(node: &mut OctreeNode, x: u32, y: u32, z: u32, level: u32, block: BlockId) {
        if level == 0 {
            // At deepest level, set the leaf
            *node = if block == BlockId::AIR {
                OctreeNode::Empty
            } else {
                OctreeNode::Leaf(block)
            };
            return;
        }

        // Ensure we have a branch node
        match node {
            OctreeNode::Branch(children) => {
                let idx = Self::child_index(x, y, z, level - 1);
                Self::set_recursive(&mut children[idx], x, y, z, level - 1, block);
            }
            OctreeNode::Empty | OctreeNode::Leaf(_) => {
                // Need to expand to branch
                let old_value = match node {
                    OctreeNode::Leaf(id) => *id,
                    _ => BlockId::AIR,
                };

                // Create branch with old value in all children
                let mut children = Box::new([
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                    if old_value == BlockId::AIR {
                        OctreeNode::Empty
                    } else {
                        OctreeNode::Leaf(old_value)
                    },
                ]);

                let idx = Self::child_index(x, y, z, level - 1);
                Self::set_recursive(&mut children[idx], x, y, z, level - 1, block);

                *node = OctreeNode::Branch(children);
            }
        }

        // Try to collapse branch if all children are the same
        Self::try_collapse(node);
    }

    /// Try to collapse a branch node if all children are the same.
    fn try_collapse(node: &mut OctreeNode) {
        let children = match node {
            OctreeNode::Branch(children) => children,
            _ => return,
        };

        // Check if all children are the same
        let first = &children[0];
        let all_same = children[1..].iter().all(|c| Self::nodes_equal(c, first));

        if all_same {
            *node = match first {
                OctreeNode::Empty => OctreeNode::Empty,
                OctreeNode::Leaf(id) => OctreeNode::Leaf(*id),
                OctreeNode::Branch(_) => return, // Don't collapse branches of branches
            };
        }
    }

    /// Check if two nodes are equal (for collapsing).
    fn nodes_equal(a: &OctreeNode, b: &OctreeNode) -> bool {
        match (a, b) {
            (OctreeNode::Empty, OctreeNode::Empty) => true,
            (OctreeNode::Leaf(id_a), OctreeNode::Leaf(id_b)) => id_a == id_b,
            _ => false,
        }
    }

    /// Fill a region with a block type.
    pub fn fill(&mut self, min: [u32; 3], max: [u32; 3], block: BlockId) {
        let size = self.size();
        for z in min[2].min(size)..max[2].min(size) {
            for y in min[1].min(size)..max[1].min(size) {
                for x in min[0].min(size)..max[0].min(size) {
                    self.set(x, y, z, block);
                }
            }
        }
    }

    /// Create a sphere of voxels.
    pub fn fill_sphere(&mut self, center: [f32; 3], radius: f32, block: BlockId) {
        let size = self.size();
        let r2 = radius * radius;

        let min_x = ((center[0] - radius).floor() as u32).max(0);
        let max_x = ((center[0] + radius).ceil() as u32).min(size);
        let min_y = ((center[1] - radius).floor() as u32).max(0);
        let max_y = ((center[1] + radius).ceil() as u32).min(size);
        let min_z = ((center[2] - radius).floor() as u32).max(0);
        let max_z = ((center[2] + radius).ceil() as u32).min(size);

        for z in min_z..max_z {
            for y in min_y..max_y {
                for x in min_x..max_x {
                    let dx = x as f32 + 0.5 - center[0];
                    let dy = y as f32 + 0.5 - center[1];
                    let dz = z as f32 + 0.5 - center[2];
                    if dx * dx + dy * dy + dz * dz <= r2 {
                        self.set(x, y, z, block);
                    }
                }
            }
        }
    }
}

impl VoxelStorage for SparseVoxelOctree {
    fn get(&self, x: u32, y: u32, z: u32) -> BlockId {
        let size = self.size();
        if x >= size || y >= size || z >= size {
            return BlockId::AIR;
        }
        Self::get_recursive(&self.root, x, y, z, self.depth)
    }

    fn set(&mut self, x: u32, y: u32, z: u32, block: BlockId) {
        let size = self.size();
        if x >= size || y >= size || z >= size {
            return;
        }
        Self::set_recursive(&mut self.root, x, y, z, self.depth, block);
    }

    fn depth(&self) -> u32 {
        self.depth
    }

    fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.root.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_octree() {
        let octree = SparseVoxelOctree::new(5); // 32x32x32
        assert!(octree.is_empty());
        assert_eq!(octree.size(), 32);
        assert_eq!(octree.get(0, 0, 0), BlockId::AIR);
    }

    #[test]
    fn set_and_get_single_voxel() {
        let mut octree = SparseVoxelOctree::new(5);
        let block = BlockId(42);

        octree.set(10, 15, 20, block);
        assert_eq!(octree.get(10, 15, 20), block);
        assert_eq!(octree.get(0, 0, 0), BlockId::AIR);
        assert!(!octree.is_empty());
    }

    #[test]
    fn set_and_get_corners() {
        let mut octree = SparseVoxelOctree::new(3); // 8x8x8
        let block = BlockId(1);

        // Set all corners
        octree.set(0, 0, 0, block);
        octree.set(7, 0, 0, block);
        octree.set(0, 7, 0, block);
        octree.set(7, 7, 0, block);
        octree.set(0, 0, 7, block);
        octree.set(7, 0, 7, block);
        octree.set(0, 7, 7, block);
        octree.set(7, 7, 7, block);

        // Verify corners
        assert_eq!(octree.get(0, 0, 0), block);
        assert_eq!(octree.get(7, 7, 7), block);

        // Verify center is empty
        assert_eq!(octree.get(4, 4, 4), BlockId::AIR);
    }

    #[test]
    fn collapse_uniform_region() {
        let mut octree = SparseVoxelOctree::new(2); // 4x4x4
        let block = BlockId(1);

        // Fill entire octree
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    octree.set(x, y, z, block);
                }
            }
        }

        // Should collapse to a single leaf
        assert!(octree.root().is_leaf());
        assert_eq!(octree.root().as_leaf(), Some(block));
    }

    #[test]
    fn out_of_bounds_access() {
        let mut octree = SparseVoxelOctree::new(3); // 8x8x8

        // Out of bounds get returns AIR
        assert_eq!(octree.get(100, 100, 100), BlockId::AIR);

        // Out of bounds set is a no-op
        octree.set(100, 100, 100, BlockId(1));
        assert!(octree.is_empty());
    }

    #[test]
    fn child_index_calculation() {
        // Level 0: bottom bits
        assert_eq!(SparseVoxelOctree::child_index(0, 0, 0, 0), 0b000);
        assert_eq!(SparseVoxelOctree::child_index(1, 0, 0, 0), 0b001);
        assert_eq!(SparseVoxelOctree::child_index(0, 1, 0, 0), 0b010);
        assert_eq!(SparseVoxelOctree::child_index(1, 1, 0, 0), 0b011);
        assert_eq!(SparseVoxelOctree::child_index(0, 0, 1, 0), 0b100);
        assert_eq!(SparseVoxelOctree::child_index(1, 1, 1, 0), 0b111);

        // Level 1: second bits
        assert_eq!(SparseVoxelOctree::child_index(0, 0, 0, 1), 0b000);
        assert_eq!(SparseVoxelOctree::child_index(2, 0, 0, 1), 0b001);
        assert_eq!(SparseVoxelOctree::child_index(0, 2, 0, 1), 0b010);
        assert_eq!(SparseVoxelOctree::child_index(0, 0, 2, 1), 0b100);
    }
}
