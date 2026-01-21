//! Sparse Voxel Octree DAG compression.
//!
//! DAG (Directed Acyclic Graph) compression merges identical subtrees
//! to significantly reduce memory usage while maintaining fast traversal.

use crate::svo::{OctreeNode, SparseVoxelOctree};
use crate::{GpuOctreeNode, VoxelStorage};
use hashbrown::HashMap;
use std::hash::{Hash, Hasher};
use voxelicous_core::types::BlockId;

/// A node in the compressed DAG.
#[derive(Clone, Debug)]
pub struct DagNode {
    /// Child indices (0 = no child/empty).
    pub children: [u32; 8],
    /// Valid child mask.
    pub valid_mask: u8,
    /// Leaf mask (which valid children are leaves).
    pub leaf_mask: u8,
    /// For leaf children, the block IDs.
    pub leaf_data: [BlockId; 8],
}

impl Default for DagNode {
    fn default() -> Self {
        Self {
            children: [0; 8],
            valid_mask: 0,
            leaf_mask: 0,
            leaf_data: [BlockId::AIR; 8],
        }
    }
}

impl DagNode {
    /// Check if a child slot is valid.
    pub fn has_child(&self, index: usize) -> bool {
        (self.valid_mask & (1 << index)) != 0
    }

    /// Check if a child slot is a leaf.
    pub fn is_leaf(&self, index: usize) -> bool {
        (self.leaf_mask & (1 << index)) != 0
    }

    /// Convert to GPU format.
    pub fn to_gpu(&self) -> GpuOctreeNode {
        let mut gpu = GpuOctreeNode::new();
        gpu.children = self.children;
        gpu.flags = (self.valid_mask as u32) | ((self.leaf_mask as u32) << 8);

        // Pack leaf data into children for leaf nodes
        for i in 0..8 {
            if self.is_leaf(i) {
                gpu.children[i] = self.leaf_data[i].0 as u32;
            }
        }

        gpu
    }
}

/// Hash key for DAG node deduplication.
#[derive(Clone, PartialEq, Eq)]
struct NodeKey {
    children: [u32; 8],
    valid_mask: u8,
    leaf_mask: u8,
    leaf_data: [u16; 8],
}

impl Hash for NodeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.children.hash(state);
        self.valid_mask.hash(state);
        self.leaf_mask.hash(state);
        self.leaf_data.hash(state);
    }
}

impl From<&DagNode> for NodeKey {
    fn from(node: &DagNode) -> Self {
        Self {
            children: node.children,
            valid_mask: node.valid_mask,
            leaf_mask: node.leaf_mask,
            leaf_data: [
                node.leaf_data[0].0,
                node.leaf_data[1].0,
                node.leaf_data[2].0,
                node.leaf_data[3].0,
                node.leaf_data[4].0,
                node.leaf_data[5].0,
                node.leaf_data[6].0,
                node.leaf_data[7].0,
            ],
        }
    }
}

/// Compressed Sparse Voxel Octree DAG.
#[derive(Clone)]
pub struct SvoDag {
    /// Node pool.
    nodes: Vec<DagNode>,
    /// Root node index.
    root: u32,
    /// Octree depth.
    depth: u32,
    /// Deduplication map (for incremental updates).
    dedup_map: HashMap<NodeKey, u32>,
}

impl SvoDag {
    /// Create a new empty DAG.
    pub fn new(depth: u32) -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
            depth,
            dedup_map: HashMap::new(),
        }
    }

    /// Compress an SVO into a DAG.
    pub fn from_svo(svo: &SparseVoxelOctree) -> Self {
        let mut dag = Self::new(svo.depth());
        dag.root = dag.compress_node(svo.root(), svo.depth());
        dag
    }

    /// Compress a node recursively.
    fn compress_node(&mut self, node: &OctreeNode, level: u32) -> u32 {
        match node {
            OctreeNode::Empty => 0, // Empty nodes are represented by index 0
            OctreeNode::Leaf(id) => {
                // Create a node with all leaf children of the same type
                let mut dag_node = DagNode::default();
                dag_node.valid_mask = 0xFF;
                dag_node.leaf_mask = 0xFF;
                for i in 0..8 {
                    dag_node.leaf_data[i] = *id;
                }
                self.add_or_reuse_node(dag_node)
            }
            OctreeNode::Branch(children) => {
                let mut dag_node = DagNode::default();

                for (i, child) in children.iter().enumerate() {
                    match child {
                        OctreeNode::Empty => {
                            // Child is empty, nothing to do
                        }
                        OctreeNode::Leaf(id) => {
                            dag_node.valid_mask |= 1 << i;
                            dag_node.leaf_mask |= 1 << i;
                            dag_node.leaf_data[i] = *id;
                        }
                        OctreeNode::Branch(_) => {
                            dag_node.valid_mask |= 1 << i;
                            // Recursively compress child
                            let child_idx = self.compress_node(child, level - 1);
                            dag_node.children[i] = child_idx;
                        }
                    }
                }

                self.add_or_reuse_node(dag_node)
            }
        }
    }

    /// Add a node or reuse an existing identical one.
    fn add_or_reuse_node(&mut self, node: DagNode) -> u32 {
        let key = NodeKey::from(&node);

        if let Some(&existing) = self.dedup_map.get(&key) {
            return existing;
        }

        // Index 0 is reserved for empty
        let index = self.nodes.len() as u32 + 1;
        self.nodes.push(node);
        self.dedup_map.insert(key, index);
        index
    }

    /// Get a node by index.
    pub fn get_node(&self, index: u32) -> Option<&DagNode> {
        if index == 0 {
            None
        } else {
            self.nodes.get(index as usize - 1)
        }
    }

    /// Get the root node index.
    pub fn root(&self) -> u32 {
        self.root
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get compression ratio compared to full octree.
    pub fn compression_ratio(&self, original_nodes: usize) -> f64 {
        if original_nodes == 0 {
            1.0
        } else {
            self.nodes.len() as f64 / original_nodes as f64
        }
    }

    /// Convert to GPU buffer format.
    pub fn to_gpu_buffer(&self) -> Vec<GpuOctreeNode> {
        // Add a null node at index 0
        let mut buffer = vec![GpuOctreeNode::default()];
        buffer.extend(self.nodes.iter().map(DagNode::to_gpu));
        buffer
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.nodes.len() * std::mem::size_of::<DagNode>()
            + self.dedup_map.len() * (std::mem::size_of::<NodeKey>() + std::mem::size_of::<u32>())
    }

    /// Get GPU buffer size in bytes.
    pub fn gpu_buffer_size(&self) -> usize {
        (self.nodes.len() + 1) * std::mem::size_of::<GpuOctreeNode>()
    }
}

impl VoxelStorage for SvoDag {
    fn get(&self, x: u32, y: u32, z: u32) -> BlockId {
        let size = self.size();
        if x >= size || y >= size || z >= size {
            return BlockId::AIR;
        }

        self.get_recursive(self.root, x, y, z, self.depth)
    }

    fn set(&mut self, _x: u32, _y: u32, _z: u32, _block: BlockId) {
        // DAG is immutable after creation
        // For modifications, convert back to SVO, modify, and recompress
        unimplemented!("DAG is immutable. Convert to SVO for modifications.");
    }

    fn depth(&self) -> u32 {
        self.depth
    }

    fn is_empty(&self) -> bool {
        self.root == 0
    }

    fn memory_usage(&self) -> usize {
        self.memory_usage()
    }
}

impl SvoDag {
    /// Get voxel at position by traversing the DAG.
    fn get_recursive(&self, node_idx: u32, x: u32, y: u32, z: u32, level: u32) -> BlockId {
        if node_idx == 0 {
            return BlockId::AIR;
        }

        let node = match self.get_node(node_idx) {
            Some(n) => n,
            None => return BlockId::AIR,
        };

        if level == 0 {
            // At the deepest level, return first valid leaf
            for i in 0..8 {
                if node.is_leaf(i) {
                    return node.leaf_data[i];
                }
            }
            return BlockId::AIR;
        }

        // Calculate child index
        let bit = 1 << (level - 1);
        let xi = ((x & bit) != 0) as usize;
        let yi = ((y & bit) != 0) as usize;
        let zi = ((z & bit) != 0) as usize;
        let child_idx = xi | (yi << 1) | (zi << 2);

        if !node.has_child(child_idx) {
            return BlockId::AIR;
        }

        if node.is_leaf(child_idx) {
            return node.leaf_data[child_idx];
        }

        // Recurse into child
        self.get_recursive(node.children[child_idx], x, y, z, level - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dag() {
        let svo = SparseVoxelOctree::new(5);
        let dag = SvoDag::from_svo(&svo);

        assert!(dag.is_empty());
        assert_eq!(dag.node_count(), 0);
    }

    #[test]
    fn single_voxel_dag() {
        let mut svo = SparseVoxelOctree::new(3);
        let block = BlockId(42);
        svo.set(0, 0, 0, block);

        let dag = SvoDag::from_svo(&svo);

        assert!(!dag.is_empty());
        assert_eq!(dag.get(0, 0, 0), block);
        assert_eq!(dag.get(1, 1, 1), BlockId::AIR);
    }

    #[test]
    fn compression_deduplication() {
        let mut svo = SparseVoxelOctree::new(3); // 8x8x8
        let block = BlockId(1);

        // Create symmetric pattern (should deduplicate)
        svo.set(0, 0, 0, block);
        svo.set(7, 7, 7, block);

        let dag = SvoDag::from_svo(&svo);
        let svo_nodes = svo.root().count_nodes();

        // DAG should have fewer nodes due to deduplication
        println!(
            "SVO nodes: {}, DAG nodes: {}, ratio: {:.2}",
            svo_nodes,
            dag.node_count(),
            dag.compression_ratio(svo_nodes)
        );

        // Verify data integrity
        assert_eq!(dag.get(0, 0, 0), block);
        assert_eq!(dag.get(7, 7, 7), block);
        assert_eq!(dag.get(4, 4, 4), BlockId::AIR);
    }

    #[test]
    fn uniform_compression() {
        let mut svo = SparseVoxelOctree::new(4); // 16x16x16
        let block = BlockId(1);

        // Fill entire octree
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    svo.set(x, y, z, block);
                }
            }
        }

        let dag = SvoDag::from_svo(&svo);

        // Uniform octree should compress extremely well
        assert!(dag.node_count() <= 2); // Just root + one leaf pattern

        // Verify data
        assert_eq!(dag.get(8, 8, 8), block);
    }

    #[test]
    fn gpu_buffer_format() {
        let mut svo = SparseVoxelOctree::new(2);
        svo.set(0, 0, 0, BlockId(1));

        let dag = SvoDag::from_svo(&svo);
        let buffer = dag.to_gpu_buffer();

        // Buffer should have null node at index 0
        assert!(buffer.len() > 1);
        assert_eq!(buffer[0].valid_mask(), 0);

        // GPU nodes should be 48 bytes each
        assert_eq!(dag.gpu_buffer_size() % 48, 0);
    }
}
