//! Chunk data structure for voxel world storage.

use voxelicous_core::coords::ChunkPos;
use voxelicous_voxel::{SparseVoxelOctree, SvoDag, VoxelStorage};

/// State of a chunk in the loading pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkState {
    /// Chunk is queued for generation.
    Pending,
    /// Chunk is currently being generated.
    Generating,
    /// Chunk has voxel data but no DAG.
    Generated,
    /// Chunk has been compressed to DAG.
    Compressed,
    /// Chunk is uploaded to GPU.
    Uploaded,
    /// Chunk is marked for unload.
    Unloading,
}

impl Default for ChunkState {
    fn default() -> Self {
        Self::Pending
    }
}

/// A single chunk of voxel data (32x32x32 voxels).
pub struct Chunk {
    /// Position in chunk coordinates.
    pub pos: ChunkPos,
    /// Current state in the loading pipeline.
    pub state: ChunkState,
    /// Raw voxel data (SVO) - present during generation/modification.
    pub svo: Option<SparseVoxelOctree>,
    /// Compressed DAG - present after compression.
    pub dag: Option<SvoDag>,
    /// Frame number when last accessed (for LRU eviction).
    pub last_access_frame: u64,
    /// Whether chunk needs GPU re-upload.
    pub dirty: bool,
}

impl Chunk {
    /// Create a new empty chunk at the given position.
    pub fn new(pos: ChunkPos) -> Self {
        Self {
            pos,
            state: ChunkState::Pending,
            svo: None,
            dag: None,
            last_access_frame: 0,
            dirty: true,
        }
    }

    /// Create a chunk with an existing SVO.
    pub fn with_svo(pos: ChunkPos, svo: SparseVoxelOctree) -> Self {
        Self {
            pos,
            state: ChunkState::Generated,
            svo: Some(svo),
            dag: None,
            last_access_frame: 0,
            dirty: true,
        }
    }

    /// Create a chunk with an existing DAG (skips SVO stage).
    ///
    /// This is used for async chunk generation where the worker thread
    /// handles both generation and compression.
    pub fn with_dag(pos: ChunkPos, dag: SvoDag) -> Self {
        Self {
            pos,
            state: ChunkState::Compressed,
            svo: None,
            dag: Some(dag),
            last_access_frame: 0,
            dirty: true,
        }
    }

    /// Check if this chunk is empty (all air).
    pub fn is_empty(&self) -> bool {
        match (&self.svo, &self.dag) {
            (Some(svo), _) => svo.is_empty(),
            (None, Some(dag)) => dag.is_empty(),
            (None, None) => true,
        }
    }

    /// Compress SVO to DAG, dropping the SVO to free memory.
    pub fn compress(&mut self) {
        if let Some(svo) = self.svo.take() {
            self.dag = Some(SvoDag::from_svo(&svo));
            self.state = ChunkState::Compressed;
            self.dirty = true;
        }
    }

    /// Mark chunk as uploaded to GPU.
    pub fn mark_uploaded(&mut self) {
        self.state = ChunkState::Uploaded;
        self.dirty = false;
    }

    /// Update the last access frame for LRU tracking.
    pub fn touch(&mut self, frame: u64) {
        self.last_access_frame = frame;
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let svo_mem = self.svo.as_ref().map_or(0, VoxelStorage::memory_usage);
        let dag_mem = self.dag.as_ref().map_or(0, SvoDag::memory_usage);
        base + svo_mem + dag_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxelicous_core::constants::OCTREE_DEPTH;
    use voxelicous_core::types::BlockId;
    use voxelicous_voxel::VoxelStorage;

    #[test]
    fn new_chunk_is_pending() {
        let chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        assert_eq!(chunk.state, ChunkState::Pending);
        assert!(chunk.is_empty());
        assert!(chunk.dirty);
    }

    #[test]
    fn chunk_with_svo_is_generated() {
        let svo = SparseVoxelOctree::new(OCTREE_DEPTH);
        let chunk = Chunk::with_svo(ChunkPos::new(0, 0, 0), svo);
        assert_eq!(chunk.state, ChunkState::Generated);
    }

    #[test]
    fn chunk_with_dag_is_compressed() {
        let svo = SparseVoxelOctree::new(OCTREE_DEPTH);
        let dag = SvoDag::from_svo(&svo);
        let chunk = Chunk::with_dag(ChunkPos::new(0, 0, 0), dag);
        assert_eq!(chunk.state, ChunkState::Compressed);
        assert!(chunk.svo.is_none());
        assert!(chunk.dag.is_some());
        assert!(chunk.dirty);
    }

    #[test]
    fn chunk_compression() {
        let mut svo = SparseVoxelOctree::new(OCTREE_DEPTH);
        svo.set(0, 0, 0, BlockId::STONE);

        let mut chunk = Chunk::with_svo(ChunkPos::new(0, 0, 0), svo);
        assert!(chunk.svo.is_some());
        assert!(chunk.dag.is_none());

        chunk.compress();

        assert!(chunk.svo.is_none());
        assert!(chunk.dag.is_some());
        assert_eq!(chunk.state, ChunkState::Compressed);
        assert!(chunk.dirty);
    }

    #[test]
    fn chunk_mark_uploaded() {
        let svo = SparseVoxelOctree::new(OCTREE_DEPTH);
        let mut chunk = Chunk::with_svo(ChunkPos::new(0, 0, 0), svo);
        chunk.compress();
        chunk.mark_uploaded();

        assert_eq!(chunk.state, ChunkState::Uploaded);
        assert!(!chunk.dirty);
    }
}
