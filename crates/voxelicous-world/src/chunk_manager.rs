//! Chunk manager with spatial indexing.

use hashbrown::HashMap;
use parking_lot::RwLock;
use voxelicous_core::coords::ChunkPos;

use crate::chunk::{Chunk, ChunkState};

/// Manages all loaded chunks with spatial indexing.
///
/// Provides thread-safe access to chunks via `RwLock`.
pub struct ChunkManager {
    /// All loaded chunks indexed by position.
    chunks: RwLock<HashMap<ChunkPos, Chunk>>,
    /// Maximum number of chunks to keep loaded.
    max_chunks: usize,
    /// Current frame number for LRU tracking.
    current_frame: u64,
}

impl ChunkManager {
    /// Create a new chunk manager with the given capacity.
    pub fn new(max_chunks: usize) -> Self {
        Self {
            chunks: RwLock::new(HashMap::with_capacity(max_chunks)),
            max_chunks,
            current_frame: 0,
        }
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Advance the frame counter and return the new frame number.
    pub fn advance_frame(&mut self) -> u64 {
        self.current_frame += 1;
        self.current_frame
    }

    /// Check if a chunk exists at the given position.
    pub fn contains(&self, pos: ChunkPos) -> bool {
        self.chunks.read().contains_key(&pos)
    }

    /// Get the number of loaded chunks.
    pub fn len(&self) -> usize {
        self.chunks.read().len()
    }

    /// Check if no chunks are loaded.
    pub fn is_empty(&self) -> bool {
        self.chunks.read().is_empty()
    }

    /// Insert or replace a chunk at the given position.
    pub fn insert(&self, chunk: Chunk) {
        let mut chunks = self.chunks.write();
        chunks.insert(chunk.pos, chunk);
    }

    /// Remove a chunk at the given position.
    pub fn remove(&self, pos: ChunkPos) -> Option<Chunk> {
        let mut chunks = self.chunks.write();
        chunks.remove(&pos)
    }

    /// Get all loaded chunk positions.
    pub fn positions(&self) -> Vec<ChunkPos> {
        self.chunks.read().keys().copied().collect()
    }

    /// Execute a function with read access to a chunk.
    ///
    /// Returns `None` if the chunk doesn't exist.
    pub fn with_chunk<F, R>(&self, pos: ChunkPos, f: F) -> Option<R>
    where
        F: FnOnce(&Chunk) -> R,
    {
        let chunks = self.chunks.read();
        chunks.get(&pos).map(f)
    }

    /// Execute a function with write access to a chunk.
    ///
    /// Returns `None` if the chunk doesn't exist.
    pub fn with_chunk_mut<F, R>(&self, pos: ChunkPos, f: F) -> Option<R>
    where
        F: FnOnce(&mut Chunk) -> R,
    {
        let mut chunks = self.chunks.write();
        chunks.get_mut(&pos).map(f)
    }

    /// Get chunks within a cubic radius of a center position.
    pub fn chunks_in_radius(&self, center: ChunkPos, radius: i32) -> Vec<ChunkPos> {
        let chunks = self.chunks.read();
        chunks
            .keys()
            .filter(|pos| {
                let dx = (pos.x - center.x).abs();
                let dy = (pos.y - center.y).abs();
                let dz = (pos.z - center.z).abs();
                dx <= radius && dy <= radius && dz <= radius
            })
            .copied()
            .collect()
    }

    /// Get chunks in a specific state.
    pub fn chunks_in_state(&self, state: ChunkState) -> Vec<ChunkPos> {
        let chunks = self.chunks.read();
        chunks
            .iter()
            .filter(|(_, chunk)| chunk.state == state)
            .map(|(pos, _)| *pos)
            .collect()
    }

    /// Get dirty chunks that need GPU upload.
    pub fn dirty_chunks(&self) -> Vec<ChunkPos> {
        let chunks = self.chunks.read();
        chunks
            .iter()
            .filter(|(_, chunk)| chunk.dirty && chunk.dag.is_some())
            .map(|(pos, _)| *pos)
            .collect()
    }

    /// Get total memory usage of all chunks.
    pub fn memory_usage(&self) -> usize {
        self.chunks.read().values().map(Chunk::memory_usage).sum()
    }

    /// Evict oldest chunks if over capacity.
    ///
    /// Returns the positions of evicted chunks.
    pub fn evict_if_needed(&self) -> Vec<ChunkPos> {
        let mut evicted = Vec::new();
        let mut chunks = self.chunks.write();

        while chunks.len() > self.max_chunks {
            // Find LRU chunk that isn't being generated
            let oldest = chunks
                .iter()
                .filter(|(_, c)| c.state != ChunkState::Generating)
                .min_by_key(|(_, c)| c.last_access_frame)
                .map(|(pos, _)| *pos);

            if let Some(pos) = oldest {
                chunks.remove(&pos);
                evicted.push(pos);
            } else {
                // All remaining chunks are being generated, can't evict
                break;
            }
        }

        evicted
    }

    /// Touch all chunks at the given positions to update their access time.
    pub fn touch_chunks(&self, positions: &[ChunkPos], frame: u64) {
        let mut chunks = self.chunks.write();
        for pos in positions {
            if let Some(chunk) = chunks.get_mut(pos) {
                chunk.touch(frame);
            }
        }
    }

    /// Get the maximum chunk capacity.
    pub fn capacity(&self) -> usize {
        self.max_chunks
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxelicous_core::constants::OCTREE_DEPTH;
    use voxelicous_voxel::SparseVoxelOctree;

    #[test]
    fn insert_and_retrieve() {
        let manager = ChunkManager::new(100);
        let pos = ChunkPos::new(1, 2, 3);
        let chunk = Chunk::new(pos);

        manager.insert(chunk);

        assert!(manager.contains(pos));
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn remove_chunk() {
        let manager = ChunkManager::new(100);
        let pos = ChunkPos::new(1, 2, 3);
        let chunk = Chunk::new(pos);

        manager.insert(chunk);
        let removed = manager.remove(pos);

        assert!(removed.is_some());
        assert!(!manager.contains(pos));
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn chunks_in_radius() {
        let manager = ChunkManager::new(100);

        // Insert chunks at various positions
        for x in -5..=5 {
            for z in -5..=5 {
                manager.insert(Chunk::new(ChunkPos::new(x, 0, z)));
            }
        }

        // Get chunks within radius 2 of origin
        let nearby = manager.chunks_in_radius(ChunkPos::new(0, 0, 0), 2);

        // Should have 5x5 = 25 chunks (from -2 to 2 on each axis)
        assert_eq!(nearby.len(), 25);
    }

    #[test]
    fn with_chunk_access() {
        let manager = ChunkManager::new(100);
        let pos = ChunkPos::new(1, 2, 3);
        let chunk = Chunk::new(pos);

        manager.insert(chunk);

        // Read access
        let state = manager.with_chunk(pos, |c| c.state);
        assert_eq!(state, Some(ChunkState::Pending));

        // Write access
        manager.with_chunk_mut(pos, |c| {
            c.state = ChunkState::Generated;
        });

        let state = manager.with_chunk(pos, |c| c.state);
        assert_eq!(state, Some(ChunkState::Generated));
    }

    #[test]
    fn eviction_order() {
        let manager = ChunkManager::new(3);

        // Insert 3 chunks with different access times
        for i in 0..3 {
            let mut chunk = Chunk::new(ChunkPos::new(i, 0, 0));
            chunk.last_access_frame = (i + 1) as u64; // frames 1, 2, 3
            manager.insert(chunk);
        }

        // Insert 4th chunk with recent access time
        let mut chunk = Chunk::new(ChunkPos::new(3, 0, 0));
        chunk.last_access_frame = 10; // Most recent
        manager.insert(chunk);

        let evicted = manager.evict_if_needed();

        // Should evict oldest (frame 1, which is position 0)
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], ChunkPos::new(0, 0, 0));
    }

    #[test]
    fn dirty_chunks() {
        let manager = ChunkManager::new(100);

        // Create a compressed chunk (has DAG)
        let pos = ChunkPos::new(0, 0, 0);
        let svo = SparseVoxelOctree::new(OCTREE_DEPTH);
        let mut chunk = Chunk::with_svo(pos, svo);
        chunk.compress();

        manager.insert(chunk);

        let dirty = manager.dirty_chunks();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0], pos);

        // Mark as uploaded
        manager.with_chunk_mut(pos, |c| c.mark_uploaded());

        let dirty = manager.dirty_chunks();
        assert!(dirty.is_empty());
    }
}
