//! World generation and streaming for the Voxelicous engine.
//!
//! This crate provides:
//! - Chunk data structures for voxel world storage
//! - Procedural terrain generation using noise
//! - Chunk streaming based on camera position
//! - Spatial indexing for efficient chunk access

pub mod chunk;
pub mod chunk_manager;
pub mod generation;
pub mod streaming;

pub use chunk::{Chunk, ChunkState};
pub use chunk_manager::ChunkManager;
pub use generation::{TerrainConfig, TerrainGenerator};
pub use streaming::{ChunkStreamer, ChunkWorkResult, StreamingConfig};

use glam::Vec3;
use voxelicous_core::coords::ChunkPos;

/// World seed for procedural generation.
pub type WorldSeed = u64;

/// The main world container combining chunk management and streaming.
pub struct World {
    /// Manager for all loaded chunks.
    pub chunk_manager: ChunkManager,
    /// Streamer for camera-based chunk loading.
    pub streamer: ChunkStreamer,
}

impl World {
    /// Create a new world with the given seed.
    pub fn new(seed: WorldSeed) -> Self {
        let terrain_config = TerrainConfig {
            seed,
            ..Default::default()
        };
        let generator = TerrainGenerator::new(terrain_config);

        Self {
            chunk_manager: ChunkManager::new(1024),
            streamer: ChunkStreamer::new(StreamingConfig::default(), generator),
        }
    }

    /// Create a new world with custom configuration.
    pub fn with_config(
        terrain_config: TerrainConfig,
        streaming_config: StreamingConfig,
        max_chunks: usize,
    ) -> Self {
        let generator = TerrainGenerator::new(terrain_config);

        Self {
            chunk_manager: ChunkManager::new(max_chunks),
            streamer: ChunkStreamer::new(streaming_config, generator),
        }
    }

    /// Create a new world with async chunk streaming.
    ///
    /// This creates a world where chunk generation and compression happen
    /// on a background worker thread, keeping the main thread responsive.
    /// Use `update_async()` instead of `update()` when using this constructor.
    pub fn with_async_streaming(
        terrain_config: TerrainConfig,
        streaming_config: StreamingConfig,
        max_chunks: usize,
    ) -> Self {
        let generator = TerrainGenerator::new(terrain_config);

        Self {
            chunk_manager: ChunkManager::new(max_chunks),
            streamer: ChunkStreamer::new_async(streaming_config, generator),
        }
    }

    /// Check if this world uses async streaming.
    pub fn is_async(&self) -> bool {
        self.streamer.is_async()
    }

    /// Update the world based on camera position.
    ///
    /// Returns a tuple of (chunks needing GPU upload, chunks unloaded).
    pub fn update(&mut self, camera_pos: Vec3) -> (Vec<ChunkPos>, Vec<ChunkPos>) {
        self.streamer.update(camera_pos, &self.chunk_manager)
    }

    /// Update the world asynchronously based on camera position.
    ///
    /// This method is non-blocking - it submits work to the background worker
    /// and collects completed results without waiting.
    ///
    /// Returns a tuple of (chunks needing GPU upload, chunks unloaded).
    ///
    /// # Panics
    ///
    /// Panics if called on a world not created with `with_async_streaming()`.
    pub fn update_async(&mut self, camera_pos: Vec3) -> (Vec<ChunkPos>, Vec<ChunkPos>) {
        self.streamer.update_async(camera_pos, &self.chunk_manager)
    }

    /// Get the number of loaded chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunk_manager.len()
    }

    /// Get total memory usage of all chunks.
    pub fn memory_usage(&self) -> usize {
        self.chunk_manager.memory_usage()
    }

    /// Get the terrain generator.
    pub fn generator(&self) -> &TerrainGenerator {
        self.streamer.generator()
    }

    /// Force generation of a specific chunk.
    pub fn force_generate(&self, pos: ChunkPos) {
        self.streamer.force_generate(pos, &self.chunk_manager);
    }

    /// Generate a grid of chunks around a position for initial world setup.
    ///
    /// This is useful for setting up a small test scene without streaming.
    pub fn generate_initial_chunks(&mut self, center: ChunkPos, radius: i32, vertical_radius: i32) {
        for dy in -vertical_radius..=vertical_radius {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    let pos = ChunkPos::new(center.x + dx, center.y + dy, center.z + dz);
                    self.force_generate(pos);
                }
            }
        }

        // Compress all generated chunks
        for pos in self.chunk_manager.chunks_in_state(ChunkState::Generated) {
            self.chunk_manager.with_chunk_mut(pos, |chunk| {
                chunk.compress();
            });
        }
    }

    /// Get all compressed chunks ready for GPU upload.
    pub fn chunks_ready_for_upload(&self) -> Vec<ChunkPos> {
        self.chunk_manager.dirty_chunks()
    }

    /// Get the number of chunks pending in the load queue.
    pub fn pending_load_count(&self) -> usize {
        self.streamer.pending_count()
    }

    /// Get the number of chunks currently being generated by the background worker.
    ///
    /// Returns 0 for sync mode.
    pub fn in_flight_count(&self) -> usize {
        self.streamer.in_flight_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_creation() {
        let world = World::new(42);
        assert_eq!(world.chunk_count(), 0);
    }

    #[test]
    fn world_update_generates_chunks() {
        let mut world = World::new(42);
        let camera_pos = Vec3::new(16.0, 80.0, 16.0);

        let (needs_upload, _) = world.update(camera_pos);

        assert!(world.chunk_count() > 0);
        assert!(!needs_upload.is_empty());
    }

    #[test]
    fn generate_initial_chunks() {
        let mut world = World::new(42);
        let center = ChunkPos::new(0, 2, 0); // Y=2 for around sea level

        world.generate_initial_chunks(center, 1, 0);

        // Should have 3x1x3 = 9 chunks
        assert_eq!(world.chunk_count(), 9);

        // All should be compressed
        let generated = world.chunk_manager.chunks_in_state(ChunkState::Generated);
        assert!(generated.is_empty());

        let compressed = world.chunk_manager.chunks_in_state(ChunkState::Compressed);
        assert_eq!(compressed.len(), 9);
    }

    #[test]
    fn force_generate_creates_chunk() {
        let world = World::new(42);
        let pos = ChunkPos::new(10, 10, 10);

        assert!(!world.chunk_manager.contains(pos));

        world.force_generate(pos);

        assert!(world.chunk_manager.contains(pos));
    }

    #[test]
    fn async_world_creation() {
        let terrain_config = TerrainConfig {
            seed: 42,
            ..Default::default()
        };
        let streaming_config = StreamingConfig::default();
        let world = World::with_async_streaming(terrain_config, streaming_config, 100);

        assert!(world.is_async());
        assert_eq!(world.chunk_count(), 0);
    }

    #[test]
    fn sync_world_is_not_async() {
        let world = World::new(42);
        assert!(!world.is_async());
    }

    #[test]
    fn async_world_generates_chunks() {
        let terrain_config = TerrainConfig {
            seed: 42,
            ..Default::default()
        };
        let streaming_config = StreamingConfig {
            load_radius: 2,
            unload_radius: 4,
            vertical_radius: 1,
            max_gen_per_update: 10,
            max_compress_per_update: 10,
        };
        let mut world = World::with_async_streaming(terrain_config, streaming_config, 100);

        let camera_pos = Vec3::new(16.0, 80.0, 16.0);

        // Run several update cycles to let background worker process
        for _ in 0..20 {
            world.update_async(camera_pos);
            std::thread::sleep(std::time::Duration::from_millis(50));
            if world.chunk_count() > 5 {
                break;
            }
        }

        // Should have generated some chunks
        assert!(world.chunk_count() > 0 || world.in_flight_count() > 0);
    }
}
