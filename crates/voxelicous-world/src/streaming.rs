//! Chunk streaming based on camera position.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::Vec3;
use voxelicous_core::constants::CHUNK_BITS;
use voxelicous_core::coords::ChunkPos;

use crate::chunk::{Chunk, ChunkState};
use crate::chunk_manager::ChunkManager;
use crate::generation::TerrainGenerator;

#[cfg(feature = "profiling")]
use voxelicous_profiler::{profile_scope, EventCategory};

/// Priority entry for chunk loading queue.
#[derive(Debug, Clone, Copy)]
struct LoadPriority {
    pos: ChunkPos,
    /// Squared distance to camera (lower = higher priority).
    distance_sq: i32,
}

impl PartialEq for LoadPriority {
    fn eq(&self, other: &Self) -> bool {
        self.distance_sq == other.distance_sq
    }
}

impl Eq for LoadPriority {}

impl PartialOrd for LoadPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LoadPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (closer chunks have higher priority)
        other.distance_sq.cmp(&self.distance_sq)
    }
}

/// Configuration for chunk streaming behavior.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Horizontal load radius in chunks.
    pub load_radius: i32,
    /// Horizontal unload radius in chunks (should be > load_radius).
    pub unload_radius: i32,
    /// Vertical load radius (usually smaller than horizontal).
    pub vertical_radius: i32,
    /// Maximum chunks to generate per update call.
    pub max_gen_per_update: usize,
    /// Maximum chunks to compress per update call.
    pub max_compress_per_update: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            load_radius: 8,
            unload_radius: 12,
            vertical_radius: 4,
            max_gen_per_update: 4,
            max_compress_per_update: 8,
        }
    }
}

/// Handles chunk streaming based on camera position.
pub struct ChunkStreamer {
    config: StreamingConfig,
    generator: TerrainGenerator,
    load_queue: BinaryHeap<LoadPriority>,
    last_center: Option<ChunkPos>,
    /// Last camera position for distance-based reprioritization.
    last_camera_pos: Option<Vec3>,
    /// Number of updates since last queue rebuild.
    updates_since_rebuild: u32,
}

impl ChunkStreamer {
    /// Create a new chunk streamer with the given configuration and generator.
    pub fn new(config: StreamingConfig, generator: TerrainGenerator) -> Self {
        Self {
            config,
            generator,
            load_queue: BinaryHeap::new(),
            last_center: None,
            last_camera_pos: None,
            updates_since_rebuild: 0,
        }
    }

    /// Get the streaming configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Get the terrain generator.
    pub fn generator(&self) -> &TerrainGenerator {
        &self.generator
    }

    /// Convert world position to chunk position.
    fn world_to_chunk(pos: Vec3) -> ChunkPos {
        ChunkPos::new(
            (pos.x.floor() as i64 >> CHUNK_BITS) as i32,
            (pos.y.floor() as i64 >> CHUNK_BITS) as i32,
            (pos.z.floor() as i64 >> CHUNK_BITS) as i32,
        )
    }

    /// Check if camera has moved to a different chunk.
    fn center_changed(&self, new_center: ChunkPos) -> bool {
        match self.last_center {
            Some(old) => old != new_center,
            None => true,
        }
    }

    /// Rebuild the load queue around the given center.
    fn rebuild_load_queue(&mut self, center: ChunkPos, chunk_manager: &ChunkManager) {
        self.load_queue.clear();

        let r = self.config.load_radius;
        let rv = self.config.vertical_radius;

        for dy in -rv..=rv {
            for dz in -r..=r {
                for dx in -r..=r {
                    let pos = ChunkPos::new(center.x + dx, center.y + dy, center.z + dz);

                    if !chunk_manager.contains(pos) {
                        let distance_sq = dx * dx + dy * dy * 4 + dz * dz; // Weight Y more
                        self.load_queue.push(LoadPriority { pos, distance_sq });
                    }
                }
            }
        }
    }

    /// Unload chunks that are too far from camera.
    fn unload_distant(&self, center: ChunkPos, chunk_manager: &ChunkManager) -> Vec<ChunkPos> {
        let r = self.config.unload_radius;
        let rv = self.config.vertical_radius + 2; // Small buffer for vertical

        let positions = chunk_manager.positions();
        let mut unloaded = Vec::new();

        for pos in positions {
            let dx = (pos.x - center.x).abs();
            let dy = (pos.y - center.y).abs();
            let dz = (pos.z - center.z).abs();

            if dx > r || dy > rv || dz > r {
                chunk_manager.remove(pos);
                unloaded.push(pos);
            }
        }

        unloaded
    }

    /// Check if we should rebuild the queue based on camera movement or time.
    fn should_rebuild_queue(&self, camera_pos: Vec3, center: ChunkPos) -> bool {
        // Always rebuild if center chunk changed
        if self.center_changed(center) {
            return true;
        }

        // Rebuild periodically to keep priorities fresh
        if self.updates_since_rebuild > 30 {
            return true;
        }

        // Rebuild if camera moved significantly within the same chunk (half chunk size)
        if let Some(last_pos) = self.last_camera_pos {
            let distance_sq = (camera_pos - last_pos).length_squared();
            if distance_sq > 256.0 {
                // 16 units = half chunk
                return true;
            }
        }

        false
    }

    /// Update streaming state based on camera position.
    ///
    /// Returns a tuple of (chunks needing GPU upload, chunks unloaded).
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        chunk_manager: &ChunkManager,
    ) -> (Vec<ChunkPos>, Vec<ChunkPos>) {
        let center = Self::world_to_chunk(camera_pos);

        // Rebuild queue if camera moved significantly or periodically
        if self.should_rebuild_queue(camera_pos, center) {
            self.rebuild_load_queue(center, chunk_manager);
            self.last_center = Some(center);
            self.last_camera_pos = Some(camera_pos);
            self.updates_since_rebuild = 0;
        } else {
            self.updates_since_rebuild += 1;
        }

        // Unload distant chunks
        let unloaded = self.unload_distant(center, chunk_manager);

        // Generate pending chunks (up to limit)
        let mut generated = Vec::new();
        for _ in 0..self.config.max_gen_per_update {
            if let Some(entry) = self.load_queue.pop() {
                if !chunk_manager.contains(entry.pos) {
                    let svo = {
                        #[cfg(feature = "profiling")]
                        profile_scope!(EventCategory::ChunkGeneration, [entry.pos.x, entry.pos.y, entry.pos.z]);
                        self.generator.generate_chunk(entry.pos)
                    };
                    let chunk = Chunk::with_svo(entry.pos, svo);
                    chunk_manager.insert(chunk);
                    generated.push(entry.pos);
                }
            } else {
                break;
            }
        }

        // Compress recently generated chunks
        let mut needs_upload = Vec::new();
        let generated_chunks = chunk_manager.chunks_in_state(ChunkState::Generated);

        for (i, pos) in generated_chunks.into_iter().enumerate() {
            if i >= self.config.max_compress_per_update {
                break;
            }

            chunk_manager.with_chunk_mut(pos, |chunk| {
                #[cfg(feature = "profiling")]
                profile_scope!(EventCategory::ChunkCompression, [pos.x, pos.y, pos.z]);
                chunk.compress();
            });
            needs_upload.push(pos);
        }

        (needs_upload, unloaded)
    }

    /// Get the number of chunks waiting to be loaded.
    pub fn pending_count(&self) -> usize {
        self.load_queue.len()
    }

    /// Force generation of a specific chunk, bypassing the queue.
    pub fn force_generate(&self, pos: ChunkPos, chunk_manager: &ChunkManager) {
        if !chunk_manager.contains(pos) {
            let svo = self.generator.generate_chunk(pos);
            let chunk = Chunk::with_svo(pos, svo);
            chunk_manager.insert(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::TerrainConfig;

    fn create_test_streamer() -> (ChunkStreamer, ChunkManager) {
        let config = StreamingConfig {
            load_radius: 2,
            unload_radius: 3,
            vertical_radius: 1,
            max_gen_per_update: 10,
            max_compress_per_update: 10,
        };
        let generator = TerrainGenerator::new(TerrainConfig::default());
        let streamer = ChunkStreamer::new(config, generator);
        let manager = ChunkManager::new(100);

        (streamer, manager)
    }

    #[test]
    fn load_priority_ordering() {
        let mut heap = BinaryHeap::new();

        heap.push(LoadPriority {
            pos: ChunkPos::new(10, 0, 0),
            distance_sq: 100,
        });
        heap.push(LoadPriority {
            pos: ChunkPos::new(1, 0, 0),
            distance_sq: 1,
        });
        heap.push(LoadPriority {
            pos: ChunkPos::new(5, 0, 0),
            distance_sq: 25,
        });

        // Closest should come first
        assert_eq!(heap.pop().unwrap().distance_sq, 1);
        assert_eq!(heap.pop().unwrap().distance_sq, 25);
        assert_eq!(heap.pop().unwrap().distance_sq, 100);
    }

    #[test]
    fn world_to_chunk_conversion() {
        // Inside chunk 0,0,0
        let chunk = ChunkStreamer::world_to_chunk(Vec3::new(15.0, 15.0, 15.0));
        assert_eq!(chunk, ChunkPos::new(0, 0, 0));

        // Inside chunk 1,0,0
        let chunk = ChunkStreamer::world_to_chunk(Vec3::new(33.0, 15.0, 15.0));
        assert_eq!(chunk, ChunkPos::new(1, 0, 0));

        // Negative coordinates
        let chunk = ChunkStreamer::world_to_chunk(Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(chunk, ChunkPos::new(-1, -1, -1));
    }

    #[test]
    fn streamer_generates_chunks() {
        let (mut streamer, manager) = create_test_streamer();

        // Camera at origin
        let camera_pos = Vec3::new(16.0, 80.0, 16.0); // Center of chunk (0, 2, 0)

        let (needs_upload, _) = streamer.update(camera_pos, &manager);

        // Should have generated some chunks
        assert!(!manager.is_empty());
        assert!(!needs_upload.is_empty());
    }

    #[test]
    fn streamer_unloads_distant_chunks() {
        let (mut streamer, manager) = create_test_streamer();

        // Generate chunks at origin
        let camera_pos = Vec3::new(16.0, 16.0, 16.0);
        streamer.update(camera_pos, &manager);

        // Move camera far away
        let far_camera = Vec3::new(500.0, 16.0, 500.0);
        let (_, unloaded) = streamer.update(far_camera, &manager);

        // Should have unloaded the original chunks
        assert!(!unloaded.is_empty());
    }

    #[test]
    fn force_generate_chunk() {
        let (streamer, manager) = create_test_streamer();
        let pos = ChunkPos::new(100, 0, 100);

        assert!(!manager.contains(pos));

        streamer.force_generate(pos, &manager);

        assert!(manager.contains(pos));
    }
}
