//! Chunk streaming based on camera position.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::thread::{self, JoinHandle};

use crossbeam::channel::{self, Receiver, Sender};
use glam::Vec3;
use hashbrown::HashSet;
use voxelicous_core::constants::CHUNK_BITS;
use voxelicous_core::coords::ChunkPos;
use voxelicous_voxel::{SvoDag, VoxelStorage};

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

/// Work request sent to the background worker thread.
#[derive(Debug)]
pub enum ChunkWorkRequest {
    /// Generate chunks at the given positions.
    Generate(Vec<ChunkPos>),
    /// Signal worker thread to shut down.
    Shutdown,
}

/// Result returned by the worker thread after processing.
pub struct ChunkWorkResult {
    /// Position of the generated chunk.
    pub pos: ChunkPos,
    /// Compressed DAG data.
    pub dag: SvoDag,
    /// Whether the chunk is empty (all air).
    pub is_empty: bool,
}

impl std::fmt::Debug for ChunkWorkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkWorkResult")
            .field("pos", &self.pos)
            .field("is_empty", &self.is_empty)
            .field("dag", &"<SvoDag>")
            .finish()
    }
}

/// Handle to the background chunk worker thread.
struct ChunkWorkerHandle {
    /// Channel to send work requests to the worker.
    request_tx: Sender<ChunkWorkRequest>,
    /// Channel to receive completed results from the worker.
    result_rx: Receiver<ChunkWorkResult>,
    /// Worker thread handle for joining on shutdown.
    thread: Option<JoinHandle<()>>,
}

impl ChunkWorkerHandle {
    /// Spawn a new worker thread with the given generator.
    fn spawn(generator: TerrainGenerator) -> Self {
        let (request_tx, request_rx) = channel::bounded::<ChunkWorkRequest>(16);
        let (result_tx, result_rx) = channel::bounded::<ChunkWorkResult>(256);

        let thread = thread::Builder::new()
            .name("chunk-worker".to_string())
            .spawn(move || {
                Self::worker_loop(generator, request_rx, result_tx);
            })
            .expect("Failed to spawn chunk worker thread");

        Self {
            request_tx,
            result_rx,
            thread: Some(thread),
        }
    }

    /// Main worker loop - blocks waiting for requests and processes them.
    fn worker_loop(
        generator: TerrainGenerator,
        request_rx: Receiver<ChunkWorkRequest>,
        result_tx: Sender<ChunkWorkResult>,
    ) {
        loop {
            // Block waiting for work
            match request_rx.recv() {
                Ok(ChunkWorkRequest::Generate(positions)) => {
                    // Use Rayon for parallel chunk generation
                    let generated = generator.generate_chunks_parallel(&positions);

                    // Compress each chunk and send results
                    for (pos, svo) in generated {
                        let dag = SvoDag::from_svo(&svo);
                        let is_empty = dag.is_empty();

                        // Send result back (blocking - backpressure if main thread is slow)
                        if result_tx
                            .send(ChunkWorkResult { pos, dag, is_empty })
                            .is_err()
                        {
                            // Receiver dropped, exit loop
                            return;
                        }
                    }
                }
                Ok(ChunkWorkRequest::Shutdown) | Err(_) => {
                    // Shutdown requested or channel disconnected
                    return;
                }
            }
        }
    }

    /// Send a batch of positions to generate (non-blocking).
    fn send_work(&self, positions: Vec<ChunkPos>) -> Result<(), Vec<ChunkPos>> {
        match self.request_tx.try_send(ChunkWorkRequest::Generate(positions.clone())) {
            Ok(()) => Ok(()),
            Err(_) => Err(positions), // Queue full, return positions
        }
    }

    /// Try to receive completed results (non-blocking).
    fn try_recv(&self) -> Option<ChunkWorkResult> {
        self.result_rx.try_recv().ok()
    }

    /// Shutdown the worker thread and wait for it to finish.
    fn shutdown(&mut self) {
        // Send shutdown signal (ignore errors - channel might be closed)
        let _ = self.request_tx.send(ChunkWorkRequest::Shutdown);

        // Wait for thread to finish
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for ChunkWorkerHandle {
    fn drop(&mut self) {
        self.shutdown();
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
    /// Background worker thread handle (None for sync mode).
    worker: Option<ChunkWorkerHandle>,
    /// Positions currently being processed by the worker.
    in_flight: HashSet<ChunkPos>,
}

impl ChunkStreamer {
    /// Create a new chunk streamer with the given configuration and generator.
    ///
    /// This creates a synchronous streamer where chunk generation happens on the main thread.
    pub fn new(config: StreamingConfig, generator: TerrainGenerator) -> Self {
        Self {
            config,
            generator,
            load_queue: BinaryHeap::new(),
            last_center: None,
            last_camera_pos: None,
            updates_since_rebuild: 0,
            worker: None,
            in_flight: HashSet::new(),
        }
    }

    /// Create a new chunk streamer with async background generation.
    ///
    /// This creates a streamer where chunk generation and compression happen
    /// on a background worker thread, keeping the main thread responsive.
    pub fn new_async(config: StreamingConfig, generator: TerrainGenerator) -> Self {
        let worker = ChunkWorkerHandle::spawn(generator.clone());

        Self {
            config,
            generator,
            load_queue: BinaryHeap::new(),
            last_center: None,
            last_camera_pos: None,
            updates_since_rebuild: 0,
            worker: Some(worker),
            in_flight: HashSet::new(),
        }
    }

    /// Check if this streamer is running in async mode.
    pub fn is_async(&self) -> bool {
        self.worker.is_some()
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

    /// Update streaming state asynchronously based on camera position.
    ///
    /// This method is non-blocking - it submits work to the background worker
    /// and collects completed results without waiting.
    ///
    /// Returns a tuple of (chunks needing GPU upload, chunks unloaded).
    ///
    /// # Panics
    ///
    /// Panics if called on a synchronous streamer (created with `new()` instead of `new_async()`).
    pub fn update_async(
        &mut self,
        camera_pos: Vec3,
        chunk_manager: &ChunkManager,
    ) -> (Vec<ChunkPos>, Vec<ChunkPos>) {
        assert!(
            self.worker.is_some(),
            "update_async called on sync streamer - use new_async()"
        );

        let center = Self::world_to_chunk(camera_pos);

        // Step 1: Collect completed chunks from worker (non-blocking)
        let needs_upload = self.collect_completed_chunks(chunk_manager);

        // Step 2: Rebuild queue if camera moved significantly or periodically
        if self.should_rebuild_queue(camera_pos, center) {
            self.rebuild_load_queue_async(center, chunk_manager);
            self.last_center = Some(center);
            self.last_camera_pos = Some(camera_pos);
            self.updates_since_rebuild = 0;
        } else {
            self.updates_since_rebuild += 1;
        }

        // Step 3: Unload distant chunks (also remove from in_flight)
        let unloaded = self.unload_distant_async(center, chunk_manager);

        // Step 4: Submit new work to worker (non-blocking)
        self.submit_pending_work();

        (needs_upload, unloaded)
    }

    /// Collect completed chunks from the worker thread.
    fn collect_completed_chunks(&mut self, chunk_manager: &ChunkManager) -> Vec<ChunkPos> {
        // First, collect all available results from the worker
        let mut results = Vec::new();
        if let Some(worker) = &self.worker {
            while let Some(result) = worker.try_recv() {
                results.push(result);
            }
        }

        // Now process the results (we can mutably borrow self here)
        let mut needs_upload = Vec::new();
        for result in results {
            // Remove from in-flight tracking
            self.in_flight.remove(&result.pos);

            // Add to chunk manager if not already present
            // (even empty chunks - to prevent them from being re-queued)
            if !chunk_manager.contains(result.pos) {
                let chunk = Chunk::with_dag(result.pos, result.dag);
                chunk_manager.insert(chunk);

                // Only upload non-empty chunks to GPU
                if !result.is_empty {
                    needs_upload.push(result.pos);
                }
            }
        }

        needs_upload
    }

    /// Rebuild load queue for async mode (excludes in-flight positions).
    fn rebuild_load_queue_async(&mut self, center: ChunkPos, chunk_manager: &ChunkManager) {
        self.load_queue.clear();

        let r = self.config.load_radius;
        let rv = self.config.vertical_radius;

        for dy in -rv..=rv {
            for dz in -r..=r {
                for dx in -r..=r {
                    let pos = ChunkPos::new(center.x + dx, center.y + dy, center.z + dz);

                    // Skip if already loaded or in-flight
                    if !chunk_manager.contains(pos) && !self.in_flight.contains(&pos) {
                        let distance_sq = dx * dx + dy * dy * 4 + dz * dz;
                        self.load_queue.push(LoadPriority { pos, distance_sq });
                    }
                }
            }
        }
    }

    /// Unload distant chunks for async mode (also clears from in_flight).
    fn unload_distant_async(
        &mut self,
        center: ChunkPos,
        chunk_manager: &ChunkManager,
    ) -> Vec<ChunkPos> {
        let r = self.config.unload_radius;
        let rv = self.config.vertical_radius + 2;

        let positions = chunk_manager.positions();
        let mut unloaded = Vec::new();

        for pos in positions {
            let dx = (pos.x - center.x).abs();
            let dy = (pos.y - center.y).abs();
            let dz = (pos.z - center.z).abs();

            if dx > r || dy > rv || dz > r {
                chunk_manager.remove(pos);
                self.in_flight.remove(&pos);
                unloaded.push(pos);
            }
        }

        // Also clean up in_flight positions that are now too far
        self.in_flight.retain(|pos| {
            let dx = (pos.x - center.x).abs();
            let dy = (pos.y - center.y).abs();
            let dz = (pos.z - center.z).abs();
            dx <= r && dy <= rv && dz <= r
        });

        unloaded
    }

    /// Submit pending work to the worker thread.
    fn submit_pending_work(&mut self) {
        // Collect positions to submit (up to limit, respecting in-flight capacity)
        let max_in_flight = self.config.max_gen_per_update * 4; // Allow some buffering
        let available_slots = max_in_flight.saturating_sub(self.in_flight.len());

        if available_slots == 0 {
            return;
        }

        let batch_size = self.config.max_gen_per_update.min(available_slots);
        let mut batch = Vec::with_capacity(batch_size);

        while batch.len() < batch_size {
            if let Some(entry) = self.load_queue.pop() {
                // Double-check not already in flight or loaded
                if !self.in_flight.contains(&entry.pos) {
                    batch.push(entry.pos);
                }
            } else {
                break;
            }
        }

        if batch.is_empty() {
            return;
        }

        // Mark as in-flight
        for &pos in &batch {
            self.in_flight.insert(pos);
        }

        // Submit to worker
        if let Some(worker) = &self.worker {
            if let Err(returned_positions) = worker.send_work(batch) {
                // Queue was full, remove from in_flight and put back in load queue
                for pos in returned_positions {
                    self.in_flight.remove(&pos);
                    // Put back at front of queue (they were highest priority)
                    self.load_queue.push(LoadPriority {
                        pos,
                        distance_sq: 0, // High priority
                    });
                }
            }
        }
    }

    /// Get the number of chunks currently being processed by the worker.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
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

    fn create_async_test_streamer() -> (ChunkStreamer, ChunkManager) {
        let config = StreamingConfig {
            load_radius: 2,
            unload_radius: 3,
            vertical_radius: 1,
            max_gen_per_update: 10,
            max_compress_per_update: 10,
        };
        let generator = TerrainGenerator::new(TerrainConfig::default());
        let streamer = ChunkStreamer::new_async(config, generator);
        let manager = ChunkManager::new(100);

        (streamer, manager)
    }

    #[test]
    fn async_streamer_is_async() {
        let (streamer, _) = create_async_test_streamer();
        assert!(streamer.is_async());
    }

    #[test]
    fn sync_streamer_is_not_async() {
        let (streamer, _) = create_test_streamer();
        assert!(!streamer.is_async());
    }

    #[test]
    fn async_streamer_generates_chunks() {
        let (mut streamer, manager) = create_async_test_streamer();

        let camera_pos = Vec3::new(16.0, 80.0, 16.0);

        // First update should submit work
        let (needs_upload_1, _) = streamer.update_async(camera_pos, &manager);

        // Worker might not have finished yet, so needs_upload might be empty
        // But we should have in-flight work
        assert!(streamer.in_flight_count() > 0 || !needs_upload_1.is_empty());

        // Wait a bit and do more updates to collect results
        std::thread::sleep(std::time::Duration::from_millis(100));

        let mut total_uploads = needs_upload_1.len();
        for _ in 0..10 {
            let (needs_upload, _) = streamer.update_async(camera_pos, &manager);
            total_uploads += needs_upload.len();
            if !manager.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // Should have received some chunks from worker
        assert!(!manager.is_empty() || total_uploads > 0);
    }

    #[test]
    fn async_streamer_unloads_distant_chunks() {
        let (mut streamer, manager) = create_async_test_streamer();

        // Generate chunks at origin
        let camera_pos = Vec3::new(16.0, 16.0, 16.0);

        // Run several updates to let chunks generate
        for _ in 0..20 {
            streamer.update_async(camera_pos, &manager);
            std::thread::sleep(std::time::Duration::from_millis(50));
            if manager.len() > 5 {
                break;
            }
        }

        let chunks_before = manager.len();

        // Move camera far away
        let far_camera = Vec3::new(500.0, 16.0, 500.0);
        let (_, unloaded) = streamer.update_async(far_camera, &manager);

        // Should have unloaded chunks (or have some if there were any)
        if chunks_before > 0 {
            assert!(!unloaded.is_empty() || manager.len() < chunks_before);
        }
    }
}
