//! Profiler event types and statistics.

use serde::{Deserialize, Serialize};

/// Categories for profiling events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EventCategory {
    /// Full frame timing.
    Frame = 0,
    /// Frame update phase (app logic).
    FrameUpdate = 1,
    /// Frame render phase (command recording + app.render).
    FrameRender = 2,
    /// Frame present phase (swapchain present).
    FramePresent = 3,
    /// GPU synchronization (fence wait + image acquire).
    GpuSync = 4,
    /// GPU queue submit.
    GpuSubmit = 5,
    /// Chunk terrain generation.
    ChunkGeneration = 10,
    /// Chunk SVO compression.
    ChunkCompression = 11,
    /// GPU chunk upload.
    GpuUpload = 12,
    /// GPU chunk unload.
    GpuUnload = 13,
    /// World streaming update.
    WorldUpdate = 14,
    /// Custom event with ID.
    Custom(u32) = 255,
}

impl EventCategory {
    /// Get a display name for this category.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Frame => "Frame",
            Self::FrameUpdate => "Update",
            Self::FrameRender => "Render",
            Self::FramePresent => "Present",
            Self::GpuSync => "GPU Sync",
            Self::GpuSubmit => "GPU Submit",
            Self::ChunkGeneration => "Generation",
            Self::ChunkCompression => "Compression",
            Self::GpuUpload => "GPU Upload",
            Self::GpuUnload => "GPU Unload",
            Self::WorldUpdate => "World Update",
            Self::Custom(_) => "Custom",
        }
    }
}

/// A single timing event.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimingEvent {
    /// Event category.
    pub category: EventCategory,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Optional context (e.g., chunk coordinates packed as i32s).
    pub context: [i32; 3],
}

impl TimingEvent {
    /// Create a new timing event.
    #[must_use]
    pub const fn new(category: EventCategory, duration_ns: u64) -> Self {
        Self {
            category,
            duration_ns,
            context: [0, 0, 0],
        }
    }

    /// Create a timing event with context.
    #[must_use]
    pub const fn with_context(
        category: EventCategory,
        duration_ns: u64,
        context: [i32; 3],
    ) -> Self {
        Self {
            category,
            duration_ns,
            context,
        }
    }
}

/// Aggregated statistics for a category.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CategoryStats {
    /// Event category.
    pub category: EventCategory,
    /// Number of events.
    pub count: u32,
    /// Total duration in nanoseconds.
    pub total_ns: u64,
    /// Minimum duration in nanoseconds.
    pub min_ns: u64,
    /// Maximum duration in nanoseconds.
    pub max_ns: u64,
    /// Average duration in nanoseconds (computed from total/count).
    pub avg_ns: u64,
    /// 95th percentile duration (approximate).
    pub p95_ns: u64,
}

impl Default for EventCategory {
    fn default() -> Self {
        Self::Frame
    }
}

impl CategoryStats {
    /// Create new empty stats for a category.
    #[must_use]
    pub const fn new(category: EventCategory) -> Self {
        Self {
            category,
            count: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            avg_ns: 0,
            p95_ns: 0,
        }
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.total_ns = 0;
        self.min_ns = u64::MAX;
        self.max_ns = 0;
        self.avg_ns = 0;
        self.p95_ns = 0;
    }

    /// Record a new timing.
    pub fn record(&mut self, duration_ns: u64) {
        self.count += 1;
        self.total_ns += duration_ns;
        self.min_ns = self.min_ns.min(duration_ns);
        self.max_ns = self.max_ns.max(duration_ns);
        if self.count > 0 {
            self.avg_ns = self.total_ns / u64::from(self.count);
        }
    }

    /// Get minimum in milliseconds.
    #[must_use]
    pub fn min_ms(&self) -> f64 {
        self.min_ns as f64 / 1_000_000.0
    }

    /// Get maximum in milliseconds.
    #[must_use]
    pub fn max_ms(&self) -> f64 {
        self.max_ns as f64 / 1_000_000.0
    }

    /// Get average in milliseconds.
    #[must_use]
    pub fn avg_ms(&self) -> f64 {
        self.avg_ns as f64 / 1_000_000.0
    }

    /// Get total in milliseconds.
    #[must_use]
    pub fn total_ms(&self) -> f64 {
        self.total_ns as f64 / 1_000_000.0
    }
}

/// Queue sizes for streaming operations.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct QueueSizes {
    /// Chunks waiting to be uploaded to GPU.
    pub pending_uploads: u32,
    /// Chunks waiting to be unloaded from GPU.
    pub pending_unloads: u32,
    /// Chunks in the load queue.
    pub load_queue_length: u32,
    /// Chunks currently being generated.
    pub chunks_generating: u32,
    /// Total chunks loaded in world.
    pub total_chunks: u32,
    /// Chunks on GPU.
    pub gpu_chunks: u32,
}

/// Memory usage statistics.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// GPU memory used in bytes.
    pub gpu_memory_bytes: u64,
    /// CPU memory for chunks in bytes.
    pub chunk_memory_bytes: u64,
}

/// Complete profiler snapshot sent to TUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerSnapshot {
    /// Current frame number.
    pub frame_number: u64,
    /// Current FPS.
    pub fps: f32,
    /// Frame time in milliseconds.
    pub frame_time_ms: f32,
    /// Per-category statistics.
    pub categories: Vec<CategoryStats>,
    /// Queue sizes.
    pub queues: QueueSizes,
    /// Memory stats.
    pub memory: MemoryStats,
}

impl Default for ProfilerSnapshot {
    fn default() -> Self {
        Self {
            frame_number: 0,
            fps: 0.0,
            frame_time_ms: 0.0,
            categories: Vec::new(),
            queues: QueueSizes::default(),
            memory: MemoryStats::default(),
        }
    }
}
