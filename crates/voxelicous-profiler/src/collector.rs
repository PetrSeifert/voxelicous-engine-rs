//! Event collection and aggregation.

use std::collections::HashMap;

use crate::events::{
    CategoryStats, EventCategory, MemoryStats, ProfilerSnapshot, QueueSizes, TimingEvent,
};
use crate::ring_buffer::RingBuffer;

/// Number of recent samples to keep for percentile calculations.
const SAMPLE_HISTORY_SIZE: usize = 100;

/// Collects and aggregates profiling events.
pub struct Collector {
    /// Ring buffer for incoming events.
    buffer: RingBuffer,
    /// Per-category statistics.
    stats: HashMap<EventCategory, CategoryStats>,
    /// Recent samples per category for percentile calculation.
    samples: HashMap<EventCategory, Vec<u64>>,
    /// Current queue sizes.
    queues: QueueSizes,
    /// Current memory stats.
    memory: MemoryStats,
    /// Current frame number.
    frame_number: u64,
    /// Current FPS.
    fps: f32,
    /// Current frame time in ms.
    frame_time_ms: f32,
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector {
    /// Create a new collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: RingBuffer::new(),
            stats: HashMap::new(),
            samples: HashMap::new(),
            queues: QueueSizes::default(),
            memory: MemoryStats::default(),
            frame_number: 0,
            fps: 0.0,
            frame_time_ms: 0.0,
        }
    }

    /// Record a timing event.
    pub fn record(&self, event: TimingEvent) {
        // This drops events if buffer is full, which is acceptable for profiling
        let _ = self.buffer.push(event);
    }

    /// Record a duration for a category.
    pub fn record_duration(&self, category: EventCategory, duration_ns: u64) {
        self.record(TimingEvent::new(category, duration_ns));
    }

    /// Update queue sizes.
    pub fn set_queue_sizes(&mut self, queues: QueueSizes) {
        self.queues = queues;
    }

    /// Update memory stats.
    pub fn set_memory_stats(&mut self, memory: MemoryStats) {
        self.memory = memory;
    }

    /// Update frame info.
    pub fn set_frame_info(&mut self, frame_number: u64, fps: f32, frame_time_ms: f32) {
        self.frame_number = frame_number;
        self.fps = fps;
        self.frame_time_ms = frame_time_ms;
    }

    /// Process all pending events and update statistics.
    pub fn flush(&mut self) {
        // Drain all events from buffer
        let events = self.buffer.drain();

        for event in events {
            // Update stats
            let stats = self
                .stats
                .entry(event.category)
                .or_insert_with(|| CategoryStats::new(event.category));
            stats.record(event.duration_ns);

            // Store sample for percentile calculation
            let samples = self
                .samples
                .entry(event.category)
                .or_insert_with(|| Vec::with_capacity(SAMPLE_HISTORY_SIZE));
            if samples.len() >= SAMPLE_HISTORY_SIZE {
                samples.remove(0);
            }
            samples.push(event.duration_ns);

            // Update p95 estimate
            if samples.len() >= 10 {
                let mut sorted = samples.clone();
                sorted.sort_unstable();
                let p95_idx = (sorted.len() * 95) / 100;
                stats.p95_ns = sorted[p95_idx];
            }
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.stats.clear();
        self.samples.clear();
    }

    /// Get a snapshot of current profiling data.
    #[must_use]
    pub fn snapshot(&self) -> ProfilerSnapshot {
        let mut categories: Vec<CategoryStats> = self.stats.values().copied().collect();

        // Sort by category for consistent display
        categories.sort_by_key(|s| match s.category {
            EventCategory::Frame => 0,
            EventCategory::FrameUpdate => 1,
            EventCategory::GpuSync => 2,
            EventCategory::FrameRender => 3,
            EventCategory::GpuSubmit => 4,
            EventCategory::FramePresent => 5,
            EventCategory::ClipmapPageBuild => 10,
            EventCategory::ClipmapEncode => 11,
            EventCategory::GpuClipmapUpload => 12,
            EventCategory::GpuClipmapUnload => 13,
            EventCategory::ClipmapUpdate => 14,
            EventCategory::Custom(id) => 100 + id as i32,
        });

        ProfilerSnapshot {
            frame_number: self.frame_number,
            fps: self.fps,
            frame_time_ms: self.frame_time_ms,
            categories,
            queues: self.queues,
            memory: self.memory,
        }
    }

    /// Get stats for a specific category.
    #[must_use]
    pub fn get_stats(&self, category: EventCategory) -> Option<&CategoryStats> {
        self.stats.get(&category)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_flush() {
        let mut collector = Collector::new();

        collector.record_duration(EventCategory::Frame, 16_000_000); // 16ms
        collector.record_duration(EventCategory::Frame, 17_000_000); // 17ms
        collector.record_duration(EventCategory::ClipmapPageBuild, 5_000_000); // 5ms

        collector.flush();

        let frame_stats = collector.get_stats(EventCategory::Frame).unwrap();
        assert_eq!(frame_stats.count, 2);
        assert_eq!(frame_stats.min_ns, 16_000_000);
        assert_eq!(frame_stats.max_ns, 17_000_000);

        let gen_stats = collector
            .get_stats(EventCategory::ClipmapPageBuild)
            .unwrap();
        assert_eq!(gen_stats.count, 1);
    }

    #[test]
    fn snapshot_contains_all_categories() {
        let mut collector = Collector::new();

        collector.record_duration(EventCategory::Frame, 16_000_000);
        collector.record_duration(EventCategory::FrameUpdate, 4_000_000);
        collector.record_duration(EventCategory::FrameRender, 10_000_000);
        collector.flush();

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.categories.len(), 3);
    }

    #[test]
    fn reset_clears_stats() {
        let mut collector = Collector::new();

        collector.record_duration(EventCategory::Frame, 16_000_000);
        collector.flush();

        assert!(collector.get_stats(EventCategory::Frame).is_some());

        collector.reset();

        assert!(collector.get_stats(EventCategory::Frame).is_none());
    }
}
