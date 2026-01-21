//! Global profiler context singleton.

use parking_lot::Mutex;

use crate::collector::Collector;
use crate::events::{EventCategory, MemoryStats, ProfilerSnapshot, QueueSizes, TimingEvent};
use crate::ipc::server::IpcServer;

/// Global profiler context.
static PROFILER: Mutex<Option<ProfilerContext>> = Mutex::new(None);

/// TCP port for profiler IPC.
pub const DEFAULT_PORT: u16 = 4242;

/// Profiler context that manages event collection and IPC.
pub struct ProfilerContext {
    collector: Collector,
    server: Option<IpcServer>,
}

impl ProfilerContext {
    /// Create a new profiler context.
    fn new() -> Self {
        Self {
            collector: Collector::new(),
            server: None,
        }
    }

    /// Start the IPC server.
    fn start_server(&mut self, port: u16) {
        if self.server.is_none() {
            match IpcServer::start(port) {
                Ok(server) => {
                    self.server = Some(server);
                }
                Err(e) => {
                    tracing::warn!("Failed to start profiler IPC server: {}", e);
                }
            }
        }
    }

    /// Stop the IPC server.
    fn stop_server(&mut self) {
        if let Some(server) = self.server.take() {
            server.stop();
        }
    }
}

/// Initialize the global profiler.
///
/// Call this once at application startup.
pub fn init() {
    init_with_port(DEFAULT_PORT);
}

/// Initialize the global profiler with a custom port.
pub fn init_with_port(port: u16) {
    let mut guard = PROFILER.lock();
    if guard.is_none() {
        let mut ctx = ProfilerContext::new();
        ctx.start_server(port);
        *guard = Some(ctx);
    }
}

/// Shutdown the profiler.
pub fn shutdown() {
    let mut guard = PROFILER.lock();
    if let Some(mut ctx) = guard.take() {
        ctx.stop_server();
    }
}

/// Record a timing event.
#[inline]
pub fn record(event: TimingEvent) {
    if let Some(ctx) = PROFILER.lock().as_ref() {
        ctx.collector.record(event);
    }
}

/// Record a duration for a category.
#[inline]
pub fn record_duration(category: EventCategory, duration_ns: u64) {
    record(TimingEvent::new(category, duration_ns));
}

/// Record a duration with context.
#[inline]
pub fn record_duration_with_context(
    category: EventCategory,
    duration_ns: u64,
    context: [i32; 3],
) {
    record(TimingEvent::with_context(category, duration_ns, context));
}

/// Report queue sizes.
pub fn report_queue_sizes(queues: QueueSizes) {
    if let Some(ctx) = PROFILER.lock().as_mut() {
        ctx.collector.set_queue_sizes(queues);
    }
}

/// Report memory stats.
pub fn report_memory(memory: MemoryStats) {
    if let Some(ctx) = PROFILER.lock().as_mut() {
        ctx.collector.set_memory_stats(memory);
    }
}

/// Mark the end of a frame, flushing events and sending to clients.
pub fn end_frame(frame_number: u64, fps: f32, frame_time_ms: f32) {
    let mut guard = PROFILER.lock();
    if let Some(ctx) = guard.as_mut() {
        ctx.collector.set_frame_info(frame_number, fps, frame_time_ms);
        ctx.collector.flush();

        // Send snapshot to connected clients
        if let Some(server) = &ctx.server {
            let snapshot = ctx.collector.snapshot();
            server.broadcast(snapshot);
        }
    }
}

/// Get a snapshot of current profiling data.
#[must_use]
pub fn snapshot() -> ProfilerSnapshot {
    PROFILER
        .lock()
        .as_ref()
        .map_or_else(ProfilerSnapshot::default, |ctx| ctx.collector.snapshot())
}

/// Reset all profiling statistics.
pub fn reset() {
    if let Some(ctx) = PROFILER.lock().as_mut() {
        ctx.collector.reset();
    }
}

/// Check if the profiler is initialized.
#[must_use]
pub fn is_initialized() -> bool {
    PROFILER.lock().is_some()
}
