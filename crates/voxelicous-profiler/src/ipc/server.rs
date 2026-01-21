//! TCP server for profiler IPC.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use parking_lot::Mutex;

use super::protocol::{encode, ClientMessage, ServerMessage, PROTOCOL_VERSION};
use crate::events::ProfilerSnapshot;

/// IPC server that broadcasts profiler data to connected clients.
pub struct IpcServer {
    /// Flag to signal shutdown.
    running: Arc<AtomicBool>,
    /// Server thread handle.
    thread: Option<JoinHandle<()>>,
    /// Connected clients.
    clients: Arc<Mutex<Vec<TcpStream>>>,
}

impl IpcServer {
    /// Start the IPC server on the given port.
    pub fn start(port: u16) -> std::io::Result<Self> {
        let running = Arc::new(AtomicBool::new(true));
        let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));

        let listener = TcpListener::bind(("127.0.0.1", port))?;
        listener.set_nonblocking(true)?;

        let running_clone = Arc::clone(&running);
        let clients_clone = Arc::clone(&clients);

        let thread = thread::Builder::new()
            .name("profiler-ipc".into())
            .spawn(move || {
                Self::server_loop(listener, running_clone, clients_clone);
            })
            .expect("Failed to spawn profiler IPC thread");

        tracing::info!("Profiler IPC server started on port {}", port);

        Ok(Self {
            running,
            thread: Some(thread),
            clients,
        })
    }

    /// Server accept loop.
    fn server_loop(
        listener: TcpListener,
        running: Arc<AtomicBool>,
        clients: Arc<Mutex<Vec<TcpStream>>>,
    ) {
        while running.load(Ordering::Relaxed) {
            // Accept new connections
            match listener.accept() {
                Ok((mut stream, addr)) => {
                    tracing::info!("Profiler client connected: {}", addr);

                    // Set non-blocking for the client stream
                    if let Err(e) = stream.set_nonblocking(true) {
                        tracing::warn!("Failed to set non-blocking: {}", e);
                        continue;
                    }

                    // Set TCP nodelay for lower latency
                    let _ = stream.set_nodelay(true);

                    // Send hello message
                    if let Ok(data) = encode(&ServerMessage::Hello {
                        version: PROTOCOL_VERSION,
                    }) {
                        let _ = stream.write_all(&data);
                    }

                    clients.lock().push(stream);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No pending connections, sleep briefly
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    tracing::warn!("Accept error: {}", e);
                    thread::sleep(Duration::from_millis(100));
                }
            }

            // Process incoming messages from clients
            Self::process_client_messages(&clients);
        }
    }

    /// Process incoming messages from connected clients.
    fn process_client_messages(clients: &Arc<Mutex<Vec<TcpStream>>>) {
        let mut guard = clients.lock();
        let mut to_remove = Vec::new();

        for (i, stream) in guard.iter_mut().enumerate() {
            let mut len_buf = [0u8; 4];
            match stream.read_exact(&mut len_buf) {
                Ok(()) => {
                    let len = u32::from_le_bytes(len_buf) as usize;
                    if len > 1024 * 1024 {
                        // Max 1MB message
                        to_remove.push(i);
                        continue;
                    }

                    let mut data = vec![0u8; len];
                    if stream.read_exact(&mut data).is_ok() {
                        if let Ok(msg) = super::protocol::decode::<ClientMessage>(&data) {
                            match msg {
                                ClientMessage::Hello => {
                                    // Already sent hello on connect
                                }
                                ClientMessage::Reset => {
                                    crate::context::reset();
                                }
                                ClientMessage::Goodbye => {
                                    to_remove.push(i);
                                }
                            }
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No data available
                }
                Err(_) => {
                    // Connection error, mark for removal
                    to_remove.push(i);
                }
            }
        }

        // Remove disconnected clients (in reverse order to maintain indices)
        for i in to_remove.into_iter().rev() {
            guard.remove(i);
        }
    }

    /// Broadcast a snapshot to all connected clients.
    pub fn broadcast(&self, snapshot: ProfilerSnapshot) {
        let data = match encode(&ServerMessage::Snapshot(snapshot)) {
            Ok(d) => d,
            Err(_) => return,
        };

        let mut guard = self.clients.lock();
        let mut to_remove = Vec::new();

        for (i, stream) in guard.iter_mut().enumerate() {
            if stream.write_all(&data).is_err() {
                to_remove.push(i);
            }
        }

        // Remove disconnected clients
        for i in to_remove.into_iter().rev() {
            guard.remove(i);
        }
    }

    /// Stop the server.
    pub fn stop(mut self) {
        self.running.store(false, Ordering::Relaxed);

        // Send goodbye to clients
        if let Ok(data) = encode(&ServerMessage::Goodbye) {
            for stream in self.clients.lock().iter_mut() {
                let _ = stream.write_all(&data);
            }
        }

        // Wait for server thread
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }

        tracing::info!("Profiler IPC server stopped");
    }

    /// Get the number of connected clients.
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.clients.lock().len()
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
