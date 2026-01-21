//! TCP client for connecting to the profiler IPC server.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use voxelicous_profiler::{
    ClientMessage, ProfilerSnapshot, ServerMessage, PROTOCOL_VERSION,
};

/// Connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
}

/// TCP client for profiler IPC.
pub struct ProfilerClient {
    stream: Option<TcpStream>,
    state: ConnectionState,
    host: String,
    port: u16,
    last_snapshot: Option<ProfilerSnapshot>,
}

impl ProfilerClient {
    /// Create a new client.
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            stream: None,
            state: ConnectionState::Disconnected,
            host: host.to_string(),
            port,
            last_snapshot: None,
        }
    }

    /// Get the connection state.
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Get the last received snapshot.
    pub fn last_snapshot(&self) -> Option<&ProfilerSnapshot> {
        self.last_snapshot.as_ref()
    }

    /// Attempt to connect to the server.
    pub fn connect(&mut self) -> anyhow::Result<()> {
        self.state = ConnectionState::Connecting;

        let addr = format!("{}:{}", self.host, self.port);
        let stream = TcpStream::connect_timeout(
            &addr.parse()?,
            Duration::from_secs(2),
        )?;

        stream.set_nonblocking(true)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_millis(10)))?;

        self.stream = Some(stream);
        self.state = ConnectionState::Connected;

        Ok(())
    }

    /// Disconnect from the server.
    pub fn disconnect(&mut self) {
        if let Some(stream) = &mut self.stream {
            // Send goodbye
            if let Ok(data) = encode(&ClientMessage::Goodbye) {
                let _ = stream.write_all(&data);
            }
        }
        self.stream = None;
        self.state = ConnectionState::Disconnected;
    }

    /// Poll for new data from the server.
    ///
    /// Returns `true` if a new snapshot was received.
    pub fn poll(&mut self) -> bool {
        let stream = match &mut self.stream {
            Some(s) => s,
            None => return false,
        };

        // Try to read messages
        loop {
            let mut len_buf = [0u8; 4];
            match stream.read_exact(&mut len_buf) {
                Ok(()) => {
                    let len = u32::from_le_bytes(len_buf) as usize;
                    if len > 10 * 1024 * 1024 {
                        // Sanity check: max 10MB
                        self.disconnect();
                        return false;
                    }

                    let mut data = vec![0u8; len];
                    if stream.read_exact(&mut data).is_err() {
                        self.disconnect();
                        return false;
                    }

                    match bincode::deserialize::<ServerMessage>(&data) {
                        Ok(ServerMessage::Hello { version }) => {
                            if version != PROTOCOL_VERSION {
                                eprintln!(
                                    "Protocol version mismatch: server={}, client={}",
                                    version, PROTOCOL_VERSION
                                );
                                self.disconnect();
                                return false;
                            }
                        }
                        Ok(ServerMessage::Snapshot(snapshot)) => {
                            self.last_snapshot = Some(snapshot);
                            return true;
                        }
                        Ok(ServerMessage::Goodbye) => {
                            self.disconnect();
                            return false;
                        }
                        Err(_) => {
                            // Invalid message, disconnect
                            self.disconnect();
                            return false;
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No more data available
                    return false;
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                    // Timeout, no data
                    return false;
                }
                Err(_) => {
                    // Connection error
                    self.disconnect();
                    return false;
                }
            }
        }
    }

    /// Send a reset command to the server.
    pub fn send_reset(&mut self) -> anyhow::Result<()> {
        if let Some(stream) = &mut self.stream {
            let data = encode(&ClientMessage::Reset)?;
            stream.write_all(&data)?;
        }
        Ok(())
    }
}

/// Encode a message with length prefix.
fn encode<T: serde::Serialize>(msg: &T) -> Result<Vec<u8>, bincode::Error> {
    let data = bincode::serialize(msg)?;
    let len = data.len() as u32;
    let mut result = Vec::with_capacity(4 + data.len());
    result.extend_from_slice(&len.to_le_bytes());
    result.extend(data);
    Ok(result)
}
