//! Wire protocol for profiler IPC.

use serde::{Deserialize, Serialize};

use crate::events::ProfilerSnapshot;

/// Protocol version for compatibility checking.
pub const PROTOCOL_VERSION: u8 = 1;

/// Message from server to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerMessage {
    /// Protocol version handshake.
    Hello { version: u8 },
    /// Profiling data snapshot.
    Snapshot(ProfilerSnapshot),
    /// Server is shutting down.
    Goodbye,
}

/// Message from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    /// Request protocol version.
    Hello,
    /// Request stats reset.
    Reset,
    /// Client disconnecting.
    Goodbye,
}

/// Encode a message to bytes with length prefix.
pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, bincode::Error> {
    let data = bincode::serialize(msg)?;
    let len = data.len() as u32;
    let mut result = Vec::with_capacity(4 + data.len());
    result.extend_from_slice(&len.to_le_bytes());
    result.extend(data);
    Ok(result)
}

/// Decode a message from bytes (without length prefix).
pub fn decode<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T, bincode::Error> {
    bincode::deserialize(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_server_message() {
        let msg = ServerMessage::Hello {
            version: PROTOCOL_VERSION,
        };
        let encoded = encode(&msg).unwrap();

        // Skip length prefix (4 bytes)
        let decoded: ServerMessage = decode(&encoded[4..]).unwrap();

        match decoded {
            ServerMessage::Hello { version } => assert_eq!(version, PROTOCOL_VERSION),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn roundtrip_client_message() {
        let msg = ClientMessage::Reset;
        let encoded = encode(&msg).unwrap();
        let decoded: ClientMessage = decode(&encoded[4..]).unwrap();

        assert!(matches!(decoded, ClientMessage::Reset));
    }
}
