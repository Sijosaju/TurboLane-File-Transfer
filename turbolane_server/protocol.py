"""
turbolane_server/protocol.py

Binary wire protocol for TurboLane file transfers.

Frame layout (all fields big-endian):
┌──────────────┬───────┬───────────────────────────────────────────────────┐
│ Field        │ Bytes │ Description                                       │
├──────────────┼───────┼───────────────────────────────────────────────────┤
│ magic        │   4   │ 0x544C414E  ("TLAN") — frame start sentinel       │
│ msg_type     │   1   │ MessageType enum                                   │
│ stream_id    │   1   │ Which parallel stream (0-255)                      │
│ chunk_idx    │   4   │ This chunk's index within the file                 │
│ total_chunks │   4   │ Total number of chunks in the transfer             │
│ file_offset  │   8   │ Byte offset in the original file                   │
│ data_len     │   4   │ Payload length in bytes (0 for control frames)     │
│ checksum     │   4   │ CRC32 of the payload (0 for control frames)        │
├──────────────┼───────┼───────────────────────────────────────────────────┤
│ TOTAL HEADER │  30   │                                                    │
└──────────────┴───────┴───────────────────────────────────────────────────┘
│ payload      │ data_len bytes │ Raw file chunk bytes                      │
└──────────────┴────────────────┴───────────────────────────────────────────┘

Message types:
  HELLO        — client initiates: sends file metadata (name, total_size, num_streams)
  HELLO_ACK    — server confirms ready
  CHUNK        — file data frame
  CHUNK_ACK    — server acknowledges a chunk (used for RTT measurement)
  PING         — RTT probe
  PONG         — RTT response
  TRANSFER_DONE — sender signals all chunks sent
  COMPLETE     — server signals file fully assembled
  ERROR        — either side reports an error
  BUSY         — server rejects because a transfer is active
"""

import struct
import zlib
import socket
from enum import IntEnum

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MAGIC = 0x544C414E          # "TLAN"
HEADER_FORMAT = "!IBBIIIQII"  # big-endian: magic(I) type(B) stream_id(B) chunk_idx(I) total_chunks(I) seq(I) file_offset(Q) data_len(I) checksum(I)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 30 bytes

CHUNK_SIZE = 1 * 1024 * 1024   # 1 MB default chunk size
RECV_BUFFER = 65536             # socket recv buffer


class MessageType(IntEnum):
    HELLO        = 0x01
    HELLO_ACK    = 0x02
    CHUNK        = 0x03
    CHUNK_ACK    = 0x04
    PING         = 0x05
    PONG         = 0x06
    TRANSFER_DONE = 0x07
    COMPLETE     = 0x08
    ERROR        = 0x09
    BUSY         = 0x0A
    STATUS_REQ   = 0x0B
    STATUS_RESP  = 0x0C


# ------------------------------------------------------------------
# Frame encode / decode
# ------------------------------------------------------------------

def encode_frame(
    msg_type: MessageType,
    stream_id: int = 0,
    chunk_idx: int = 0,
    total_chunks: int = 0,
    seq: int = 0,
    file_offset: int = 0,
    payload: bytes = b"",
) -> bytes:
    """
    Pack a frame into bytes ready to send over a TCP socket.

    Args:
        msg_type:     MessageType enum value
        stream_id:    Which parallel stream (0-indexed)
        chunk_idx:    Chunk index within the file
        total_chunks: Total chunks in this transfer
        seq:          Sequence number (for ordering / ACK correlation)
        file_offset:  Byte offset in the source file
        payload:      Raw bytes to send (empty for control frames)

    Returns:
        bytes: header + payload
    """
    data_len = len(payload)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF if payload else 0

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        int(msg_type),
        stream_id & 0xFF,
        chunk_idx,
        total_chunks,
        seq,
        file_offset,
        data_len,
        checksum,
    )
    return header + payload


def decode_header(raw: bytes) -> dict:
    """
    Unpack a raw 30-byte header into a dict.

    Raises:
        ValueError: if magic bytes don't match or header is truncated
    """
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"Header too short: got {len(raw)}, need {HEADER_SIZE}")

    fields = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])
    magic, msg_type, stream_id, chunk_idx, total_chunks, seq, file_offset, data_len, checksum = fields

    if magic != MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")

    return {
        "msg_type":     MessageType(msg_type),
        "stream_id":    stream_id,
        "chunk_idx":    chunk_idx,
        "total_chunks": total_chunks,
        "seq":          seq,
        "file_offset":  file_offset,
        "data_len":     data_len,
        "checksum":     checksum,
    }


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Read exactly n bytes from sock.

    Raises:
        ConnectionError: if the socket closes before n bytes are read
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), RECV_BUFFER))
        if not chunk:
            raise ConnectionError(f"Socket closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    """
    Read one complete frame (header + payload) from sock.

    Returns:
        (header_dict, payload_bytes)

    Raises:
        ValueError: bad magic / truncated header
        ConnectionError: socket closed mid-read
    """
    raw_header = recv_exact(sock, HEADER_SIZE)
    hdr = decode_header(raw_header)

    payload = b""
    if hdr["data_len"] > 0:
        payload = recv_exact(sock, hdr["data_len"])
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != hdr["checksum"]:
            raise ValueError(
                f"CRC mismatch on chunk {hdr['chunk_idx']}: "
                f"expected 0x{hdr['checksum']:08X}, got 0x{actual_crc:08X}"
            )

    return hdr, payload


# ------------------------------------------------------------------
# Helper: encode a JSON-like metadata payload as UTF-8
# (used in HELLO / ERROR / STATUS frames — no heavy deps)
# ------------------------------------------------------------------
import json

def encode_meta(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")

def decode_meta(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))
