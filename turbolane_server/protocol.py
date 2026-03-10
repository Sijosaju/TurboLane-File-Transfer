"""
turbolane_server/protocol.py

Binary wire protocol for TurboLane file transfers.

Frame layout (all fields big-endian):
┌──────────────┬───────┬───────────────────────────────────────────────────┐
│ Field        │ Bytes │ Description                                       │
├──────────────┼───────┼───────────────────────────────────────────────────┤
│ magic        │   4   │ 0x544C414E  ("TLAN") — frame start sentinel       │
│ msg_type     │   1   │ MessageType enum                                   │
│ stream_id    │   2   │ Which parallel stream (0-65535)                    │
│ chunk_idx    │   4   │ This chunk's index within the file                 │
│ total_chunks │   4   │ Total number of chunks in the transfer             │
│ seq          │   4   │ Sequence number                                    │
│ file_offset  │   8   │ Byte offset in the original file                   │
│ data_len     │   4   │ Payload length in bytes (0 for control frames)     │
│ checksum     │   4   │ CRC32 of the payload (0 for control frames)        │
├──────────────┼───────┼───────────────────────────────────────────────────┤
│ TOTAL HEADER │  35   │                                                   │
└──────────────┴───────┴───────────────────────────────────────────────────┘
│ payload      │ data_len bytes │ Raw file chunk bytes                      │
└──────────────┴────────────────┴───────────────────────────────────────────┘

Format string breakdown — all big-endian (!):
  I  = magic        (4 bytes)
  B  = msg_type     (1 byte)
  H  = stream_id    (2 bytes)
  I  = chunk_idx    (4 bytes)
  I  = total_chunks (4 bytes)
  I  = seq          (4 bytes)
  Q  = file_offset  (8 bytes)
  I  = data_len     (4 bytes)
  I  = checksum     (4 bytes)
Total: 4+1+2+4+4+4+8+4+4 = 35 bytes
"""

import struct
import zlib
import socket
import json
from enum import IntEnum

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MAGIC = 0x544C414E          # "TLAN"

HEADER_FORMAT = "!IBHIIIQII"
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)   # 35 bytes

# BUG B FIX: reduced from 1 MB to 256 KB.
#
# With 1 MB chunks and 27+ parallel streams, the sender pushes
# 27+ MB into the Wi-Fi send buffer simultaneously. This causes
# severe bufferbloat: measured RTT jumped from ~4 ms to ~30,000 ms
# (29.8 s), which is right at ACK_TIMEOUT (30 s), triggering a
# cascade of ACK timeouts, stream deaths, and endless respawning.
#
# 256 KB × 32 streams = 8 MB max in-flight — a much more manageable
# burst size for a typical Wi-Fi LAN (80-300 Mbps).  The total
# number of chunks increases (×4) but each chunk round-trips ~4× 
# faster, keeping RTT low and the RL adapter's measurements accurate.
#
# IMPORTANT: both sender and receiver must use the same protocol.py.
# The chunk_size is negotiated in the HELLO frame so an existing
# transfer is not affected by this change mid-flight.
CHUNK_SIZE  = 256 * 1024    # 256 KB (keeps ACK latency and Wi-Fi queueing under control)
RECV_BUFFER = 256 * 1024    # 256 KB


class MessageType(IntEnum):
    HELLO         = 0x01
    HELLO_ACK     = 0x02
    CHUNK         = 0x03
    CHUNK_ACK     = 0x04
    PING          = 0x05
    PONG          = 0x06
    TRANSFER_DONE = 0x07
    COMPLETE      = 0x08
    ERROR         = 0x09
    BUSY          = 0x0A
    STATUS_REQ    = 0x0B
    STATUS_RESP   = 0x0C


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
    data_len = len(payload)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF if payload else 0

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        int(msg_type),
        stream_id & 0xFFFF,
        chunk_idx,
        total_chunks,
        seq,
        file_offset,
        data_len,
        checksum,
    )
    return header + payload


def decode_header(raw: bytes) -> dict:
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
    """Read exactly n bytes from sock."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), RECV_BUFFER))
        if not chunk:
            raise ConnectionError(f"Socket closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    """Read one complete frame (header + payload) from sock."""
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
# Metadata helpers
# ------------------------------------------------------------------

def encode_meta(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")

def decode_meta(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))
