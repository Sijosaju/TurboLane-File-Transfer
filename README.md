# TurboLane — Phase 2: CLI File Transfer Server

RL-optimized parallel TCP file transfer system. Phase 2 builds a production-grade
CLI application on top of the Phase 1 TurboLane engine.

---

## Folder Structure

```
.
├── turbolane/                  # Phase 1 — RL engine (UNCHANGED)
│   ├── __init__.py
│   ├── engine.py               # TurboLaneEngine — only public import
│   ├── policies/
│   │   └── federated.py        # FederatedPolicy (DCI / Q-learning)
│   └── rl/
│       ├── agent.py            # RLAgent (Q-table, Bellman updates)
│       └── storage.py          # QTableStorage (atomic JSON persistence)
│
├── turbolane_server/           # Phase 2 — CLI application layer
│   ├── __init__.py
│   ├── protocol.py             # Binary wire protocol (struct + CRC32)
│   ├── metrics.py              # In-app RTT / throughput / loss metrics
│   ├── adapter.py              # TurboLaneAdapter — engine ↔ server bridge
│   ├── transfer.py             # TransferSession + StreamWorker (sender side)
│   ├── server.py               # TurboLaneServer + FileAssembler (receiver)
│   ├── sender.py               # TurboLaneSender — orchestrator
│   └── cli.py                  # argparse CLI: start / send / status
│
├── models/
│   └── dci/                    # Q-table persistence directory
├── setup.py
└── README.md
```

---

## Architecture

```
SENDER SIDE                          RECEIVER SIDE
───────────────────────────────      ────────────────────────────
  turbolane-server send               turbolane-server start
        │                                     │
  TurboLaneSender                    TurboLaneServer
        │                                     │
  ┌─────────────────┐                ┌────────────────────┐
  │ TurboLaneAdapter│                │   accept loop      │
  │  (5s RL loop)   │                │   (one thread/conn)│
  │                 │                └────────────────────┘
  │ TurboLaneEngine │                         │
  │ (embedded DCI)  │                ┌────────────────────┐
  └────────┬────────┘                │  StreamHandler     │
           │ adjust_streams(n)       │  HELLO/CHUNK/PING  │
           ▼                         └────────────────────┘
  TransferSession                              │
  ┌─────────────────────────────┐    ┌────────────────────┐
  │  ChunkQueue (thread-safe)   │    │  FileAssembler     │
  │  StreamWorker × N           │    │  (sparse write,    │
  │  (one thread per stream)    │    │   out-of-order OK) │
  └─────────────────────────────┘    └────────────────────┘
           │ N×TCP connections
           └──────────────────────────────────────────────┘

MetricsCollector (shared)
  ├── per-stream StreamMetrics
  ├── RTT: in-app PING/PONG round-trip timing
  ├── Throughput: bytes_sent / elapsed per snapshot
  └── Loss: chunk retransmit rate proxy
```

### Key design rules
- **TurboLane engine is completely decoupled** — only `adapter.py` imports from `turbolane.*`
- **No networking in the engine** — sockets live only in `transfer.py` and `server.py`
- **Single transfer lock** — server rejects new connections with `BUSY` during active transfer
- **RTT without root** — measured via application-layer PING/PONG timing per stream

---

## Wire Protocol

Binary struct header (34 bytes, big-endian) + payload:

| Field        | Bytes | Description                          |
|--------------|-------|--------------------------------------|
| magic        | 4     | `0x544C414E` ("TLAN")                |
| msg_type     | 1     | MessageType enum                     |
| stream_id    | 1     | Parallel stream index (0-255)        |
| chunk_idx    | 4     | Chunk index within file              |
| total_chunks | 4     | Total chunks in transfer             |
| seq          | 4     | Sequence number                      |
| file_offset  | 8     | Byte offset in source file           |
| data_len     | 4     | Payload length (0 for control)       |
| checksum     | 4     | CRC32 of payload                     |
| **payload**  | N     | Raw file bytes / JSON metadata       |

Message types: `HELLO`, `HELLO_ACK`, `CHUNK`, `CHUNK_ACK`, `PING`, `PONG`,
`TRANSFER_DONE`, `COMPLETE`, `ERROR`, `BUSY`, `STATUS_REQ`, `STATUS_RESP`

---

## Installation

```bash
# From the project root (where setup.py lives)
pip install -e .
```

---

## Usage

### 1. Start the receiver server

```bash
turbolane-server start --port 9000 --output-dir ./received
```

Options:
```
--host HOST         Bind interface (default: 0.0.0.0)
--port PORT         TCP port (default: 9000)
--output-dir DIR    Where to save received files (default: ./received)
--verbose / -v      Debug logging
```

### 2. Send a file

```bash
turbolane-server send /data/large_dataset.tar \
    --host 192.168.1.50 --port 9000 \
    --streams 6 \
    --min-streams 1 --max-streams 32 \
    --model-dir models/dci \
    --interval 5.0
```

Options:
```
FILE                File to send (required positional)
--host HOST         Receiver hostname/IP (required)
--port PORT         Receiver port (default: 9000)
--streams N         Initial parallel TCP streams (default: 4)
--min-streams N     Minimum streams TurboLane may use (default: 1)
--max-streams N     Maximum streams TurboLane may use (default: 32)
--model-dir DIR     Q-table persistence directory (default: models/dci)
--interval SECS     RL decision interval in seconds (default: 5.0)
--timeout SECS      Max wait for completion (default: unlimited)
--verbose / -v      Debug logging
```

### 3. Query server status

```bash
turbolane-server status --host 192.168.1.50 --port 9000
```

---

## How TurboLane integrates (5-second loop)

```
Every 5 seconds (TurboLaneAdapter._tick):
  1. MetricsCollector.snapshot()
       → throughput_mbps (sum of per-stream byte rates)
       → rtt_ms          (mean of PING/PONG RTT samples)
       → loss_pct        (chunk retransmit rate proxy)

  2. engine.learn(throughput, rtt, loss)
       → Q-table Bellman update from previous decision's outcome

  3. engine.decide(throughput, rtt, loss)
       → Q-learning ε-greedy action → new stream count

  4. session.adjust_streams(new_count)
       → spawn / stop StreamWorker threads to match recommendation
```

---

## Future upgrades (designed-in hooks)

| Capability           | Where to add              |
|----------------------|---------------------------|
| PPO algorithm        | `turbolane/rl/` only      |
| Multi-session        | `server.py` busy logic    |
| Shared policy learning | `adapter.py` FederatedPolicy |
| Resume / checkpointing | `ChunkQueue` + `FileAssembler` |
| TLS encryption       | `StreamWorker` + `StreamHandler` |
