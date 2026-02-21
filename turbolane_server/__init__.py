"""
turbolane_server — CLI application layer for TurboLane RL-optimized file transfer.

Modules
───────
  protocol  — Binary wire protocol (struct framing, CRC32, message types)
  metrics   — In-app RTT / throughput / loss measurement (no root required)
  adapter   — TurboLaneAdapter: bridges metrics → TurboLaneEngine → stream control
  transfer  — TransferSession + StreamWorker: sender-side parallel TCP streams
  server    — TurboLaneServer: receiver, FileAssembler, single-transfer lock
  sender    — TurboLaneSender: orchestrates transfer + embedded RL
  cli       — argparse CLI: start / send / status subcommands

TurboLane engine is fully decoupled:
  - Only adapter.py imports from turbolane.*
  - No engine code in any other module
"""

__version__ = "2.0.0"
