# Next steps for a Saudi port (King Abdulaziz Port, Dammam)

Only `config/port_config.yaml` and the three agent datasets change:
- AIS from Mawani / national feeds → same hourly aggregation + berth-occupancy script with the port's real berth polygons.
- Terminal operating system (TOS) exports replace the synthetic cargo layer (yard inventory, gate transactions, crane moves).
- Official berth/crane/gate capacities per terminal → terminal-level twin (multi-operator).
- Saudi calendar (weekend Fri–Sat, Eid, National Day) in `features.py`.
- Score weights from operational cost data; LLM narrative in Arabic for the operations room.
