# Agent-native delivery record

The embedded Agent overhaul has shipped. The [design guide](agent-first-design.md) describes the current product, the [architecture reference](architecture.md) explains the implementation, and the [panel playbook](panel-development.md) covers extensions.

| Phase | Delivered |
| --- | --- |
| Session and initial integration | Embedded Pi RPC, visible progress, workspace attachments, and user approval of panel proposals. |
| Panel contract | Panel-owned operations across Electron and Ion, after-run form adoption, and a TPX3 Dump → Ion TOF workflow with an optional GMD2 gate and alignment checkpoint. The Electron Pipeline placeholder was retired. |
| Evidence | Screenshot input for image-capable models, panel captures, and exact measurements for linked Ion TOF runs. |
| User-authored analysis | Approved Python runs with records and artifacts, declarative analysis forms, saved workspace panels, and a separate container worker in shared mode. |
| Deployment | One-server local, manual-only shared, and sandboxed shared options. Workspace protection for shared subprocesses and file transfers. |

## Remaining work

- Validate a full pilot on the lab's production Linux host with its actual service account, model endpoint, acquisition workload, and chosen container runtime. Local and simulated tests cover parts of this setup.
- Extend panel coverage when a real workflow needs a new operation, view hook, or measured result. The capability matrix lists the current limits.
- Keep the ion fallback tools available for manual comparison.
- Revisit durable conversations, restartable checkpoints, and external Agent integrations if product requirements call for them. Current recovery uses reconnect within the active server process and saved files after restart.
