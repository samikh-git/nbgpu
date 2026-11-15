# nbgpu

Run a local [marimo](https://marimo.io) notebook on a remote GPU. The notebook
declares the compute it needs in its own header; `nbgpu start` provisions it,
syncs the project both ways, and serves the marimo UI on a local URL.

Work in progress. The kernel-tunnel prototype for VS Code lives in `shim/`.

## Development

```bash
uv sync
uv run pytest                 # unit, contract (local backend) and e2e tests
NBGPU_MODAL_TESTS=1 uv run pytest -m modal   # creates real Modal sandboxes
```
