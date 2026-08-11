# MCP Inspector Quick Guide (6.3 only)

From repo root:

```powershell
cd agenticai_coderepo
```

## 1) Run MCP server (6.3)

```powershell
uv run --with "mcp[cli]" python module06_enterprise/6.3_mcp_sdk_server.py
```

If you see `ModuleNotFoundError: No module named 'mcp'`, use the same command above (it bootstraps `mcp` automatically).

Server endpoint:
- `http://localhost:8000/mcp`

## 2) Run official MCP Inspector (separate terminal)

```powershell
npx @modelcontextprotocol/inspector
```

This command starts the Inspector app (it prints a local URL to open in browser).

## 3) Connect Inspector to server

Use this exact sequence:

1. Keep Terminal A running `6.3`.
2. In Terminal B run `npx @modelcontextprotocol/inspector`.
3. Open the Inspector URL shown in Terminal B.
4. In Inspector choose transport `http` (Streamable HTTP).
5. Set server URL to `http://localhost:8000/mcp`.
6. Click **Initialize**.
7. Open Tools / Resources / Prompts tabs and invoke calls.

Optional (prefilled connect from CLI):

```powershell
npx @modelcontextprotocol/inspector --transport http --server-url http://localhost:8000/mcp
```

Note: Inspector runs as a separate process and connects to the already-running `6.3` server.

## 4) Single command (teaching shortcut)

Run server and Inspector together in one line:

**PowerShell:**
```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command','uv run --with "mcp[cli]" python .\6.3_mcp_sdk_server.py'; Start-Sleep -Seconds 4; npx @modelcontextprotocol/inspector --transport http --server-url http://localhost:8000/mcp
```

**bash / Git Bash:**
```bash
uv run --with "mcp[cli]" python ./6.3_mcp_sdk_server.py &
sleep 2 && npx @modelcontextprotocol/inspector
```

Both start the server in the background, wait 2 seconds for it to be ready, then open the Inspector.
After Inspector opens, use transport `http` and URL `http://localhost:8000/mcp`.
