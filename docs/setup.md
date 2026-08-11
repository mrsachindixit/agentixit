# Setup

Everything runs locally against [Ollama](https://ollama.com). No OpenAI key, no cloud
account, no billing. Budget about 10 GB of free disk for the models.

All commands are run from the repository root unless stated otherwise.

---

## 1. Install prerequisites

- **Python 3.11+** — https://python.org (3.9 works for most samples; 3.11.3 is what the
  material was written against)
- **Ollama** — https://ollama.com/download

Check both:

```bash
python --version
ollama --version
```

## 2. Create and activate a virtual environment

Use a virtualenv so this course cannot break your other Python projects.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows cmd:

```bat
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers every module including the optional framework samples.
If you only want modules 01–03, `requirements.short.txt` is a much smaller install:

```bash
pip install -r requirements.short.txt
```

## 4. Pull the local models

Minimum to get started:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Additional models used by the advanced samples:

```bash
ollama pull llama3.1:latest
ollama pull lfm2.5-thinking:latest
ollama pull llava
```

Check what you have at any time:

```bash
ollama list
```

## 5. Start Ollama

```bash
ollama serve
```

Leave this running in its own terminal. Everything else assumes it is up on
`localhost:11434`.

## 6. Run your first example

```bash
cd code
python capstones/capstone1_sql_agent/cap1_app.py "List engineering employees with salary > 2000000"
```

You should see a natural-language answer backed by a live SQL query.

---

## Configuration

Every sample reads plain environment variables, and the defaults work out of the box.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Chat / completion model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |

To override, either export them or copy `.env.example` to `.env` in the repository root
and edit it.

PowerShell:

```powershell
$env:OLLAMA_MODEL = "llama3.1:latest"
```

Bash / zsh:

```bash
export OLLAMA_MODEL=llama3.1:latest
```

---

## Running the tests

From the `code/` directory:

```bash
cd code
pytest tests -q
```

By default the suite checks that every sample file exists and compiles. Tests that need a
live model are skipped unless Ollama is running, and the slower end-to-end smoke tests are
opt-in:

```bash
# PowerShell
$env:AGENTICAI_RUN_MODULE_SMOKE = "1"; pytest tests -q
```

```bash
# Bash / zsh
AGENTICAI_RUN_MODULE_SMOKE=1 pytest tests -q
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | Ollama is not running — start it with `ollama serve` |
| `Model not found` | `ollama pull llama3.1` then confirm with `ollama list` |
| Import errors | `pip install --upgrade -r requirements.txt` |
| Missing advanced model | Pull as needed: `llama3.1:latest`, `lfm2.5-thinking:latest`, `llava` |
| A sample cannot find `utils` | Run it from the `code/` directory, not from the module folder |
