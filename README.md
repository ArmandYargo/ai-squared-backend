# RAM LangGraph Agent (Spyder/Anaconda-friendly)

This project wraps your existing RAM / input-sheet pipeline into a **stateful LangGraph agent** with:
- an **Orchestrator** (intent routing + conversation flow)
- a **RAG knowledge base** (manuals/regulations) with *smart chunking*
- **persistent memory** (SQLite checkpoints)
- a **RAM tool agent** that calls your existing `func_*.py` modules

## 1) Install (in your Anaconda environment)

Open **Anaconda Prompt** (or the terminal inside Navigator) and run:

```bash
cd /path/to/ram_langgraph_agent
pip install -r requirements.txt
```

> If you cannot install some packages on Windows, start with:
> `pip install -U langgraph openai pandas openpyxl python-dotenv`
> and we can swap the RAG backend to something lighter.

## 2) Set your OpenAI key (required by your existing RAM modules)

Create a file named `.env` in the project root, with:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.2
```

## 3) Run

From Spyder:
- open `run_agent.py`
- press **Run**

Or from terminal:

```bash
python run_agent.py
```

## 4) How to use (chat examples)

- **Run RAM inputs**
  - "Build a RAM input sheet for conveyors, date range 2023-2024"
  - The agent will open a file dialog to pick the SAP/CMMS Excel, propose coarse categories, apply your edits, then export the classified workbook and input sheet.

- **Ingest manuals/regulations**
  - "Ingest manuals from this folder"
  - It will ask you to select files, then build a searchable knowledge base.

- **Ask questions**
  - "Explain what Beta and Eta mean in the output"
  - "Why was this classified as drive_gearbox?"
  - It will answer using conversation context + RAG (when available).

## 5) Outputs

All outputs are written into `outputs/` by default:
- `*_classified.xlsx`
- `*_input.xlsx`
- `ram_input_sheet.xlsx` (latest export copy)

## Notes
- Your original scripts are included under `ram_module/` unchanged.
- The agent uses a non-interactive wrapper (`ram_module/ram_pipeline.py`) so the CLI prompts become chat steps.
