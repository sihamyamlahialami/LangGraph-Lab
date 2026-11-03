# Intelligent University Assistant — LangGraph

A PhD practical lab project demonstrating LangGraph nodes and conditional edges, specialized LLM agents, tool calling, and thread-based conversational memory.

## Verify files without installing anything

Run `python verify_package.py` to check source syntax, SQLite contents, and diagram integrity. This is a structural check, not a live Groq test.

## Quick start

### Google Colab
1. Open `langgraph_lab.ipynb`.
2. If using the standalone source, upload `university_assistant.py`, `requirements.txt`, and `university_courses_demo.db` to the Colab Files pane.
3. Execute `%pip install -r requirements.txt` in a notebook cell, then `!python university_assistant.py --self-test`.
4. Add `GROQ_API_KEY` as an environment variable, or enter it securely when prompted; run `!python university_assistant.py --demo` for LLM tests.

### On your PC
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python university_assistant.py --self-test
python university_assistant.py --demo
```
`--self-test` checks the SQLite catalog and calculator without an API key. `--demo` calls Groq, requires a valid key and internet access, and may encounter rate limits.

## Design choices

- **LLM**: Groq `openai/gpt-oss-20b`; low reasoning effort, deterministic generation (`temperature=0`).
- **Classifier**: one LLM node routes latest user message to COURSES, PROGRAMMING, MATHEMATICS, ADMINISTRATION, GENERAL. Empty input goes to fallback.
- **Specialist agents**: one node per category with distinct system instructions.
- **Tools**: `calculator` (AST-based arithmetic; no Python `eval`), `current_datetime`, and `search_courses` (SQLite).
- **Tools registration order**: all 3 are defined and included in `tools` *before* `llm.bind_tools(tools)`.
- **Tool flow**: `ToolNode` executes calls and returns to the same category agent for final response.
- **Memory**: `MemorySaver` checkpoint keyed by `thread_id`; not persistent across application restarts.
- **Database**: 5 **fictional** course records. The database is example data, not official university information.
- **Error handling**: empty messages, invalid categories, invalid expressions, Groq errors, and tool errors.

## Testing and observed results

- Router: 6/6 correct (100%) on six questions, according to the notebook outputs shared in chat.
- Capstone: five specialist response categories and empty-input handling were demonstrated.
- Memory: same-thread question successfully recalled `Ahmed`.
- Calculator: visible ToolMessage contained `126` for `(15 + 27) * 3`.
- SQLite extension: source and fictional records are included; **the final Groq/SQLite tool-call test has not been observed yet**.
- Date/time: answer shown; **the actual `current_datetime` tool call has not yet been independently verified**.

## Limitations

Small test set; nonofficial university records; API rate limits; volatile in-memory checkpointer; no quality guarantee outside the observed test set.