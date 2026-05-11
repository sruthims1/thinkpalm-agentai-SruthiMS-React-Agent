# ReAct Agent Lab

| Field  | Details                               |
|--------|---------------------------------------|
| Name   | Sruthi M S                       |
| Track  | QA/Testing                  |
| Lab    | ReAct Agent Lab                       |
| Model  | Claude claude-sonnet-4-6 (Anthropic)          |

---

## What It Does

A **ReAct (Reasoning + Acting)** agent powered by Claude Sonnet 4.6 that interleaves step-by-step reasoning with real tool calls — covering math, weather, web search, unit conversion, and Python execution — before delivering a grounded final answer. The agent supports both single-query and interactive conversational modes, with extended thinking enabled to make every reasoning step visible and auditable.

1. **Thinks** — reasons about what information or computation it needs.
2. **Acts** — calls the right tool; never guesses numbers or live facts.
3. **Observes** — reads the tool result and decides if more steps are needed.
4. **Answers** — gives a clear, accurate final answer backed by tool evidence.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ReActAgent.run()                         │
│                                                             │
│  ┌───────────────┐    ┌──────────────────────────────────┐  │
│  │  Claude       │◄──►│   Agentic Loop  (max 10 steps)   │  │
│  │  Sonnet 4.6   │    │                                  │  │
│  │  (extended    │    │  [Step N]                        │  │
│  │   thinking)   │    │  THOUGHT   ← thinking block      │  │
│  └───────────────┘    │  ACTION    ← tool_use block      │  │
│                       │  OBSERVE   ← tool_result         │  │
│                       │      ↓ loop until done           │  │
│                       │  FINAL ANSWER  (end_turn)        │  │
│                       └──────────────────────────────────┘  │
│                                   │                         │
│   Retry + backoff on rate limits  │                         │
└───────────────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
                             Final Answer
```

---

## Tools (7 total)

| Tool              | Description                                             | Example Input                            |
|-------------------|---------------------------------------------------------|------------------------------------------|
| `calculator`      | Math: arithmetic, powers, roots, trig, logarithms       | `"sqrt(256) + 2**10"`                    |
| `get_current_time`| Current local date and time                             | _(no input)_                             |
| `lookup`          | AI/tech concept definitions from knowledge base         | `"ReAct"`, `"rag"`, `"transformer"`      |
| `unit_converter`  | Temperature, distance, and weight conversions           | `100 km → miles`, `37°C → °F`           |
| `web_search`      | DuckDuckGo Instant Answers for real-time facts          | `"What is Claude AI?"`                   |
| `get_weather`     | Live weather for any city via wttr.in                   | `"London"`, `"New York"`                 |
| `run_python`      | Executes Python code in a subprocess (10s timeout)      | Fibonacci, sorting, string processing    |

---

## Project Structure

```
React Agent/
├── src/
│   ├── react_agent.py      # ReActAgent class, agentic loop, CLI
│   └── tools.py            # All 7 tool schemas + implementations
├── react_agent_chat.ipynb  # Interactive conversational mode notebook
├── react_agent_colab.ipynb # Colab demo notebook
├── screenshots/            # Output screenshots
├── README.md
├── requirements.txt
└── .env.example
```

---

## How to Run

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env:  ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run the agent

**Interactive conversational mode (default):**
```bash
python src/react_agent.py
```
Starts a multi-turn chat session. Type `exit` or `quit` to end.

**Single custom query:**
```bash
python src/react_agent.py -q "What is the square root of 144?"
python src/react_agent.py -q "Weather in Tokyo?"
python src/react_agent.py -q "Write Python to sort a list of strings alphabetically"
```

**All 8 demo queries:**
```bash
python src/react_agent.py --demo
```

**Flags:**
```bash
--verbose / -v    Show extended Thought blocks
--quiet           Print only final answers
--no-save         Skip writing trace files
```

---

## Sample Output

```
══════════════════════════════════════════════════════════════
QUERY: Calculate (15 * 7) + sqrt(81), then convert 37°C to Fahrenheit.
══════════════════════════════════════════════════════════════

[Step 1]  ACTION
──────────────────────────────────────────────────────────────
Tool  : calculator
Input : {"expression": "(15 * 7) + sqrt(81)"}

[Step 1]  OBSERVATION
──────────────────────────────────────────────────────────────
114.0

[Step 2]  ACTION
──────────────────────────────────────────────────────────────
Tool  : unit_converter
Input : {"value": 37, "from_unit": "celsius", "to_unit": "fahrenheit"}

[Step 2]  OBSERVATION
──────────────────────────────────────────────────────────────
37 celsius = 98.6000 fahrenheit

══════════════════════════════════════════════════════════════
FINAL ANSWER
──────────────────────────────────────────────────────────────
(15 × 7) + √81 = 105 + 9 = **114**
37°C = **98.6°F**
══════════════════════════════════════════════════════════════

╔══════════════════════════════════════════════════════════════╗
║                    EXECUTION SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Steps taken  : 2                                            ║
║  Tools called : calculator×1, unit_converter×1               ║
║  Elapsed time : 5.12s                                        ║
║  Status       : ✓ Success                                    ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Observations

1. **Structured ReAct loop** — Claude consistently separates Thought (adaptive thinking), Action (tool_use block), and Observation (tool_result), making the chain fully auditable.

2. **Multi-tool queries** — Compound questions (e.g., calculate + convert) are handled across sequential steps without mixing results.

3. **Grounded answers** — The `calculator` and `run_python` tools eliminate hallucinated math; `web_search` and `get_weather` anchor real-time facts.

4. **Extended thinking** — `claude-sonnet-4-6` always generates a Thought block (budget: 8000 tokens), making every reasoning step visible and auditable.

5. **Robustness** — Exponential backoff handles rate limits; tool errors return structured messages so Claude can retry with a different approach.

6. **Screenshots** — Colab output screenshots are saved in the `screenshots/` folder as submission evidence.

---

## Screenshots

Google Colab output screenshots are saved in the `screenshots/` folder as submission evidence.
