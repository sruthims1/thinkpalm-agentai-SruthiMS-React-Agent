"""ReAct (Reasoning + Acting) agent."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── .env loader ────────────────────────────────────────────────────────────────
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    raise SystemExit(
        "\n[ERROR] ANTHROPIC_API_KEY not set.\n"
        "Create a .env file in the project root:\n"
        "  ANTHROPIC_API_KEY=sk-ant-your-key-here\n"
    )

import anthropic
from tools import TOOL_SCHEMAS, execute_tool

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL       = "claude-sonnet-4-6"
MAX_STEPS   = 10
MAX_RETRIES = 3
TRACE_DIR   = Path(__file__).parent.parent / "screenshots"

SYSTEM_PROMPT = """\
You are a ReAct (Reasoning + Acting) agent with access to a suite of powerful tools.

For every user question, follow this exact loop:
1. THINK  — Reason step by step. What do you need to find out or compute?
2. ACT    — Call the most appropriate tool. NEVER guess numbers, dates, or live facts.
3. OBSERVE — Examine the result. Is the question fully answered? If not, repeat.
4. ANSWER — Once fully informed, give a clear, accurate, complete final answer.

Tool selection guide:
  calculator      → arithmetic, powers, roots, trig, logarithms
  get_current_time → current date and time
  lookup          → definitions of AI/tech concepts (claude, react, llm, rag, etc.)
  unit_converter  → temperature, distance, weight conversions
  web_search      → real-time facts, news, or anything outside the knowledge base
  get_weather     → live weather for any city
  run_python      → complex algorithms, list/string processing, multi-step computations

Rules:
  - Always use a tool for math — never compute in your head.
  - Always use web_search for real-time or unknown information.
  - If a tool returns an error, try a different tool or rephrase the input.
  - Be explicit and transparent in your reasoning so users can follow along.
"""

DEMO_QUERIES = [
    "What is 2 raised to the power of 10?",
    "What is the current date and time?",
    "What is the square root of 256, and what does 'ReAct' mean in AI?",
    "Convert 100 km to miles, and tell me what Anthropic is.",
    "Calculate (15 * 7) + sqrt(81), then convert 37 degrees Celsius to Fahrenheit.",
    "Search the web: what is Claude AI?",
    "What's the weather like in London right now?",
    "Write and run Python code to generate the first 10 Fibonacci numbers.",
]

# ── Output helpers ─────────────────────────────────────────────────────────────
_W         = 62          # inner box width
_DIVIDER   = "─" * _W
_VERBOSE   = False
_QUIET     = False
_CHAT_MODE = False


def _emit(text: str, lines: list, display: str | None = None) -> None:
    """Append full text to trace; print display version (or full text) unless quiet mode suppresses it."""
    lines.append(text)
    shown = display if display is not None else text

    if _CHAT_MODE:
        if _VERBOSE:
            # Show full THOUGHT / ACTION / OBSERVATION steps.
            # Suppress only the QUERY header, FINAL ANSWER section, and EXECUTION SUMMARY.
            if "FINAL ANSWER" in text:
                return
            if text.strip().startswith("╔"):
                return
            # Suppress the ══ QUERY: ══ banner (starts with newline then ═)
            stripped = text.lstrip("\n")
            if stripped.startswith("═") and "QUERY:" in text:
                return
            print(shown)
        else:
            # Compact mode: one-line tool indicator only.
            if "ACTION" in text and "Tool  :" in text:
                for line in text.splitlines():
                    if line.strip().startswith("Tool"):
                        tool_name = line.split(":", 1)[-1].strip()
                        print(f"  [Using {tool_name}...]")
                        break
        return

    if _QUIET:
        if "FINAL ANSWER" in text or text.strip().startswith("╔"):
            print(shown)
        return
    print(shown)


def _box(label: str, content: str) -> str:
    """Format a labelled section block for terminal and trace output."""
    return f"\n{label}\n{_DIVIDER}\n{content}\n"


def _summary_box(step: int, tools_summary: str, elapsed: float, status: str) -> str:
    """Render a Unicode box table summarising one query's execution stats."""
    def row(content: str) -> str:
        return f"║ {content:<{_W - 2}} ║"

    return "\n".join([
        "",
        f"╔{'═' * _W}╗",
        row(f"{'EXECUTION SUMMARY':^{_W - 2}}"),
        f"╠{'═' * _W}╣",
        row(f"  Steps taken  : {step}"),
        row(f"  Tools called : {tools_summary}"),
        row(f"  Elapsed time : {elapsed:.2f}s"),
        row(f"  Status       : {status}"),
        f"╚{'═' * _W}╝",
    ])


# ── PNG renderer ───────────────────────────────────────────────────────────────

def _save_png(trace_lines: list[str], path: Path) -> bool:
    """Render the trace as a dark-themed terminal PNG. Returns True on success."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    # ── Font ──────────────────────────────────────────────────────────────────
    font_size = 14
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    for fp in [
        "C:/Windows/Fonts/consola.ttf",   # Consolas (Windows)
        "C:/Windows/Fonts/cour.ttf",      # Courier New (Windows)
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (IOError, OSError):
            pass
    if font is None:
        font = ImageFont.load_default()

    # ── Color scheme (VS Code Dark+) ──────────────────────────────────────────
    BG          = (30,  30,  30)
    C_DEFAULT   = (212, 212, 212)
    C_BORDER    = ( 80,  80,  90)
    C_QUERY     = ( 86, 156, 214)   # blue   — QUERY banner
    C_THOUGHT   = (156, 220, 254)   # sky    — THOUGHT
    C_ACTION    = (206, 145, 120)   # orange — ACTION
    C_OBSERVE   = (106, 153,  85)   # green  — OBSERVATION
    C_ANSWER    = ( 78, 201, 176)   # teal   — FINAL ANSWER
    C_SUMMARY   = (220, 220, 100)   # yellow — EXECUTION SUMMARY

    SECTION_COLORS = {
        "thought":  C_THOUGHT,
        "action":   C_ACTION,
        "observe":  C_OBSERVE,
        "answer":   C_ANSWER,
        "summary":  C_SUMMARY,
    }

    def _color(line: str, section: str) -> tuple:
        stripped = line.strip()
        if not stripped:
            return C_DEFAULT
        if all(c in "═─╔╗╠╣╚╝║ " for c in stripped):
            return C_BORDER
        if "QUERY:" in line:
            return C_QUERY
        if "THOUGHT" in line and line.strip().startswith("[Step"):
            return C_THOUGHT
        if "ACTION" in line and line.strip().startswith("[Step"):
            return C_ACTION
        if "OBSERVATION" in line and line.strip().startswith("[Step"):
            return C_OBSERVE
        if "FINAL ANSWER" in line:
            return C_ANSWER
        if "EXECUTION SUMMARY" in line:
            return C_SUMMARY
        return SECTION_COLORS.get(section, C_DEFAULT)

    # ── Determine section per line ────────────────────────────────────────────
    raw_lines = "\n".join(trace_lines).splitlines()
    section   = "default"
    colored: list[tuple[str, tuple]] = []
    for line in raw_lines:
        if "THOUGHT" in line:   section = "thought"
        elif "ACTION" in line:  section = "action"
        elif "OBSERVATION" in line: section = "observe"
        elif "FINAL ANSWER" in line: section = "answer"
        elif "EXECUTION SUMMARY" in line: section = "summary"
        elif line.strip().startswith("╚"): section = "default"
        colored.append((line, _color(line, section)))

    # ── Measure dimensions ────────────────────────────────────────────────────
    PAD         = 24
    LINE_H      = font_size + 5
    dummy_img   = Image.new("RGB", (1, 1))
    dummy_draw  = ImageDraw.Draw(dummy_img)
    max_w       = max(
        (int(dummy_draw.textlength(ln, font=font)) for ln, _ in colored if ln.strip()),
        default=400,
    )
    img_w = max_w + PAD * 2
    img_h = len(colored) * LINE_H + PAD * 2

    # ── Draw ──────────────────────────────────────────────────────────────────
    img  = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)
    for i, (line, color) in enumerate(colored):
        draw.text((PAD, PAD + i * LINE_H), line, font=font, fill=color)

    img.save(path, "PNG")
    return True


# ── ReAct Agent ────────────────────────────────────────────────────────────────

class ReActAgent:
    """ReAct agent that interleaves Claude's reasoning with tool calls until a final answer is reached."""

    def __init__(self, model: str = MODEL, max_steps: int = MAX_STEPS):
        self.client    = anthropic.Anthropic(api_key=_api_key)
        self.model     = model
        self.max_steps = max_steps

    # ── Retry wrapper ──────────────────────────────────────────────────────────
    def _call_api(self, messages: list) -> anthropic.types.Message:
        """Call the Claude API with exponential backoff on rate-limit and server errors."""
        delay = 1
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    thinking={"type": "enabled", "budget_tokens": 8000},
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            except anthropic.RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"  [Rate limited — retrying in {delay}s ({attempt + 1}/{MAX_RETRIES})]")
                time.sleep(delay)
                delay *= 2
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500 and attempt < MAX_RETRIES - 1:
                    print(f"  [Server error {exc.status_code} — retrying in {delay}s]")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self, query: str, save_trace: bool = True,
            _prior_messages: list | None = None) -> str:
        """Run the Thought→Action→Observation loop for a query; returns the final answer string."""
        trace_lines: list[str] = []
        tool_stats: dict[str, int] = defaultdict(int)
        start_time = time.time()

        header = f"\n{'═' * _W}\nQUERY: {query}\n{'═' * _W}"
        _emit(header, trace_lines)

        messages = list(_prior_messages) if _prior_messages else []
        messages.append({"role": "user", "content": query})
        step          = 0
        final_answer  = ""

        while step < self.max_steps:
            step += 1

            response = self._call_api(messages)

            assistant_blocks = []
            thought_text     = ""
            text_output      = ""
            tool_calls       = []

            for block in response.content:
                assistant_blocks.append(block)
                if block.type == "thinking":
                    thought_text += block.thinking
                elif block.type == "text":
                    text_output += block.text
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # Show THOUGHT block from extended thinking only (not pre-tool narrative text)
            if thought_text:
                full_thought = thought_text.strip()
                preview = full_thought if len(full_thought) <= 350 else full_thought[:350] + f"\n... [{len(full_thought)-350} chars — full in trace]"
                _emit(
                    _box(f"[Step {step}]  THOUGHT", full_thought),
                    trace_lines,
                    display=_box(f"[Step {step}]  THOUGHT", preview),
                )

            # No tool calls → final answer
            if not tool_calls:
                final_answer = text_output.strip()
                _emit(
                    f"\n{'═' * _W}\nFINAL ANSWER\n{_DIVIDER}\n{final_answer}\n{'═' * _W}",
                    trace_lines,
                )
                break

            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_results = []
            for tc in tool_calls:
                tool_stats[tc.name] += 1

                _emit(
                    _box(
                        f"[Step {step}]  ACTION",
                        f"Tool  : {tc.name}\nInput : {json.dumps(tc.input, indent=2)}",
                    ),
                    trace_lines,
                )

                result = execute_tool(tc.name, tc.input)

                _emit(_box(f"[Step {step}]  OBSERVATION", result), trace_lines)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            final_answer = "[Max steps reached without a final answer]"
            _emit(f"\n[WARNING] {final_answer}", trace_lines)

        elapsed = time.time() - start_time
        tools_summary = (
            ", ".join(f"{k}×{v}" for k, v in sorted(tool_stats.items()))
            or "none"
        )
        status = "✓ Success" if final_answer and not final_answer.startswith("[") else "✗ Incomplete"

        _emit(_summary_box(step, tools_summary, elapsed, status), trace_lines)

        # ── Save PNG ───────────────────────────────────────────────────────────
        if save_trace:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            slug = "".join(c if c.isalnum() else "_" for c in query[:40]).strip("_")
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

            png_path = TRACE_DIR / f"{ts}_{slug}.png"
            png_ok   = _save_png(trace_lines, png_path)
            png_note = png_path.name if png_ok else "(PNG skipped — install Pillow)"
            print(f"\n[Saved → {png_note}]")

        return final_answer

    # ── Conversational loop ────────────────────────────────────────────────────
    def chat(self) -> None:
        """Start an interactive multi-turn conversation, maintaining history across turns."""
        global _CHAT_MODE
        _CHAT_MODE = True

        history: list[dict] = []

        print("\n" + "═" * _W)
        print("  ReAct Agent  —  Conversational Mode")
        print("  Type 'exit' or 'quit' to end the session.")
        print("═" * _W + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
                print("\nGoodbye!")
                break

            print()
            answer = self.run(user_input, save_trace=False, _prior_messages=history)

            # Keep only plain-text turns in history to avoid bloating context
            # with tool_use / tool_result / thinking blocks from prior steps.
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})

            print(f"\nAgent: {answer}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the agent in chat, single-query, or demo mode."""
    global _VERBOSE, _QUIET

    parser = argparse.ArgumentParser(
        description="ReAct Agent — Claude claude-sonnet-4-6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/react_agent.py                    # interactive chat (default)\n"
            "  python src/react_agent.py -q 'What is 15*7?' # single query\n"
            "  python src/react_agent.py --demo             # run all demo queries\n"
            "  python src/react_agent.py --verbose          # show thinking blocks in chat\n"
        ),
    )
    parser.add_argument("--query", "-q", default=None,
                        help="Run a single query and exit.")
    parser.add_argument("--demo", action="store_true",
                        help="Run all built-in demo queries and exit.")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving trace files to screenshots/.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show extended thinking (Thought) blocks.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only final answers (single-query / demo modes).")
    args = parser.parse_args()

    _VERBOSE = args.verbose
    _QUIET   = args.quiet
    save     = not args.no_save

    agent = ReActAgent()

    if args.query:
        agent.run(args.query, save_trace=save)
    elif args.demo:
        print(f"Running {len(DEMO_QUERIES)} demo queries...\n")
        for i, q in enumerate(DEMO_QUERIES, 1):
            print(f"\n{'─' * _W}\n[Demo {i}/{len(DEMO_QUERIES)}]")
            agent.run(q, save_trace=save)
        print(f"\n{'═' * _W}")
        print("All demo queries complete. Check screenshots/ for traces.")
    else:
        agent.chat()


if __name__ == "__main__":
    main()
