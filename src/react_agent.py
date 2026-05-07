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

# Fix 1: use python-dotenv instead of manual parser
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
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

_W       = 62
_DIVIDER = "─" * _W

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


# ── PNG renderer (Fix 3: split into focused helpers) ───────────────────────────

def _make_font(size: int = 14):
    """Load a monospace font from common OS paths, falling back to PIL default."""
    from PIL import ImageFont
    for fp in [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _colorize_lines(raw_text: str) -> list[tuple[str, tuple]]:
    """Assign a VS Code Dark+ color to each line based on its ReAct section."""
    C_DEFAULT = (212, 212, 212)
    C_BORDER  = ( 80,  80,  90)
    C_QUERY   = ( 86, 156, 214)
    C_THOUGHT = (156, 220, 254)
    C_ACTION  = (206, 145, 120)
    C_OBSERVE = (106, 153,  85)
    C_ANSWER  = ( 78, 201, 176)
    C_SUMMARY = (220, 220, 100)

    SECTION_COLORS = {
        "thought": C_THOUGHT, "action": C_ACTION, "observe": C_OBSERVE,
        "answer":  C_ANSWER,  "summary": C_SUMMARY,
    }

    def _color(line: str, section: str) -> tuple:
        s = line.strip()
        if not s:                                              return C_DEFAULT
        if all(c in "═─╔╗╠╣╚╝║ " for c in s):               return C_BORDER
        if "QUERY:" in line:                                   return C_QUERY
        if "THOUGHT" in line and s.startswith("[Step"):        return C_THOUGHT
        if "ACTION" in line and s.startswith("[Step"):         return C_ACTION
        if "OBSERVATION" in line and s.startswith("[Step"):    return C_OBSERVE
        if "FINAL ANSWER" in line:                             return C_ANSWER
        if "EXECUTION SUMMARY" in line:                        return C_SUMMARY
        return SECTION_COLORS.get(section, C_DEFAULT)

    section = "default"
    colored: list[tuple[str, tuple]] = []
    for line in raw_text.splitlines():
        if "THOUGHT" in line:            section = "thought"
        elif "ACTION" in line:           section = "action"
        elif "OBSERVATION" in line:      section = "observe"
        elif "FINAL ANSWER" in line:     section = "answer"
        elif "EXECUTION SUMMARY" in line: section = "summary"
        elif line.strip().startswith("╚"): section = "default"
        colored.append((line, _color(line, section)))
    return colored


def _save_png(trace_lines: list[str], path: Path) -> bool:
    """Render the trace as a dark-themed terminal PNG. Returns True on success."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    font    = _make_font()
    colored = _colorize_lines("\n".join(trace_lines))

    PAD, LINE_H, BG = 24, 19, (30, 30, 30)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_w = max((int(dummy.textlength(ln, font=font)) for ln, _ in colored if ln.strip()), default=400)

    img  = Image.new("RGB", (max_w + PAD * 2, len(colored) * LINE_H + PAD * 2), BG)
    draw = ImageDraw.Draw(img)
    for i, (line, color) in enumerate(colored):
        draw.text((PAD, PAD + i * LINE_H), line, font=font, fill=color)

    img.save(path, "PNG")
    return True


# ── ReAct Agent ────────────────────────────────────────────────────────────────

class ReActAgent:
    """ReAct agent that interleaves reasoning with tool calls until a final answer is reached."""

    # Fix 2: verbose/quiet as constructor params; no mutable module globals
    def __init__(self, model: str = MODEL, max_steps: int = MAX_STEPS,
                 verbose: bool = False, quiet: bool = False):
        self.client     = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model      = model
        self.max_steps  = max_steps
        self.verbose    = verbose
        self.quiet      = quiet
        self._chat_mode = False

    def _emit(self, text: str, lines: list, display: str | None = None) -> None:
        """Append to trace and print according to current display mode."""
        lines.append(text)
        shown = display if display is not None else text

        if self._chat_mode:
            if self.verbose:
                if "FINAL ANSWER" in text or text.strip().startswith("╔"):
                    return
                if text.lstrip("\n").startswith("═") and "QUERY:" in text:
                    return
                print(shown)
            else:
                if "ACTION" in text and "Tool  :" in text:
                    for line in text.splitlines():
                        if line.strip().startswith("Tool"):
                            print(f"  [Using {line.split(':', 1)[-1].strip()}...]")
                            break
            return

        if self.quiet:
            if "FINAL ANSWER" in text or text.strip().startswith("╔"):
                print(shown)
            return
        print(shown)

    def _call_api(self, messages: list) -> anthropic.types.Message:
        """Call the API with exponential backoff on rate-limit and server errors."""
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

    def run(self, query: str, save_trace: bool = True,
            _prior_messages: list | None = None) -> str:
        """Run the Thought→Action→Observation loop; returns the final answer string."""
        trace_lines: list[str] = []
        tool_stats: dict[str, int] = defaultdict(int)
        start_time = time.time()

        self._emit(f"\n{'═' * _W}\nQUERY: {query}\n{'═' * _W}", trace_lines)

        messages = list(_prior_messages) if _prior_messages else []
        messages.append({"role": "user", "content": query})
        step, final_answer = 0, ""

        while step < self.max_steps:
            step += 1
            response = self._call_api(messages)

            thought_text, text_output, tool_calls, assistant_blocks = "", "", [], []
            for block in response.content:
                assistant_blocks.append(block)
                if block.type == "thinking":   thought_text += block.thinking
                elif block.type == "text":     text_output  += block.text
                elif block.type == "tool_use": tool_calls.append(block)

            if thought_text:
                full = thought_text.strip()
                preview = full if len(full) <= 350 else full[:350] + f"\n... [{len(full)-350} chars — full in trace]"
                self._emit(_box(f"[Step {step}]  THOUGHT", full), trace_lines,
                           display=_box(f"[Step {step}]  THOUGHT", preview))

            if not tool_calls:
                final_answer = text_output.strip()
                self._emit(f"\n{'═'*_W}\nFINAL ANSWER\n{_DIVIDER}\n{final_answer}\n{'═'*_W}", trace_lines)
                break

            messages.append({"role": "assistant", "content": assistant_blocks})
            tool_results = []
            for tc in tool_calls:
                tool_stats[tc.name] += 1
                self._emit(_box(f"[Step {step}]  ACTION",
                                f"Tool  : {tc.name}\nInput : {json.dumps(tc.input, indent=2)}"), trace_lines)
                result = execute_tool(tc.name, tc.input)
                self._emit(_box(f"[Step {step}]  OBSERVATION", result), trace_lines)
                tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
            messages.append({"role": "user", "content": tool_results})

        else:
            final_answer = "[Max steps reached without a final answer]"
            self._emit(f"\n[WARNING] {final_answer}", trace_lines)

        elapsed = time.time() - start_time
        tools_summary = ", ".join(f"{k}×{v}" for k, v in sorted(tool_stats.items())) or "none"
        status = "✓ Success" if final_answer and not final_answer.startswith("[") else "✗ Incomplete"
        self._emit(_summary_box(step, tools_summary, elapsed, status), trace_lines)

        if save_trace:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            slug     = "".join(c if c.isalnum() else "_" for c in query[:40]).strip("_")
            png_path = TRACE_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.png"
            png_ok   = _save_png(trace_lines, png_path)
            print(f"\n[Saved → {png_path.name if png_ok else '(PNG skipped — install Pillow)'}]")

        return final_answer

    def chat(self) -> None:
        """Start an interactive multi-turn conversation, maintaining history across turns."""
        self._chat_mode = True
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
            # Store only plain-text turns; tool_use/tool_result/thinking blocks stay out of history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
            print(f"\nAgent: {answer}\n")

        self._chat_mode = False


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the agent in chat, single-query, or demo mode."""
    parser = argparse.ArgumentParser(
        description="ReAct Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/react_agent.py                    # interactive chat (default)\n"
            "  python src/react_agent.py -q 'What is 15*7?' # single query\n"
            "  python src/react_agent.py --demo             # run all demo queries\n"
            "  python src/react_agent.py --verbose          # show thinking blocks in chat\n"
        ),
    )
    parser.add_argument("--query",   "-q", default=None,      help="Run a single query and exit.")
    parser.add_argument("--demo",          action="store_true", help="Run all built-in demo queries.")
    parser.add_argument("--no-save",       action="store_true", help="Skip saving trace files.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extended thinking blocks.")
    parser.add_argument("--quiet",         action="store_true", help="Print only final answers.")
    args = parser.parse_args()

    agent = ReActAgent(verbose=args.verbose, quiet=args.quiet)
    save  = not args.no_save

    if args.query:
        agent.run(args.query, save_trace=save)
    elif args.demo:
        print(f"Running {len(DEMO_QUERIES)} demo queries...\n")
        for i, q in enumerate(DEMO_QUERIES, 1):
            print(f"\n{'─' * _W}\n[Demo {i}/{len(DEMO_QUERIES)}]")
            agent.run(q, save_trace=save)
        print(f"\n{'═' * _W}\nAll demo queries complete. Check screenshots/ for traces.")
    else:
        agent.chat()


if __name__ == "__main__":
    main()
