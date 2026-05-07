"""Tool definitions and implementations for the ReAct agent."""

import math
import datetime
import subprocess
import sys
from urllib.parse import quote

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ── Knowledge base ─────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "python": "Python is a high-level, interpreted programming language prized for its readability and versatility. It is widely used in data science, web development, automation, and AI.",
    "anthropic": "Anthropic is an AI safety company founded in 2021 that created the Claude family of large language models. Its mission is to build AI systems that are safe, beneficial, and understandable.",
    "claude": "Claude is a family of large language models developed by Anthropic. Claude models are designed to be helpful, harmless, and honest, and support advanced features like tool use and extended thinking.",
    "react": "ReAct (Reasoning + Acting) is an AI agent paradigm that interleaves step-by-step reasoning (Thought) with tool invocations (Action) and their results (Observation) before producing a final answer.",
    "llm": "A Large Language Model (LLM) is a deep-learning model trained on massive text corpora that can generate, summarise, translate, and reason about natural language. Examples include Claude, GPT-4, and Gemini.",
    "agent": "An AI agent is a system that perceives its environment, reasons about goals, and takes actions — often via tools — to accomplish tasks autonomously, iterating until the goal is reached.",
    "tool use": "Tool use (function calling) lets an LLM invoke external functions or APIs during generation, grounding its responses in real data, computation, or live information.",
    "agentic ai": "Agentic AI refers to AI systems that operate autonomously over multiple steps, using tools and memory to complete complex, open-ended tasks without constant human intervention.",
    "transformer": "The Transformer is a neural network architecture introduced in 2017 that uses self-attention mechanisms to process sequences in parallel. It is the foundation of nearly all modern LLMs.",
    "prompt engineering": "Prompt engineering is the practice of carefully crafting inputs to language models to elicit the desired outputs. Techniques include chain-of-thought, few-shot examples, and role assignment.",
    "rag": "Retrieval-Augmented Generation (RAG) combines an LLM with a retrieval system. The model fetches relevant documents before generating an answer, improving factual accuracy and reducing hallucinations.",
    "fine-tuning": "Fine-tuning adapts a pre-trained model to a specific task or domain by continuing training on a smaller, curated dataset. It is cheaper than training from scratch and preserves general capabilities.",
}

# ── Tool schemas (sent to Claude) ──────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": (
            "Evaluates a mathematical expression and returns the numeric result. "
            "Supports all Python arithmetic operators (+, -, *, /, //, **, %) and "
            "every function in Python's math module (sqrt, log, sin, cos, ceil, floor, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid math expression, e.g. '2**10' or 'sqrt(256) + log(100, 10)'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Returns the current local date and time as a human-readable string.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "lookup",
        "description": (
            "Looks up an AI/tech concept in the knowledge base. "
            "Supported terms: python, anthropic, claude, react, llm, agent, tool use, "
            "agentic ai, transformer, prompt engineering, rag, fine-tuning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "term": {"type": "string", "description": "Concept or term to look up."}
            },
            "required": ["term"],
        },
    },
    {
        "name": "unit_converter",
        "description": (
            "Converts a value between units. "
            "Temperature: celsius/fahrenheit/kelvin. "
            "Distance: km/miles/meters/feet. "
            "Weight: kg/pounds/grams/ounces."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value":     {"type": "number", "description": "Numeric value to convert."},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'celsius'."},
                "to_unit":   {"type": "string", "description": "Target unit, e.g. 'fahrenheit'."},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Searches the web via DuckDuckGo Instant Answers for real-time facts, definitions, "
            "or current information not available in the knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Returns current weather conditions for any city or location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name, e.g. 'London' or 'New York'.",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Executes a Python code snippet in a subprocess and returns its stdout output. "
            "Use for complex algorithms, list/string manipulation, data processing, or anything "
            "beyond simple arithmetic. Always use print() to output results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Must use print() to produce output.",
                }
            },
            "required": ["code"],
        },
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    """Evaluate a math expression using Python's math module; returns the result as a string."""
    safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns["__builtins__"] = {}
    try:
        result = eval(expression, safe_ns)  # noqa: S307
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


def get_current_time() -> str:
    """Return the current local date and time as a formatted human-readable string."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")


def lookup(term: str) -> str:
    """Look up an AI/tech concept in the knowledge base; returns the definition or a not-found message."""
    key = term.strip().lower()
    if key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[key]
    for kb_key, kb_val in KNOWLEDGE_BASE.items():
        if key in kb_key or kb_key in key:
            return kb_val
    available = ", ".join(KNOWLEDGE_BASE.keys())
    return f"Term '{term}' not found in knowledge base. Available: {available}."


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between temperature, distance, or weight units; returns the result as a string."""
    f, t = from_unit.strip().lower(), to_unit.strip().lower()

    def _to_c(v, u):
        if u == "celsius":    return v
        if u == "fahrenheit": return (v - 32) * 5 / 9
        if u == "kelvin":     return v - 273.15
        return None

    def _from_c(v, u):
        if u == "celsius":    return v
        if u == "fahrenheit": return v * 9 / 5 + 32
        if u == "kelvin":     return v + 273.15
        return None

    if f in {"celsius", "fahrenheit", "kelvin"} and t in {"celsius", "fahrenheit", "kelvin"}:
        c = _to_c(value, f)
        if c is None: return f"Unknown temperature unit: {from_unit}"
        r = _from_c(c, t)
        if r is None: return f"Unknown temperature unit: {to_unit}"
        return f"{value} {from_unit} = {r:.4f} {to_unit}"

    dist = {"meters": 1, "metres": 1, "m": 1, "km": 1000, "kilometers": 1000,
            "miles": 1609.344, "feet": 0.3048, "foot": 0.3048, "ft": 0.3048}
    if f in dist and t in dist:
        return f"{value} {from_unit} = {value * dist[f] / dist[t]:.4f} {to_unit}"

    weight = {"grams": 1, "gram": 1, "g": 1, "kg": 1000, "kilograms": 1000,
              "pounds": 453.592, "pound": 453.592, "lb": 453.592, "lbs": 453.592,
              "ounces": 28.3495, "ounce": 28.3495, "oz": 28.3495}
    if f in weight and t in weight:
        return f"{value} {from_unit} = {value * weight[f] / weight[t]:.4f} {to_unit}"

    return (
        f"Unsupported conversion: '{from_unit}' → '{to_unit}'. "
        "Supported: temperature (celsius/fahrenheit/kelvin), "
        "distance (km/miles/meters/feet), weight (kg/pounds/grams/ounces)."
    )


def web_search(query: str) -> str:
    """Search DuckDuckGo Instant Answers for real-time facts; returns a summary with source attribution."""
    if not _REQUESTS_OK:
        return "Error: 'requests' not installed. Run: pip install requests"
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1",
                    "skip_disambig": "1", "no_redirect": "1"},
            headers={"User-Agent": "ReActAgent/2.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        parts = []
        if data.get("AbstractText"):
            src = data.get("AbstractSource", "")
            parts.append(f"{data['AbstractText']}\n(Source: {src})")
        if data.get("Answer"):
            parts.append(f"Quick answer: {data['Answer']}")
        if data.get("Definition"):
            parts.append(f"Definition: {data['Definition']}")

        topics = [
            f"• {t['Text']}"
            for t in data.get("RelatedTopics", [])[:4]
            if isinstance(t, dict) and t.get("Text")
        ]
        if topics:
            parts.append("Related:\n" + "\n".join(topics))

        if not parts:
            return (
                f"No instant answer found for '{query}'.\n"
                "The DuckDuckGo Instant Answers API works best for well-known factual topics. "
                "Try rephrasing or use the lookup tool for AI/tech concepts."
            )
        return "\n\n".join(parts)
    except requests.Timeout:
        return "Error: Web search timed out (10s)."
    except Exception as exc:
        return f"Web search error: {exc}"


def get_weather(location: str) -> str:
    """Fetch current weather for a city from wttr.in; returns condition, temperature, humidity, and wind."""
    if not _REQUESTS_OK:
        return "Error: 'requests' not installed. Run: pip install requests"
    try:
        resp = requests.get(
            f"https://wttr.in/{quote(location)}?format=j1",
            headers={"User-Agent": "ReActAgent/2.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        cur  = data["current_condition"][0]
        area = data["nearest_area"][0]
        city    = area["areaName"][0]["value"]
        country = area["country"][0]["value"]

        return (
            f"Weather for {city}, {country}:\n"
            f"  Condition   : {cur['weatherDesc'][0]['value']}\n"
            f"  Temperature : {cur['temp_C']}°C / {cur['temp_F']}°F "
            f"(feels like {cur['FeelsLikeC']}°C)\n"
            f"  Humidity    : {cur['humidity']}%\n"
            f"  Wind        : {cur['windspeedKmph']} km/h\n"
            f"  Visibility  : {cur.get('visibility', 'N/A')} km"
        )
    except requests.Timeout:
        return f"Error: Weather request timed out for '{location}'."
    except Exception as exc:
        return f"Error fetching weather for '{location}': {exc}"


def run_python(code: str) -> str:
    """Execute a Python snippet in a subprocess (10s timeout); returns stdout or a structured error message."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            return f"RuntimeError (exit {result.returncode}):\n{stderr}"
        if stderr:
            return f"{stdout}\n[stderr]: {stderr}" if stdout else f"[stderr]: {stderr}"
        return stdout or "(ran successfully — no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (10s limit)."
    except Exception as exc:
        return f"Error running code: {exc}"


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call by name and return its string result, or a structured error on failure."""
    dispatch = {
        "calculator":       lambda i: calculator(i["expression"]),
        "get_current_time": lambda _: get_current_time(),
        "lookup":           lambda i: lookup(i["term"]),
        "unit_converter":   lambda i: unit_converter(i["value"], i["from_unit"], i["to_unit"]),
        "web_search":       lambda i: web_search(i["query"]),
        "get_weather":      lambda i: get_weather(i["location"]),
        "run_python":       lambda i: run_python(i["code"]),
    }
    handler = dispatch.get(name)
    if handler is None:
        return f"Unknown tool '{name}'. Available: {', '.join(dispatch)}"
    try:
        return handler(inputs)
    except KeyError as exc:
        return f"Missing required parameter for '{name}': {exc}"
    except Exception as exc:
        return f"Tool '{name}' error: {exc}"
