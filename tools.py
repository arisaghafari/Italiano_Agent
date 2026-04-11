import json
import os
import base64
import sqlite3
from datetime import datetime, timezone
from langchain_core.tools import tool

DB_FILE = "knowledge.db"

# ── DATABASE SETUP ──────────────────────────────────────────────
def get_connection():
    """Returns a SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn

def init_db():
    """Creates tables if they don't exist. Called once at startup."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS vocabulary (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            word        TEXT    NOT NULL UNIQUE,
            meaning     TEXT    NOT NULL,
            example     TEXT,
            learned_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS grammar (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            rule        TEXT    NOT NULL UNIQUE,
            explanation TEXT    NOT NULL,
            example     TEXT,
            learned_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS expressions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            expression  TEXT    NOT NULL UNIQUE,
            meaning     TEXT    NOT NULL,
            example     TEXT,
            learned_at  TEXT    NOT NULL
        );
    """)

    conn.commit()
    conn.close()

def get_timestamp() -> str:
    """Returns current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()

# ── TOOLS ───────────────────────────────────────────────────────

@tool
def extract_knowledge(image_path: str) -> str:
    """Extracts Italian vocabulary, grammar rules, and expressions from
    a photo of a book page. Use this when the user provides an image path.
    Args:
        image_path: The full path to the image file on disk.
    """
    if not os.path.exists(image_path):
        return f"Error: file not found at {image_path}"

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    return json.dumps({
        "image_data": image_data,
        "instruction": (
            "Look at this Italian textbook page carefully. "
            "Extract and return a JSON object with exactly these 3 keys:\n"
            "- 'vocabulary': list of {word, meaning, example}\n"
            "- 'grammar': list of {rule, explanation, example}\n"
            "- 'expressions': list of {expression, meaning, example}\n"
            "Only include items clearly present on the page. "
            "Return ONLY the JSON object, no extra text."
        )
    })


@tool
def save_knowledge(extracted_json: str) -> str:
    """Saves newly extracted Italian vocabulary, grammar rules, and expressions
    to the database. Automatically skips duplicates using UNIQUE constraints.
    Args:
        extracted_json: A JSON string with keys: vocabulary, grammar, expressions.
    """
    try:
        new_items = json.loads(extracted_json)
    except json.JSONDecodeError:
        return "Error: invalid JSON format. Expected keys: vocabulary, grammar, expressions."

    conn = get_connection()
    cursor = conn.cursor()
    now = get_timestamp()
    added = {"vocabulary": 0, "grammar": 0, "expressions": 0}

    # Vocabulary
    for item in new_items.get("vocabulary", []):
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO vocabulary (word, meaning, example, learned_at) "
                "VALUES (?, ?, ?, ?)",
                (item.get("word"), item.get("meaning"), item.get("example"), now)
            )
            if cursor.rowcount > 0:
                added["vocabulary"] += 1
        except sqlite3.Error:
            continue

    # Grammar
    for item in new_items.get("grammar", []):
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO grammar (rule, explanation, example, learned_at) "
                "VALUES (?, ?, ?, ?)",
                (item.get("rule"), item.get("explanation"), item.get("example"), now)
            )
            if cursor.rowcount > 0:
                added["grammar"] += 1
        except sqlite3.Error:
            continue

    # Expressions
    for item in new_items.get("expressions", []):
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO expressions (expression, meaning, example, learned_at) "
                "VALUES (?, ?, ?, ?)",
                (item.get("expression"), item.get("meaning"), item.get("example"), now)
            )
            if cursor.rowcount > 0:
                added["expressions"] += 1
        except sqlite3.Error:
            continue

    conn.commit()
    conn.close()

    return (
        f"Saved to database! Added: "
        f"{added['vocabulary']} new words, "
        f"{added['grammar']} grammar rules, "
        f"{added['expressions']} expressions. "
        f"(Duplicates were automatically skipped.)"
    )


@tool
def load_knowledge() -> str:
    """Loads all Italian knowledge learned so far from the database,
    separating items into 'new' (last 24 hours) and 'old' categories.
    Use this before generating a story.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cutoff = "datetime('now', '-24 hours')"

    # New items (last 24h)
    cursor.execute(f"SELECT word, meaning, example FROM vocabulary WHERE learned_at >= {cutoff}")
    new_vocab = [dict(row) for row in cursor.fetchall()]

    cursor.execute(f"SELECT word, meaning, example FROM vocabulary WHERE learned_at < {cutoff}")
    old_vocab = [dict(row) for row in cursor.fetchall()]

    cursor.execute(f"SELECT rule, explanation, example FROM grammar WHERE learned_at >= {cutoff}")
    new_grammar = [dict(row) for row in cursor.fetchall()]

    cursor.execute(f"SELECT rule, explanation, example FROM grammar WHERE learned_at < {cutoff}")
    old_grammar = [dict(row) for row in cursor.fetchall()]

    cursor.execute(f"SELECT expression, meaning, example FROM expressions WHERE learned_at >= {cutoff}")
    new_exprs = [dict(row) for row in cursor.fetchall()]

    cursor.execute(f"SELECT expression, meaning, example FROM expressions WHERE learned_at < {cutoff}")
    old_exprs = [dict(row) for row in cursor.fetchall()]

    conn.close()

    total = len(new_vocab) + len(old_vocab) + len(new_grammar) + len(old_grammar) + len(new_exprs) + len(old_exprs)
    if total == 0:
        return "No knowledge saved yet. Process a book page first."

    return json.dumps({
        "new": {
            "vocabulary":   new_vocab,
            "grammar":      new_grammar,
            "expressions":  new_exprs
        },
        "old": {
            "vocabulary":   old_vocab,
            "grammar":      old_grammar,
            "expressions":  old_exprs
        }
    }, ensure_ascii=False, indent=2)


@tool
def generate_story(knowledge_json: str) -> str:
    """Generates a short Italian story using all vocabulary, grammar rules,
    and expressions the user has learned. Focuses heavily on items learned
    in the last 24 hours while naturally weaving in older knowledge.
    Args:
        knowledge_json: JSON string from load_knowledge with 'new' and 'old' sections.
    """
    try:
        data = json.loads(knowledge_json)
    except json.JSONDecodeError:
        return "Error: invalid knowledge format."

    new = data.get("new", {})
    old = data.get("old", {})

    new_words = [v["word"] for v in new.get("vocabulary", [])]
    new_rules = [g["rule"] for g in new.get("grammar", [])]
    new_exprs = [e["expression"] for e in new.get("expressions", [])]

    old_words = [v["word"] for v in old.get("vocabulary", [])]
    old_rules = [g["rule"] for g in old.get("grammar", [])]
    old_exprs = [e["expression"] for e in old.get("expressions", [])]

    return json.dumps({
        "instruction": (
            "Write a short, engaging story in Italian (10-15 sentences).\n\n"
            f"TODAY'S NEW items — use these as the main focus:\n"
            f"  Vocabulary:   {new_words}\n"
            f"  Grammar:      {new_rules}\n"
            f"  Expressions:  {new_exprs}\n\n"
            f"OLDER items — weave these in naturally:\n"
            f"  Vocabulary:   {old_words}\n"
            f"  Grammar:      {old_rules}\n"
            f"  Expressions:  {old_exprs}\n\n"
            "Formatting rules:\n"
            "1. Match difficulty to the vocabulary/grammar level above.\n"
            "2. Bold every new word using **word** in the Italian text.\n"
            "3. After the story, add an English translation.\n"
            "4. End with: 'Parole nuove usate: ...' listing the new words used."
        )
    })


# Initialize DB when tools.py is imported
init_db()