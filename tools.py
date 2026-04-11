import json
import os
import base64
import sqlite3
from datetime import datetime, timezone
from langchain_core.tools import tool
from PIL import Image
import io
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage as LCHumanMessage
from dotenv import load_dotenv

load_dotenv()

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
    a photo of a book page using a vision model.
    Args:
        image_path: The full path to the image file on disk.
    """
    if not os.path.exists(image_path):
        return f"Error: file not found at {image_path}"

    # Resize image
    img = Image.open(image_path).convert("RGB")
    max_size = 800
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    image_data = base64.standard_b64encode(buffer.read()).decode("utf-8")

    # Call vision model directly here — result is just text, not the image
    vision_llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )

    vision_response = vision_llm.invoke([
        LCHumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            },
            {
                "type": "text",
                "text": (
                    "Look at this Italian textbook page carefully. "
                    "Extract and return ONLY a valid JSON object with exactly these 3 keys:\n\n"
                    "{\n"
                    '  "vocabulary": [{"word": "...", "meaning": "English meaning", "example": "Italian sentence using the word"}],\n'
                    '  "grammar": [{"rule": "rule name", "explanation": "brief explanation in English", "example": "Italian example sentence"}],\n'
                    "}\n\n"
                    "IMPORTANT:\n"
                    "- Every item MUST have a non-empty meaning AND example.\n"
                    "- If you cannot find an example on the page, create a natural one yourself.\n"
                    "- Return ONLY the JSON, no extra text, no markdown code blocks."
                )
            }
        ])
    ])

    # Return only the TEXT result — no image data in the agent history!
    return vision_response.content


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
    added = {"vocabulary": 0, "grammar": 0}

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

    conn.commit()
    conn.close()

    return (
        f"Saved to database! Added: "
        f"{added['vocabulary']} new words, "
        f"{added['grammar']} grammar rules, "
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

    conn.close()

    total = len(new_vocab) + len(old_vocab) + len(new_grammar) + len(old_grammar)
    if total == 0:
        return "No knowledge saved yet. Process a book page first."

    return json.dumps({
        "new": {
            "vocabulary":   new_vocab,
            "grammar":      new_grammar
        },
        "old": {
            "vocabulary":   old_vocab,
            "grammar":      old_grammar
        }
    }, ensure_ascii=False, indent=2)


@tool
def generate_story(dummy: str = "") -> str:
    """Generates a short Italian story using all vocabulary and grammar rules
    the user has learned so far. Focuses heavily on items learned in the last
    24 hours. Call this with no arguments when the user asks for a story.
    """
    # Load directly from DB — no LLM truncation risk
    conn = get_connection()
    cursor = conn.cursor()
    cutoff = "datetime('now', '-24 hours')"

    cursor.execute(f"SELECT word FROM vocabulary WHERE learned_at >= {cutoff}")
    new_words = [r[0] for r in cursor.fetchall()]

    cursor.execute(f"SELECT word FROM vocabulary WHERE learned_at < {cutoff}")
    old_words = [r[0] for r in cursor.fetchall()]

    cursor.execute(f"SELECT rule FROM grammar WHERE learned_at >= {cutoff}")
    new_rules = [r[0] for r in cursor.fetchall()]

    cursor.execute(f"SELECT rule FROM grammar WHERE learned_at < {cutoff}")
    old_rules = [r[0] for r in cursor.fetchall()]

    conn.close()

    if not new_words and not old_words:
        return "No knowledge saved yet. Process a book page first."

    return (
        "Write a short, engaging story in Italian (10-15 sentences).\n\n"
        f"TODAY'S NEW items — use these as the main focus:\n"
        f"  Vocabulary: {new_words}\n"
        f"  Grammar:    {new_rules}\n\n"
        f"OLDER items — weave these in naturally:\n"
        f"  Vocabulary: {old_words}\n"
        f"  Grammar:    {old_rules}\n\n"
        "Formatting rules:\n"
        "1. Match difficulty to the vocabulary/grammar level above.\n"
        "2. Bold every new word using **word** in the Italian text.\n"
        "3. After the story, add an English translation.\n"
        "4. End with: 'Parole nuove usate: ...' listing the new words used."
    )

# Initialize DB when tools.py is imported
init_db()