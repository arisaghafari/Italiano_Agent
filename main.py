import os
import json
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import render_text_description
from tools import extract_knowledge, save_knowledge, load_knowledge, generate_story, init_db

load_dotenv()
init_db()

# ── TOOLS LIST ───────────────────────────────────────────────────
TOOLS = [extract_knowledge, save_knowledge, load_knowledge, generate_story]
TOOL_MAP = {t.name: t for t in TOOLS}

# ── SYSTEM PROMPT ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are 'Agente Italiano', an intelligent Italian language learning assistant.

Your job is to help the user learn Italian from their textbook photos by:
1. Extracting vocabulary, grammar rules, and expressions from book page images
2. Saving them to a database (skipping duplicates automatically)
3. Generating personalized Italian stories that reinforce what they've learned

You have access to these tools:
{tool_descriptions}

WORKFLOW — follow these rules strictly:

When the user gives you an IMAGE PATH:
  Step 1: Call extract_knowledge(image_path)
  Step 2: Call save_knowledge(extracted_json)
  Step 3: Confirm to the user: "✅ Page processed! X new words and Y grammar rules saved."
  STOP HERE. Do NOT load knowledge or generate a story automatically.

When the user says 'storia' or asks for a story:
  Step 1: Call generate_story(knowledge_json)
  Step 2: Write the story using the instruction from generate_story

NEVER generate a story automatically after processing an image.
"""

# ── AGENT CLASS ──────────────────────────────────────────────────
class ItalianoAgent:
    def __init__(self):
        # Vision model — for image reading + extraction (low temp = precise)
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Larger model — for story generation (medium temp = creative)
        self.llm_story = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.65,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.llm_extraction = self.llm.bind_tools(TOOLS)
        self.llm_story_bound = self.llm_story.bind_tools(TOOLS)

        self.history = []
        tool_desc = render_text_description(TOOLS)
        self.system_msg = SystemMessage(
            content=SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
        )

    def _pick_llm(self):
        """Choose the right LLM based on the last message context.
        If the last tool result came from load_knowledge → use story LLM.
        Otherwise → use extraction LLM.
        """
        for msg in reversed(self.history):
            if isinstance(msg, ToolMessage):
                # If the last tool result looks like knowledge data → story mode
                if '"new"' in msg.content or '"old"' in msg.content:
                    return self.llm_story_bound
                break
        return self.llm_extraction

    def chat(self, user_input: str) -> str:
        self.history.append(HumanMessage(content=user_input))

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            messages = [self.system_msg] + self.history

            # Pick the right LLM for this step
            active_llm = self._pick_llm()
            response = active_llm.invoke(messages)
            self.history.append(response)

            # No tool calls → final answer
            if not response.tool_calls:
                return response.content

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"\n  🔧 Agent calling: {tool_name}")

                tool_fn = TOOL_MAP.get(tool_name)
                if not tool_fn:
                    tool_result = f"Error: tool '{tool_name}' not found."
                else:
                    try:
                        tool_result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Tool error: {str(e)}"

                print(f"  ✅ {tool_name}: {str(tool_result)[:80]}...")

                self.history.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

        return "Sorry, I reached the maximum number of steps. Please try again."

    def reset(self):
        self.history = []
        print("🔄 Conversation history cleared.")

# ── MAIN LOOP ────────────────────────────────────────────────────
def main():
    agent = ItalianoAgent()

    print("=" * 55)
    print("  🇮🇹  Agente Italiano — Il tuo tutor personale")
    print("=" * 55)
    print("Commands:")
    print("  • Type an image path → agent processes the page")
    print("  • Type 'storia'      → generate story from all learned items")
    print("  • Type 'reset'       → clear conversation history")
    print("  • Type 'esci'        → quit")
    print("=" * 55 + "\n")

    while True:
        try:
            user_input = input("Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nArrivederci! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("esci", "quit", "exit"):
            print("Arrivederci! 👋")
            break

        if user_input.lower() == "reset":
            agent.reset()
            continue

        if user_input.lower() == "storia":
            user_input = "Generate an Italian story using everything I have learned so far."

        print()
        response = agent.chat(user_input)
        print(f"\nAgente: {response}\n")
        print("-" * 55)


if __name__ == "__main__":
    main()