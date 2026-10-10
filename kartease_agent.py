from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

from answer_policies import (
    answer_policy_question,
    FALLBACK_MESSAGE,
    is_clearly_unrelated,
)

from order_lookup import lookup_order


# Load environment variables
load_dotenv()


# ============================================================
# TOOL 1: POLICY SEARCH
# ============================================================

@tool
def search_policy(question: str) -> str:
    """Search KartEase policy documents for policy questions."""
    return answer_policy_question(question)


# ============================================================
# TOOL 2: ORDER LOOKUP
# ============================================================

@tool
def check_order(order_id: str) -> str:
    """Retrieve a KartEase order using its order ID."""
    return lookup_order(order_id)


tools = [search_policy, check_order]

tools_by_name = {
    selected_tool.name: selected_tool
    for selected_tool in tools
}


# ============================================================
# INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def ask_kartease(question: str) -> str:
    """Answer a question using KartEase tools when appropriate."""

    question = question.strip()

    if not question:
        return "Please enter a question."

    # Handle clearly unrelated questions without calling Gemini
    if is_clearly_unrelated(question):
        return f"{FALLBACK_MESSAGE}\nTools used: None"

    messages = [
        HumanMessage(
            content=(
                "You are KartEase's customer support assistant.\n"
                "You must follow these rules:\n"
                "1. For questions about KartEase policies, use "
                "search_policy.\n"
                "2. For questions about a specific order, use "
                "check_order.\n"
                "3. For questions requiring both policy information "
                "and order information, use BOTH tools.\n"
                "4. Never invent policy rules or order details.\n"
                "5. If an order ID is missing, ask the customer "
                "for it.\n"
                f"6. For unrelated questions, reply exactly: "
                f'"{FALLBACK_MESSAGE}"\n'
                "7. Keep answers concise and customer-friendly.\n\n"
                f"Customer question: {question}"
            )
        )
    ]

    used_tools = []

    # Allow multiple rounds of tool calls
    for _ in range(5):

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        # Return the final answer if no more tools are requested
        if not response.tool_calls:

            content = response.content

            if isinstance(content, str):
                answer = content.strip()

            elif isinstance(content, list):
                text_parts = []

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(
                                block.get("text", "")
                            )

                    elif hasattr(block, "text"):
                        text_parts.append(block.text)

                answer = "\n".join(text_parts).strip()

            else:
                answer = str(content).strip()

            if not answer:
                answer = (
                    "Sorry, I couldn't generate an answer. "
                    "Please try again."
                )

            if FALLBACK_MESSAGE in answer:
                return (
                    f"{FALLBACK_MESSAGE}\n"
                    f"Tools used: "
                    f"{', '.join(used_tools) if used_tools else 'None'}"
                )

            if used_tools:
                return (
                    f"{answer}\n\n"
                    f"Tools used: {', '.join(used_tools)}"
                )

            return (
                f"{answer}\n\n"
                "Tools used: None"
            )

        # Execute every requested tool
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            selected_tool = tools_by_name.get(tool_name)

            if selected_tool is None:
                result = f"Unknown tool: {tool_name}"

            else:
                if tool_name not in used_tools:
                    used_tools.append(tool_name)

                try:
                    result = selected_tool.invoke(tool_args)

                except Exception as error:
                    result = f"Tool error: {error}"

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                )
            )

    return (
        "I couldn't complete your request. Please try again.\n\n"
        f"Tools used: {', '.join(used_tools) if used_tools else 'None'}"
    )


# ============================================================
# RUN THE KARTEASE ASSISTANT
# ============================================================

if __name__ == "__main__":

    print("=" * 45)
    print("       KartEase AI Customer Support")
    print("=" * 45)

    print(
        "Ask about orders, returns, refunds, shipping,\n"
        "warranties, and other KartEase policies."
    )

    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("KartEase: Goodbye!")
            break

        if not question:
            continue

        print("\nKartEase: Thinking...\n")

        try:
            answer = ask_kartease(question)
            print(f"KartEase: {answer}\n")

        except Exception as error:
            print(f"An error occurred: {error}\n")