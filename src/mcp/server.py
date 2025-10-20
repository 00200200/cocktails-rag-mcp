import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP
from src.rag.rag import RAG

mcp = FastMCP("cocktails")

rag = RAG()


def process_query(user_question: str) -> str:
    """Helper function to process RAG queries and return answers."""
    state = {
        "user_question": user_question,
        "context": [],
        "reranked_context": [],
        "response": "",
    }
    state = rag.retrieve(state)
    state = rag.rerank_context(state)
    state = rag.generate_answer(state)
    return state["response"]


@mcp.tool
def suggest_cocktail(text: str) -> str:
    """Suggest a cocktail based on user preferences.

    Args:
        text: The text to suggest a cocktail
    """
    return process_query(text)


@mcp.tool
def ask_about_cocktail(text: str) -> str:
    """Ask any question about cocktails.

    Args:
        text: The question to ask about cocktails
    """
    return process_query(text)


@mcp.tool
def suggest_cocktail_by_ingredients(ingredients: list[str]) -> str:
    """Suggest a cocktail based on specific ingredients.

    Args:
        ingredients: List of ingredients you have or want to use
    """
    ingredients_str = ", ".join(ingredients)
    return process_query(
        f"Suggest a cocktail that can be made with these ingredients: {ingredients_str}"
    )


@mcp.tool
def get_cocktail_by_name(name: str) -> str:
    """Get detailed information about a specific cocktail by its name.

    Args:
        name: The name of the cocktail
    """
    return process_query(f"Tell me everything about the cocktail: {name}")


@mcp.tool
def get_cocktail_by_tags(tags: list[str]) -> str:
    """Find cocktails by tags.

    Args:
        tags: List of tags to filter by

    """
    tags_str = ", ".join(tags)
    return process_query(f"Find cocktails with these tags: {tags_str}")


@mcp.tool
def get_cocktail_by_glass(glass: str) -> str:
    """Find cocktails served in a specific type of glass.

    Args:
        glass: Type of glass

    """
    return process_query(f"Find cocktails served in: {glass}")


@mcp.tool
def get_cocktail_by_category(category: str) -> str:
    """Find cocktails by category.

    Args:
        category: The category to search for

    """
    return process_query(f"Find cocktails in category: {category}")


if __name__ == "__main__":
    mcp.run()
