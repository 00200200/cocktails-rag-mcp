from langchain_community.document_loaders import JSONLoader
from pathlib import Path


def metadata_func(record: dict, metadata: dict) -> dict:
    """Metadata function for JSONLoader."""
    metadata["id"] = record.get("id")
    metadata["name"] = record.get("name")
    metadata["category"] = record.get("category")
    metadata["glass"] = record.get("glass")
    metadata["tags"] = record.get("tags")
    metadata["instructions"] = record.get("instructions")
    metadata["ingredients"] = record.get("ingredients")
    return metadata


def load_documents(path: str = "data/processed_cocktail_dataset.json"):
    """Load json and create LangChain documents."""
    path_obj = Path(path)
    if not path_obj.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        path_obj = project_root / path

    loader = JSONLoader(
        str(path_obj), jq_schema=".[]", content_key="text", metadata_func=metadata_func
    )
    return loader.load()
