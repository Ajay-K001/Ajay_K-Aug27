import chromadb

from core.config import CHROMA_DIR


_COLLECTION_NAME = "idamp_memory"


def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name=_COLLECTION_NAME):
    client = get_chroma_client()
    return client.get_or_create_collection(name=name, embedding_function=None)


def store_document(doc_id, text, metadata=None):
    try:
        document = {"ids": [doc_id], "documents": [text]}
        if metadata is not None:
            document["metadatas"] = [metadata]
        get_collection().upsert(**document)
    except Exception:
        pass


def query_memory(query_text, n_results=5):
    try:
        results = get_collection().get(
            limit=n_results,
            include=["documents", "metadatas"],
        )
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        return [
            {
                "id": doc_id,
                "document": document,
                "metadata": metadata,
            }
            for doc_id, document, metadata in zip(ids, documents, metadatas)
        ]
    except Exception:
        return []
