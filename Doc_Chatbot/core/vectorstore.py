from __future__ import annotations
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | None = None,
        embedding_fn=None,
        qdrant_host: str | None = None,
        qdrant_port: int = 6333,
    ) -> None:

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from config import QDRANT_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_DIM

        self._name = collection_name or QDRANT_COLLECTION
        self._persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self._embedding_fn = embedding_fn
        self._dim = EMBEDDING_DIM

        # Connect to Qdrant (server or local mode)
        if qdrant_host:
            self._client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info("Qdrant → server %s:%d", qdrant_host, qdrant_port)
        else:
            self._client = QdrantClient(path=str(self._persist_dir))
            logger.info("Qdrant → embedded at %s", self._persist_dir)

        # Create collection if missing
        existing = [c.name for c in self._client.get_collections().collections]
        if self._name not in existing:
            self._client.create_collection(
                collection_name=self._name,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info("Created collection '%s'.", self._name)

        logger.info("VectorStore ready — %d chunks", self.count())

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
        batch_size: int = 100,
    ) -> int:
        from qdrant_client.models import PointStruct

        if not chunks:
            return 0

        if embeddings is None:
            if self._embedding_fn is None:
                raise ValueError("No embedding_fn set.")
            logger.info("Generating embeddings for %d chunks…", len(chunks))
            embeddings = self._embedding_fn.embed_documents([c["text"] for c in chunks])

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]

            points = []
            for chunk, emb in zip(batch_chunks, batch_embs):
                uid = chunk.get("chunk_uid") or str(uuid.uuid4())
                int_id = abs(hash(uid)) % (2 ** 63)

                payload = {"text": chunk["text"]}
                payload.update({
                    k: v for k, v in chunk.items()
                    if k != "text" and isinstance(v, (str, int, float, bool, type(None)))
                })
                payload = {k: ("" if v is None else v) for k, v in payload.items()}

                points.append(PointStruct(id=int_id, vector=emb, payload=payload))

            self._client.upsert(collection_name=self._name, points=points)
            added += len(batch_chunks)

        logger.info("Stored %d chunks.", added)
        return added

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if self.count() == 0:
            return []

        # Generate embedding if necessary
        if query_embedding is None:
            if self._embedding_fn is None or query_text is None:
                raise ValueError(
                    "Provide query_text + embedding_fn, or query_embedding."
                )
            query_embedding = self._embedding_fn.embed_query(query_text)

        qdrant_filter = None
        if where:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in where.items()
                ]
            )

        # Use the documented query_points() API (the correct vector search API) :contentReference[oaicite:2]{index=2}
        response = self._client.query_points(
            collection_name=self._name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        hits = response.points if hasattr(response, "points") else response
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            results.append({
                "text": payload.pop("text", ""),
                "score": round(float(hit.score), 4),
                **payload,
            })
        return results

    # ── Management ────────────────────────────────────────────────────────────

    def list_sources(self) -> list[str]:
        if self.count() == 0:
            return []
        sources: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._name,
                limit=100,
                offset=offset,
                with_payload=["source"],
            )
            for r in records:
                src = (r.payload or {}).get("source", "")
                if src:
                    sources.add(src)
            if offset is None:
                break
        return sorted(sources)

    def delete_source(self, source: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        before = self.count()
        self._client.delete(
            collection_name=self._name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source))
                ]
            ),
        )
        deleted = before - self.count()
        logger.info("Deleted %d chunks for '%s'.", deleted, source)
        return deleted

    def clear(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        self._client.delete_collection(self._name)
        self._client.create_collection(
            collection_name=self._name,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
        logger.info("Collection cleared.")

    def count(self) -> int:
        return self._client.count(collection_name=self._name).count


def get_vectorstore(embedding_fn=None) -> VectorStore:
    from config import QDRANT_HOST, QDRANT_PORT
    if embedding_fn is None:
        from core.embeddings import get_embeddings
        embedding_fn = get_embeddings()
    return VectorStore(
        embedding_fn=embedding_fn,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
    )
