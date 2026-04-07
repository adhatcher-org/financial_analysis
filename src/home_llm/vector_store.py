from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

import requests

from home_llm.config import QdrantConfig
from home_llm.models import SearchResult


class QdrantStore:
    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["api-key"] = config.api_key

    def ensure_collection(self, vector_size: int) -> None:
        response = requests.get(
            f"{self.config.base_url}/collections/{self.config.collection}",
            headers=self.headers,
            timeout=30,
        )
        if response.status_code == 200:
            return
        create = requests.put(
            f"{self.config.base_url}/collections/{self.config.collection}",
            headers=self.headers,
            json={
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine",
                }
            },
            timeout=30,
        )
        create.raise_for_status()

    def replace_document_chunks(self, file_path: str, points: list[dict[str, Any]]) -> None:
        self.delete_document(file_path)
        if not points:
            return
        response = requests.put(
            f"{self.config.base_url}/collections/{self.config.collection}/points",
            headers=self.headers,
            json={"points": points},
            timeout=120,
        )
        response.raise_for_status()

    def delete_document(self, file_path: str) -> None:
        response = requests.post(
            f"{self.config.base_url}/collections/{self.config.collection}/points/delete",
            headers=self.headers,
            json={
                "filter": {
                    "must": [
                        {"key": "file_path", "match": {"value": file_path}},
                    ]
                }
            },
            timeout=60,
        )
        if response.status_code not in {200, 404}:
            response.raise_for_status()

    def search(self, embedding: list[float], limit: int) -> list[SearchResult]:
        response = requests.post(
            f"{self.config.base_url}/collections/{self.config.collection}/points/search",
            headers=self.headers,
            json={
                "vector": embedding,
                "limit": limit,
                "with_payload": True,
            },
            timeout=60,
        )
        response.raise_for_status()
        rows = response.json()["result"]
        return [
            SearchResult(
                chunk_id=0,
                file_path=str(item["payload"]["file_path"]),
                doc_type=str(item["payload"]["doc_type"]),
                page_label=str(item["payload"]["page_label"]),
                content=str(item["payload"]["content"]),
                score=float(item["score"]),
            )
            for item in rows
        ]


def build_point_id(file_path: str, page_label: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{file_path}:{page_label}:{chunk_index}"))
