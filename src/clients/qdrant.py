from distutils.command.clean import clean
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, UpdateResult, PayloadFieldSchema, TextIndexParams, \
    TokenizerType, QueryResponse, ScoredPoint, SearchParams, HnswConfigDiff, FieldCondition, MatchText, Filter, \
    MatchValue
from qdrant_client.models import VectorParams, Distance


class Qdrant:
    def __init__(self, service, config):
        self.service  = service
        self.config = config
        self.client = QdrantClient(
            host=self.config['QDRANT']['HOST'],
            port=self.config['QDRANT']['PORT'],
        )


    def create_collection(self, collection_name) -> bool:
        try:

            assert collection_name, "Collection name is required"

            if "/" in collection_name and len(collection_name.split("/")) > 1:
                collection_name = collection_name.split("/")[1]

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config['QDRANT_COLLECTION_DIM'],
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                on_disk_payload=True,
                hnsw_config=HnswConfigDiff(
                    ef_construct=128,
                    m=16,
                    on_disk=True,
                    full_scan_threshold=4096,
                ),
            )

            self.create_index(collection_name, "message")

            logger.info(f"Creating collection: {collection_name} completed")

            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise e

    def delete_collection(self, collection_name) -> bool:
        try:

            assert collection_name, "Collection name is required"

            if "/" in collection_name and len(collection_name.split("/")) > 1:
                collection_name = collection_name.split("/")[1]

            logger.info(f"Deleting collection: {collection_name}")
            return self.client.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise e

    def check_collection(self, collection_name: str) -> bool:
        try:

            assert collection_name, "Collection name is required"

            if "/" in collection_name and len(collection_name.split("/")) > 1:
                collection_name = collection_name.split("/")[1]

            logger.info(f"Checking collection: {collection_name}")
            return self.client.collection_exists(collection_name)
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            raise e

    def create_point(self, id: int, vector: list[float], payload: dict) -> PointStruct:
        try:
            assert vector, "Vector is required"

            logger.info(f"Creating point with vector: {vector[:10]}")

            return PointStruct(
                id=id,
                vector=vector,
                payload=payload
            )
        except Exception as e:
            logger.error(f"Error creating point: {e}")
            raise e

    def insert_point(self, collection_name: str, point: PointStruct):
        try:

            assert collection_name, "Collection name is required"

            assert point, "Points are required"

            if "/" in collection_name and len(collection_name.split("/")) > 1:
                collection_name = collection_name.split("/")[1]

            upsert: UpdateResult = self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            logger.info(f"Inserting vectors into collection: {collection_name} completed, with status", upsert.status)

            return upsert.status
        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            raise e

    def create_index(self, collection_name: str, payload_flied: str):
        try:

            assert collection_name, "Collection name is required"
            assert payload_flied, "Payload field is required"

            if "/" in collection_name and len(collection_name.split("/")) > 1:
                collection_name = collection_name.split("/")[1]

            logger.info(f"Creating index for collection: {collection_name}")

            index: UpdateResult = self.client.create_payload_index(
                collection_name=collection_name,
                field_name=payload_flied,
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
            logger.info(f"Creating index for collection: {collection_name} completed, with status", index.status)
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise e



    def query_points(self, collection_name: str, vector: list[float], score_threshold: Optional[float] = 0.85, text: Optional[str] = "") -> list[ScoredPoint]:
        try:

            assert collection_name, "Collection name is required"
            assert vector, "Vector is required"

            if "/" in collection_name and len(collection_name.split("/")) > 1:
                collection_name = collection_name.split("/")[1]

            query: QueryResponse= self.client.query_points(
                collection_name=collection_name,
                query=vector,
                with_payload=True,
                with_vectors=True,
                score_threshold= score_threshold,
                search_params=SearchParams(
                    hnsw_ef=128,
                    exact=True,
                )
            )

            logger.info(f"Searching for similar vectors in collection: {collection_name} completed, with {len(query.points)} results")

            return query.points

        except Exception as e:
            logger.error(f"Error searching for similar vectors: {e}")
            raise e

    def get_point(self, collection_name, sid):
        try:
            assert collection_name, "Collection name is required"
            assert sid, "ID is required"

            if "/" in collection_name and len(collection_name.split("/")) > 1:
                collection_name = collection_name.split("/")[1]

            logger.debug(f"Getting point from collection: {collection_name} with id: {sid}")

            point = self.client.query_points(collection_name=collection_name, query=sid, with_vectors=True, with_payload=True).points[0]

            logger.info(f"Getting point from collection: {collection_name} completed")

            return point
        except Exception as e:
            logger.error(f"Error getting point: {e}")
            raise e
