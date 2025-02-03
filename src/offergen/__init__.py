from pydantic import BaseModel
from src.utils.paths import ROOT_DIR as root_dir
from src.offergen.vector_db import VectorDBService, instantiate_db_service
import json
from functools import lru_cache
from typing import Optional
from src.utils.logging import setup_logging

logger = setup_logging(logging_path=str(root_dir / 'logs' / 'digital_assistant.log'))

class Offer(BaseModel):
    category: str
    name: str
    image_url: str
    image_path: str
    offer_url: str
    short_description: str
    full_description: str


with open(root_dir / "content/json/offers/raw/offers.json", "r", encoding="utf-8") as f:
    offers_db = {int(k): Offer(**v) for k, v in json.load(f).items()}


class DBServiceSingleton:
    _instance: Optional[VectorDBService] = None

    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(
        cls,
        json_path: str,
        chunk_overlap: int = 300,
        chunk_size: int = 700,
        dump_path: str = None,
        persist_directory: str = None,
    ) -> VectorDBService:
        if cls._instance is None:
            logger.info(
                f"Initializing rag db_service for offers generation with json_path={json_path}, chunk_overlap={chunk_overlap}, chunk_size={chunk_size}, dump_path={dump_path}"
            )
            cls._instance = instantiate_db_service(
                json_path=json_path,
                chunk_overlap=chunk_overlap,
                chunk_size=chunk_size,
                dump_path=dump_path,
                persist_directory=persist_directory,
            )
            logger.info(f"Rag db_service for offers generation initialized")
        return cls._instance


db_service = DBServiceSingleton.get_instance(
    json_path=root_dir / "content/json/offers/offers_by_category.json",
    chunk_overlap=300,
    chunk_size=700,
    dump_path=root_dir / "content/json/offers/offers_docs.json",
    persist_directory=str(root_dir / "content/json/offers/offers_persist_db"),
)
