from typing import Any
from modal import Stub, Volume
from pydantic import BaseModel

class Data(BaseModel):
    text : str
    id : str

class Embedding(BaseModel):
    embedding : Any
    id : str



stub = Stub("image_from_audio")
DB_DIR = "/db"
TRAIN_DIR_VOLUME = Volume.from_name("adapter-TRAIN", create_if_missing=True)
TRAIN_DATASET_PATH = "/dataset"
EMBEDDINGS_DATASET_PATH = f"{TRAIN_DATASET_PATH}/embeddings"

