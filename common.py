from modal import Stub, Volume

stub = Stub("image_from_audio")
DB_DIR = "/db"
TRAIN_DIR_VOLUME = Volume.from_name("adapter-TRAIN", create_if_missing=True)
TRAIN_DATASET_PATH = "/dataset"

