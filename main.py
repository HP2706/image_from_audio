from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME
from model import ImageBindModel, image , DiffusionEncoder
from getData import download_dataset, embed_data

@stub.function(image = image)
def check_version():
    import pydantic
    print("pydantic version", pydantic.__version__)

@stub.local_entrypoint()
def main():
    embed_data.remote()