from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME
from model import ImageBindModel, image #,DiffusionEncoder
from getData import download_dataset

@stub.local_entrypoint()
def main():
    download_dataset.remote()


