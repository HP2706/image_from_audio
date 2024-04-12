from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME
from model import ImageBindModel, image , DiffusionEncoder
from getData import download_dataset, embed_data,check, prep_df
from train_adapter import train

@stub.local_entrypoint()
def main():
    prep_df.remote()
    embed_data.remote()
    train.remote()