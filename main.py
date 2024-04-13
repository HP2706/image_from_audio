from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME
from model import ImageBindModel, image , DiffusionEncoder
from getData import download_dataset, embed_data,check, prep_df
from imagepipeline import try_diffusion_pipeline
from train_adapter import train


@stub.local_entrypoint()
def main():
    audio_path = f"{TRAIN_DATASET_PATH}/dataset/eagle.wav"
   
    try_diffusion_pipeline.remote(audio_path)

