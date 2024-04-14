from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME
from model import ImageBindModel, image , DiffusionEncoder
from getData import download_dataset, embed_data,check, prep_df
from imagepipeline import try_diffusion_pipeline
from train_adapter import train
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
@stub.local_entrypoint()
def main():
    """ with TRAIN_DIR_VOLUME.batch_upload() as batch:
        batch.put_file("banana.wav", "/dataset/dataset/banana.wav")
    """


    #embed_data.remote()
    audio_path = f"{TRAIN_DATASET_PATH}/dataset/eagle.wav"
   
    output = try_diffusion_pipeline.remote(audio_path)
    print(output)
    if isinstance(output, tuple) and isinstance(output[0], list):
        image = output[0][0]  # Access the first list in the tuple, then the first image in the list
        if isinstance(image, PIL.Image.Image):
            plt.imshow(image)
            plt.show()
        elif isinstance(image, np.ndarray):
            plt.imshow(image)
            plt.show()
