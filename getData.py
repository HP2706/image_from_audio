from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME
from model import ImageBindModel, image #,DiffusionEncoder
from modal import Image
from modal import method, enter
from typing import List
with image.imports(): #type: ignore
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import snapshot_download, hf_hub_download
    import os 
    # Download an

@stub.function(
    volumes={TRAIN_DATASET_PATH: TRAIN_DIR_VOLUME}, 
    timeout=3000, 
    image=image
)
def download_dataset(cache=False):

    file_names = [
        "train-00000-of-00041.parquet",
        "train-00001-of-00041.parquet",
        "train-00002-of-00041.parquet",
        "train-00003-of-00041.parquet",
        "train-00004-of-00041.parquet",
        "train-00005-of-00041.parquet",
        "train-00006-of-00041.parquet",
        "train-00007-of-00041.parquet",
        "train-00008-of-00041.parquet",
        "train-00009-of-00041.parquet",
    ]

    for file_name in file_names:
        hf_hub_download(
            repo_id='wikipedia', repo_type="dataset",
            subfolder='data/20220301.en', filename=file_name,
            local_dir=TRAIN_DATASET_PATH
        )

    # Commit and save to the volume
    TRAIN_DIR_VOLUME.commit()
    print("Dataset downloaded and saved to ", TRAIN_DATASET_PATH)
    print("check if they are there", os.listdir(TRAIN_DATASET_PATH))

def batch_rows(df, batch_size : int) -> List[List[str]]:
    return [list(df[i:i+batch_size]['text']) for i in range(0, df.shape[0], batch_size)]

@stub.function(
    volumes={TRAIN_DATASET_PATH: TRAIN_DIR_VOLUME}, 
    timeout=3000, image=image
)
def embed(cache=False):
    df = load_dataset(TRAIN_DATASET_PATH).to_pandas() # dataset is aroung 5 GB so we need to batch
    model = ImageBindModel()
    chunks = 4
    last_idx = 0
    for i in range(len(dataset), step=dataset.shape[0]//chunks):
        dataset_chunk = dataset[last_idx:i]
        last_idx = i
        
        batch_size = 1000

    embeddings = model.embed.map(batch_rows(df, batch_size), order_outputs = True)
    
    df['embedding'] = embeddings
    

