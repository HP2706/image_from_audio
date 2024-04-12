from re import split

from numpy import isin
from pyparsing import C
from common import stub, TRAIN_DATASET_PATH, EMBEDDINGS_DATASET_PATH, TRAIN_DIR_VOLUME, Data, Embedding
from model import DiffusionEncoder, ImageBindModel, image, CLIP_TOKENIZER_PATH
from modal import Image
from modal import method, enter
from typing import List
with image.imports(): #type: ignore
    import pandas as pd
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import snapshot_download, hf_hub_download
    import os 
    # Download an

@stub.function(
    volumes={TRAIN_DATASET_PATH: TRAIN_DIR_VOLUME}, 
    timeout=3000, 
    image=image
)
def download_dataset(cache=False):

    dataset = load_dataset(
        "wikipedia", "20220301.en", num_proc=os.cpu_count(),
        split="train[:10%]"
    )
    dataset.save_to_disk(TRAIN_DATASET_PATH)
        
    # Commit and save to the volume
    TRAIN_DIR_VOLUME.commit()
    print("Dataset downloaded and saved to ", TRAIN_DATASET_PATH)
    print("check if they are there", os.listdir(TRAIN_DATASET_PATH))

def batch_and_chunk_rows(df :pd.DataFrame, batch_size : int) -> List[List[Data]]:
    data = [Data(text = row['text'], id = row['id']) for i, row in df.iterrows()]
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    return batches

@stub.function(
    volumes={TRAIN_DATASET_PATH: TRAIN_DIR_VOLUME}, 
    timeout=3000, 
    image=image
)
def embed_data(cache=False):
    pd_dataset : pd.DataFrame = load_from_disk(dataset_path=TRAIN_DATASET_PATH).to_pandas() # type: ignore
    print("len original", len(pd_dataset))

    texts = []
    for i in range(0, len(pd_dataset), 1000):
        inner = []
        for (id, text) in zip(pd_dataset[i:i+1000]['text'], pd_dataset[i:i+1000]['id']):
            inner.append(Data(id=id, text=text))
        texts.append(inner)
    chunked_texts = DiffusionEncoder().chunk_tokenize.map(texts) # returns List[List[Data]]

    chunked_df = pd.DataFrame([data.model_dump() for _list in chunked_texts for data in _list])
    print("chunked_df", chunked_df.head())
    print("len", len(chunked_df))
    print("num unique ids", len(chunked_df['id'].unique()))

    for model_class in [DiffusionEncoder, ImageBindModel]:
        if model_class.__name__ == "ImageBindModel":
            batch_size = 2*4096
        else:
            batch_size = 4*4096
        model = model_class()

        print(f"embedding using model {model_class.__name__} with batch size {batch_size}")
        chunks = 4
        chunk_size = len(chunked_df) // chunks
        for chunk_index in range(chunks):
            start_idx = chunk_index * chunk_size
            if chunk_index == chunks - 1:  #Handle last chunk which might be larger
                end_idx = len(chunked_df)
            else:
                end_idx = start_idx + chunk_size
            dataset_chunk = chunked_df[start_idx:end_idx]
            batches = batch_and_chunk_rows(dataset_chunk, batch_size)
            print("Processing chunk", chunk_index + 1, "with", len(batches), "batches")
            embeddings = []
            ids = []

            for batch_embeddings in list(model.embed.map(batches)):
                embeddings.extend([emb.embedding for emb in batch_embeddings])
                ids.extend([emb.id for emb in batch_embeddings])
            
            # Save embeddings of the current chunk
            embeddings_df = pd.DataFrame({
                'id': ids,
                'embedding': embeddings
            })
            merged_df = dataset_chunk.merge(embeddings_df, on='id')
            filename = f"{EMBEDDINGS_DATASET_PATH}/embeddings_chunk_{chunk_index + 1}.parquet"
            merged_df.to_parquet(filename, index=False)
            print(f"Saved embeddings for chunk {chunk_index + 1} with size",  os.path.getsize(filename) / 1024 / 1024, "MB")
        
        TRAIN_DIR_VOLUME.commit()
