from common import stub, TRAIN_DIR_VOLUME, TRAIN_DATASET_PATH, EMBEDDINGS_DATASET_PATH
from modal import Image, gpu
from model import image
import json

with image.imports():
    from torch import nn
    from torch.optim import Adam
    import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.intermediate_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.non_lins = nn.ModuleList([nn.GELU() for _ in range(n_layers)])
        self.l_out = nn.Linear(hidden_dim, output_dim)
        self.dtype = torch.float16
    
    def forward(self, x):
        #x is of shape [batch_size, input_dim]
        x = self.l1(x)
        print("l1", x)
        for intermediate_layer, non_lin in zip(self.intermediate_layers, self.non_lins):
            x = non_lin(intermediate_layer(x))
        return self.l_out(x)

with image.imports():
    import os
    import pandas as pd
    import torch

@stub.function(    
    image = image,
    gpu = gpu.T4(),
    volumes={TRAIN_DATASET_PATH: TRAIN_DIR_VOLUME}, 
    timeout=3000,
)
def train():
    parquet_files = os.listdir(EMBEDDINGS_DATASET_PATH)
    image_bind_df = pd.concat(
        [pd.read_parquet(os.path.join(EMBEDDINGS_DATASET_PATH, file)) for file in parquet_files if 'ImageBindModel' in file]
    ).rename(columns={'embedding': 'imagebind_embedding'})
    diffusion_df = pd.concat(
        [pd.read_parquet(os.path.join(EMBEDDINGS_DATASET_PATH, file)) for file in parquet_files if 'DiffusionEncoder' in file]
    ).rename(columns={'embedding': 'clip_embedding'})

    input_dim = torch.tensor(image_bind_df.iloc[0]['imagebind_embedding']).shape[0]
    print("imagebind_embedding dim", input_dim)
    output_dim = torch.tensor(diffusion_df.iloc[0]['clip_embedding']).shape[0]
    print("clip_embedding dim", output_dim)

    df = image_bind_df.merge(diffusion_df, on='id')

    #define model
    dtype = torch.float32
    n_epochs = 10
    lr = 0.001
    batch_size = 500
    #load dataset
    for n_layers in [1, 2, 3]:
        model = Adapter(input_dim=input_dim, hidden_dim=input_dim*2, output_dim=output_dim, n_layers=n_layers)
        model.to("cuda").to(dtype)
        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            for i in range(0, len(df), batch_size):
                batch = df[i:i+batch_size] 
                x = torch.tensor(list(batch['imagebind_embedding'])).to("cuda").to(dtype)
                y = torch.tensor(list(batch['clip_embedding'])).to("cuda").to(dtype)
                print(x.shape, y.shape)
                print("isnan?", torch.isnan(x).any(), torch.isnan(y).any())
                y_pred = model.forward(x)
                print("ypred", y_pred)
                print("y", y)
                print("mse", torch.mean(torch.square(y_pred - y)))
                loss = F.mse_loss(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"mse loss: {loss.item()} at index {i} epoch {epoch}")

        torch.save(model.state_dict(), f"{EMBEDDINGS_DATASET_PATH}/model_{n_layers}.pt")
        with open(f"{EMBEDDINGS_DATASET_PATH}/model_{n_layers}.json", 'w') as f:
            f.write(json.dumps({'input_dim': input_dim, 'output_dim': output_dim, 'n_layers': n_layers}))
        TRAIN_DIR_VOLUME.commit()

