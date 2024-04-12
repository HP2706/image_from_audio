from modal import enter, build, gpu, Image, method, Stub, Volume
from typing import Union, List, Tuple, Any
import os
from typing import Dict
import numpy as np
from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME, Data, Embedding
from huggingface_hub import snapshot_download
SDXL_PATH = "/sdxl"
CLIP_MODEL_PATH = f"{SDXL_PATH}/text_encoder_2"
CLIP_TOKENIZER_PATH = f"{SDXL_PATH}/tokenizer_2"


def download_model():
    os.makedirs(".checkpoints", exist_ok=True)
    torch.hub.download_url_to_file(
        "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
        ".checkpoints/imagebind_huge.pth",
        progress=True,
    )

def setup_func():
    import sys
    path = '/ImageBind'
    lib_path = f'/{path}/imagebind'
    tokenizer_path = f'/{path}/bpe'
    sys.path.append(path)
    sys.path.append(lib_path)
    sys.path.append(tokenizer_path)

image = Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
    ).apt_install(
        "git",
        "clang",
    ).run_commands(
        "clang --version",
    ).run_commands(
        "apt-get update",
        "apt-get install -y libgeos-dev libproj-dev proj-data proj-bin libproj-dev",
        "apt-get install -y python3-cartopy",
        "apt-get install -y python3-vtk7",
    ).pip_install(
        "uv"
    ).env(
        {"VIRTUAL_ENV": "/usr/local"}
    ).run_commands(
        "git clone https://github.com/facebookresearch/ImageBind.git",
        "cd ImageBind && pip install -e .",
        "pip install torch",
        "pip install datasets",
        "pip install pyarrow",
        "pip install numpy",
        "pip install huggingface_hub",
        "pip install pandas",
        "pip install transformers",
        "pip install diffusers",
        "pip install hf-transfer",
        "pip install accelerate",
        "pip install -U pydantic",
    ).run_function(
        setup_func
    ).run_function(
        download_model
    ).env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )

with image.imports():
    import torch
    from imagebind.models import imagebind_model
    import imagebind.data as data
    import pyarrow as pa
    import tempfile
    import numpy as np

GPU_CONCURRENCY = 20
GPU_CONFIG = gpu.A10G()

@stub.cls(
    image=image,
    concurrency_limit=GPU_CONCURRENCY,
    allow_concurrent_inputs=True,
    gpu = GPU_CONFIG,
)
class ImageBindModel:
    @enter()
    def load_model(self):
        print("gpu memory before model load", torch.cuda.memory_allocated())
        self.model = imagebind_model.imagebind_huge(pretrained=True) 
        self.model.eval()
        self.model.to("cuda")
        self.device = torch.device("cuda")
        print("gpu memory after model load in gb", torch.cuda.memory_allocated() / 1e9) 
        self.gb_left = torch.cuda.get_device_properties(0).total_memory / 1e9 - torch.cuda.memory_allocated() / 1e9
        print("gb left", self.gb_left)

    @method()
    async def embed(self, batch : List[Data]) -> List[Embedding]:
        assert isinstance(batch, list)
        assert isinstance(batch[0], Data)
        original_dir = os.getcwd()
        os.chdir('/ImageBind')
        data_dict = { 
            'text': data.load_and_transform_text([x.text for x in batch], self.device), # we only use the text part
        }
        import time
        t0 = time.time()
        with torch.no_grad():
            embeddings_dict = self.model(data_dict)
        print("time to embed", time.time() - t0, "for batch size", len(batch))
        os.chdir(original_dir)
        embeddings = embeddings_dict['text'].cpu()
        return [Embedding(embedding=np.array(tensor), id=id) for tensor, id in zip(embeddings.unbind(dim=0), [x.id for x in batch])]

@stub.cls(
    image=image,
    concurrency_limit=GPU_CONCURRENCY,
    allow_concurrent_inputs=True,
    gpu = GPU_CONFIG,
)
class DiffusionEncoder():
    @build()
    def build(self):
        snapshot_download(
        repo_id="stabilityai/sdxl-turbo", local_dir=SDXL_PATH
    )
    @enter()
    def load(self):
        print("loading")
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        print("loading model")
        import os
        
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_PATH).to("cuda") #type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_TOKENIZER_PATH)
    
    @method()
    def chunk_tokenize(self,  texts: List[Data]) -> List[Data]:
        '''Tokenizes texts into chunks of up to 77 tokens, ensuring no chunk exceeds the model's max sequence length.'''
        max_model_length = 77  # This should be set to the maximum length your model can handle
        tokenized_texts = []
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text.text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  
            chunked_token_ids = [token_ids[i:i + max_model_length] for i in range(0, len(token_ids), max_model_length)]
            chunked_texts = [self.tokenizer.decode(chunk) for chunk in chunked_token_ids]
            tokenized_texts.extend([Data(text=chunk, id = f"{text.id}_{i}") for i, chunk in enumerate(chunked_texts)])
        return tokenized_texts

    @method()
    def embed(self, batch : List[Data])->List[Embedding]:
        inputs = self.tokenizer([x.text for x in batch], padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.text_encoder(**inputs).text_embeds.cpu().numpy()
            print("outputs shape", outputs.shape)
            outputs = outputs.tolist() #type: ignore
        return [Embedding(embedding=np.array(tensor), id=id) for tensor, id in zip(outputs, [x.id for x in batch])]

