from modal import enter, build, gpu, Image, method, Stub, Volume
from typing import Union, List, Tuple, Any
import os
from typing import Dict
import numpy as np
from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME
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
        "pip install hf-transfer"
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
    async def embed(self, batch : List[str]) -> List[np.ndarray]:
        ''' 
            we have to send receive the data 
            as dict although it definitely is a list of DataFormat because of serialization issues
        '''
        
        data_dict = { 
            'text': data.load_and_transform_text(batch, self.device)
        }

        original_dir = os.getcwd()
        os.chdir('/ImageBind')
        original_dir = os.getcwd()
       

        import time
        t0 = time.time()
        with torch.no_grad():
            embeddings_dict = self.model(data_dict)
        print("time to embed", time.time() - t0, "for batch size", len(data_dict.keys()))
        os.chdir(original_dir)
        embeddings = embeddings_dict['text'].cpu()
        return [np.array(tensor) for tensor in embeddings.unbind(dim=0)]


""" @stub.cls(
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
        
        model_path = os.path.join(os.getcwd(), CLIP_MODEL_PATH)
        tokenizer_path = os.path.join(os.getcwd(), CLIP_TOKENIZER_PATH)
        print(f"current_path ls", os.listdir(os.getcwd()))
        for file in os.listdir(os.getcwd()):
            if os.path.isdir(file):
                print(f"nested dir ls for file {file}", os.listdir(os.path.join(os.getcwd(), file)))
        print(f"path: {CLIP_MODEL_PATH}", os.listdir(model_path))
        print(f"path: {CLIP_TOKENIZER_PATH}", os.listdir(tokenizer_path))
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_PATH)
        print("loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_TOKENIZER_PATH)
    
    @method()
    def embed(self, batch : List[str])->List[np.ndarray]:
        inputs = self.tokenizer(batch, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_encoder(**inputs).detach().cpu().numpy().tolist() #type: ignore
        return outputs
 """