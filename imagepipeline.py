from re import I
from common import stub, TRAIN_DATASET_PATH, TRAIN_DIR_VOLUME, EMBEDDINGS_DATASET_PATH, Data, Embedding
from model import DiffusionEncoder, ImageBindModel, image, SDXL_PATH, CLIP_TOKENIZER_PATH, CLIP_MODEL_PATH
from modal import gpu, method, enter, Image
import torch
import os
from io import BytesIO
from typing import List, Optional, Union
from torch import nn
from torch.nn import functional as F
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from accelerate import cpu_offload_with_hook 
import tempfile
from typing import Dict, Any, Callable, List, Optional, Tuple, Union
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.schedulers import PNDMScheduler, LMSDiscreteScheduler, PNDMScheduler
import inspect
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from train_adapter import Adapter
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline, 
    retrieve_timesteps,
    rescale_noise_cfg
)

import json

with image.imports():
    import imagebind.data as data
    from imagebind.models import imagebind_model

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value






class AudioToImgPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "imagebind->adapter->unet->vae"

    def __init__(
        self,
        imagebind : Any,
        adapter: Adapter,
        tokenizer: CLIPTokenizer,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        image_processor : VaeImageProcessor,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            adapter=adapter,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            imagebind=imagebind,
            image_processor=image_processor
        )
        self.register_to_config(mean=mean, std=std)

    # Copied from diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_audio(self, audio_path : List[str], device=None):
        device = device or self._execution_device

        ''' os.chdir('/ImageBind')
        data_dict = { 
            'audio': data.load_and_transform_audio_data(audio_path, device), # we only use the text part
        }
        with torch.no_grad():
            #self.imagebind.to(device)
            print("loading imagebind to device", device)
            embeddings_dict = self.imagebind(data_dict)
        embeddings = embeddings_dict['audio']
        #self.imagebind.to('cpu')
        #self.adapter.to(device)
        print("loading adapter to device", device) '''
        emb = torch.randn([len(audio_path), 1024]).to('cuda')
        aligned_audio_embeddings = self.adapter.forward(emb)

        aligned_audio_embeddings = aligned_audio_embeddings.unsqueeze(1)
        return aligned_audio_embeddings

    @torch.no_grad()
    def __call__(
        self,
        audio_path: List[str],
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            reference_image (`PIL.Image.Image`):
                The reference image to condition the generation on.
            source_subject_category (`List[str]`):
                The source subject category.
            target_subject_category (`List[str]`):
                The target subject category.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by random sampling.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            height (`int`, *optional*, defaults to 512):
                The height of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            neg_prompt (`str`, *optional*, defaults to ""):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_strength (`float`, *optional*, defaults to 1.0):
                The strength of the prompt. Specifies the number of times the prompt is repeated along with prompt_reps
                to amplify the prompt.
            prompt_reps (`int`, *optional*, defaults to 20):
                The number of times the prompt is repeated along with prompt_strength to amplify the prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        device = "cuda"
        assert isinstance(audio_path, list), "audio_path should be a list of strings"
        assert all(isinstance(path, str) for path in audio_path), "audio_path should be a list of strings"
        batch_size = len(audio_path)
        audio_embedding = self.encode_audio(audio_path, device)
        print("audio_embedding", audio_embedding.shape, audio_embedding.device)
        do_classifier_free_guidance = guidance_scale > 1.0
      
      
        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=height // scale_down_factor,
            width=width // scale_down_factor,
            generator=generator,
            latents=latents,
            dtype=self.unet.dtype,
            device=device,
        )
        print("latents", latents.shape, latents.device)
        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        self.unet.to(device)
        self.vae.to(device)
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            do_classifier_free_guidance = guidance_scale > 1.0
            self.unet.config.addition_embed_type = "none" 
            latent_model_input =  latents
            print("latent_model_input", latent_model_input.shape, latent_model_input.device)
            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=audio_embedding,
                added_cond_kwargs = {'image_embeds' : torch.zeros(audio_embedding.shape) }, # hack
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]


            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

def download_audio(url: str) -> BytesIO:
    import requests
    response = requests.get(url)
    audio_file = BytesIO(response.content)
    return audio_file

@stub.function(
    image = image.run_commands("pip install -U diffusers"),    
    gpu = gpu.A10G(), #24gb
    volumes={TRAIN_DATASET_PATH: TRAIN_DIR_VOLUME}
)
def try_diffusion_pipeline(audio_path : str):

    if "https" in audio_path:
        audio_file = download_audio(audio_path)
        #save as a tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(audio_file.read())
        temp_file.close()
        audio_path = temp_file.name
    else:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        print("audio_path", audio_path)

    cfg = json.load(open(f"{EMBEDDINGS_DATASET_PATH}/model_1.json"))
    print("config", cfg)
    adapter = Adapter(
    input_dim=cfg['input_dim'], hidden_dim=2*cfg['input_dim'], output_dim=cfg['output_dim'], n_layers=cfg['n_layers']
    )
    adapter.load_state_dict(
        torch.load(f"{EMBEDDINGS_DATASET_PATH}/model_1.pt")
    )

    pipeline = AudioToImgPipeline(
        imagebind= None,#imagebind_model.imagebind_huge(pretrained=True),
        adapter=adapter, #type: ignore
        vae=AutoencoderKL.from_pretrained(SDXL_PATH, subfolder="vae"), #type: ignore
        tokenizer=CLIPTokenizer.from_pretrained(SDXL_PATH, subfolder="tokenizer"), #type: ignore
        unet=UNet2DConditionModel.from_pretrained(SDXL_PATH, subfolder ="unet"), #type: ignore
        image_processor=VaeImageProcessor(),
        scheduler=PNDMScheduler(),
    ).to("cuda")
    #pipeline = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
    return pipeline(audio_path=[audio_path], output_type="pil", return_dict=False, num_inference_steps=20)
