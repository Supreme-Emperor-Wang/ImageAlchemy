import argparse, torch
from utils import output_log
from base import BaseMiner
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image


class SDXLMiner(BaseMiner):
    @classmethod
    def add_args(cls, argp: argparse.ArgumentParser):
        argp.add_argument(
            "--miner.model",
            type=str,
            default="stabilityai/stable-diffusion-xl-base-1.0",
        )
        ### Add any args here that you'd like to add-in to this miner
        ...

    def load_models(self):
        ### Load the text-to-image model
        t2i_model = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)

        ### Load the image to image model using the same pipeline (efficient)
        i2i_model = AutoPipelineForImage2Image.from_pipe(t2i_model).to("cuda")

        return t2i_model, i2i_model

    def __init__(self):
        super(SDXLMiner, self).__init__()
