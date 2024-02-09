import torch
from base import BaseMiner
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from utils import output_log, warm_up

import bittensor as bt


class StableMiner(BaseMiner):
    def __init__(self):
        super().__init__()

        #### Load the model
        self.load_models()

        #### Optimize model
        self.optimize_models()

        #### Serve the axon
        self.start_axon()

        #### Start the miner loop
        self.loop()

    def load_models(self):
        ### Load the text-to-image model
        self.t2i_model = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)
        self.t2i_model.set_progress_bar_config(disable=True)
        self.t2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.t2i_model.scheduler.config
        )

        ### Load the image to image model using the same pipeline (efficient)
        self.i2i_model = AutoPipelineForImage2Image.from_pipe(self.t2i_model).to(
            self.config.miner.device,
        )
        self.i2i_model.set_progress_bar_config(disable=True)
        self.i2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.i2i_model.scheduler.config
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        self.processor = CLIPImageProcessor()

        ### Set up mapping for the different synapse types
        self.mapping = {
            "text_to_image": {"args": self.t2i_args, "model": self.t2i_model},
            "image_to_image": {"args": self.i2i_args, "model": self.i2i_model},
        }

    def optimize_models(self):
        if self.config.miner.optimize:
            self.t2i_model.unet = torch.compile(
                self.t2i_model.unet, mode="reduce-overhead", fullgraph=True
            )

            #### Warm up model
            output_log(
                ">>> Warming up model with compile... this takes roughly two minutes...",
                color_key="y",
            )
            warm_up(self.t2i_model, self.t2i_args)
