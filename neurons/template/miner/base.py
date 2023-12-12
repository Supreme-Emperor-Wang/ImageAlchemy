import argparse
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from template.miner.utils import output_log
from template.validator.reward import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

import bittensor as bt


@dataclass
class Stats:
    start_time: datetime
    start_dt: datetime
    total_requests: int
    timeouts: int
    response_times: list


class BaseMiner(ABC):
    def get_defaults(self):
        now = datetime.now()
        stats = Stats(
            start_time=now,
            start_dt=datetime.strftime(now, "%Y/%m/%d %H:%M"),
            total_requests=0,
            timeouts=0,
            response_times=[],
        )
        return stats

    def get_args(self) -> Dict:
        return {
            "guidance_scale": self.config.miner.guidance_scale,
            "num_inference_steps": self.config.miner.steps,
            "num_images_per_prompt": self.config.miner.num_images,
            "generator": torch.Generator(device=self.config.miner.device).manual_seed(
                self.config.miner.seed
            ),
        }, {
            "guidance_scale": self.config.miner.guidance_scale,
            "num_inference_steps": self.config.miner.steps,
            "num_images_per_prompt": self.config.miner.num_images,
            "generator": torch.Generator(device=self.config.miner.device).manual_seed(
                self.config.miner.seed
            ),
        }

    def get_config(self) -> "bt.config":
        argp = argparse.ArgumentParser(description="Miner Configs")

        #### Add any args from the parent class
        self.add_args(argp)

        argp.add_argument("--netuid", type=int, default=1)
        argp.add_argument("--wandb.project", type=str, default="")
        argp.add_argument("--wandb.entity", type=str, default="")
        argp.add_argument("--wandb.api_key", type=str, default="")
        argp.add_argument("--miner.device", type=str, default="cuda:0")

        seed = random.randint(0, 100_000_000_000)
        argp.add_argument("--miner.seed", type=int, default=seed)

        argp.add_argument("--miner.guidance_scale", type=float, default=7.5)
        argp.add_argument("--miner.steps", type=int, default=30)
        argp.add_argument("--miner.num_images", type=int, default=1)
        argp.add_argument(
            "--miner.model",
            type=str,
            default="stabilityai/stable-diffusion-xl-base-1.0",
        )
        argp.add_argument(
            "--miner.nsfw_filter",
            action="store_true",
            help="Applies an nsfw filter on the miner's outputs",
            default=True,
        )

        bt.subtensor.add_args(argp)
        bt.logging.add_args(argp)
        bt.wallet.add_args(argp)
        bt.axon.add_args(argp)

        config = bt.config(argp)

        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "miner",
            )
        )
        #### Ensure the directory for logging exists
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path, exist_ok=True)

        return config

    def load_models(self):
        ### Load the text-to-image model
        t2i_model = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)
        t2i_model.set_progress_bar_config(disable=True)

        ### Load the image to image model using the same pipeline (efficient)
        i2i_model = AutoPipelineForImage2Image.from_pipe(t2i_model).to(
            self.config.miner.device,
        )
        i2i_model.set_progress_bar_config(disable=True)

        safetychecker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        processor = CLIPImageProcessor()
        return t2i_model, i2i_model, safetychecker, processor

    def add_args(cls, argp: argparse.ArgumentParser):
        pass

    def loop_until_registered(self):
        index = None
        while True:
            index = self.get_miner_index()
            if index is not None:
                self.miner_index = index
                output_log(
                    f"Miner {self.config.wallet.hotkey} is registered on uid {self.metagraph.uids[self.miner_index]}.",
                    "g",
                )
                break
            output_log(
                f"Miner {self.config.wallet.hotkey} is not registered. Sleeping for 30 seconds...",
                "r",
            )
            time.sleep(120)
            self.metagraph.sync(lite=True)

    def get_miner_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.S[self.miner_index],
            "trust": self.metagraph.T[self.miner_index],
            "consensus": self.metagraph.C[self.miner_index],
            "incentive": self.metagraph.I[self.miner_index],
            "emissions": self.metagraph.E[self.miner_index],
        }

    def get_miner_index(self):
        """
        Retrieve the given miner's index in the metagraph.
        """
        index = None
        try:
            index = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            pass
        return index

    def check_still_registered(self):
        self.miner_index = self.get_miner_index()
        return True if self.miner_index is not None else False

    def get_incentive(self):
        return (
            self.metagraph.I[self.miner_index] * 100_000
            if self.miner_index is not None
            else 0
        )

    def get_trust(self):
        return (
            self.metagraph.T[self.miner_index] * 100
            if self.miner_index is not None
            else 0
        )

    def get_consensus(self):
        return (
            self.metagraph.C[self.miner_index] * 100_000
            if self.miner_index is not None
            else 0
        )
