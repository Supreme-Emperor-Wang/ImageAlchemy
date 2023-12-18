import argparse
import asyncio
import copy
import os
import random
import time
import traceback
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Union

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from neurons.protocol import ImageGeneration, IsAlive
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from utils import (
    BackgroundTimer,
    background_loop,
    clean_nsfw_from_prompt,
    do_logs,
    generate,
    get_caller_stake,
    nsfw_image_filter,
    output_log,
    sh,
    warm_up,
)
from wandb_utils import WandbUtils

import bittensor as bt


@dataclass
class Stats:
    start_time: datetime
    start_dt: datetime
    total_requests: int
    timeouts: int
    response_times: list


class BaseMiner(ABC):
    def __init__(self):
        #### Parse the config
        self.config = self.get_config()

        self.wandb = None

        if self.config.logging.debug:
            bt.debug()
            output_log("Enabling debug mode...", type="debug")

        #### Output the config
        output_log("Outputting miner config:", "c")
        output_log(f"{self.config}", color_key="na")

        #### Build args
        self.t2i_args, self.i2i_args = self.get_args()

        ####
        self.hotkey_blacklist = set()
        self.coldkey_blacklist = set()
        self.hotkey_whitelist = set(
            ["5C5PXHeYLV5fAx31HkosfCkv8ark3QjbABbjEusiD3HXH2Ta"]
        )

        self.storage_client = None

        #### Initialise event dict
        self.event = {}

        #### Establish subtensor connection
        output_log("Establishing subtensor connection.", "g", type="debug")
        self.subtensor = bt.subtensor(config=self.config)

        #### Create the metagraph
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)

        #### Configure the wallet
        self.wallet = bt.wallet(config=self.config)

        #### Wait until the miner is registered
        self.loop_until_registered()

        ### Defaults
        self.stats = self.get_defaults()

        ### Start the wandb logging thread if both project and entity have been provided
        if all(
            [
                self.config.wandb.project,
                self.config.wandb.entity,
                self.config.wandb.api_key,
            ]
        ):
            self.wandb = WandbUtils(
                self, self.metagraph, self.config, self.wallet, self.event
            )

        #### Start the generic background loop
        self.background_steps = 1
        self.background_timer = BackgroundTimer(300, background_loop, [self])

    def start_axon(self):
        #### Serve the axon
        output_log(f"Serving axon on port {self.config.axon.port}.", "g", type="debug")
        self.axon = (
            bt.axon(
                wallet=self.wallet,
                external_ip=bt.utils.networking.get_external_ip(),
                port=self.config.axon.port,
            )
            .attach(
                forward_fn=self.is_alive,
                blacklist_fn=self.blacklist_is_alive,
                priority_fn=self.priority_is_alive,
            )
            .attach(
                forward_fn=self.generate_image,
                blacklist_fn=self.blacklist_image_generation,
                priority_fn=self.priority_image_generation,
            )
            .start()
        )
        output_log(f"Axon created: {self.axon}", "g", type="debug")

        self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)

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
        }, {
            "guidance_scale": self.config.miner.guidance_scale,
            "num_inference_steps": self.config.miner.steps,
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

        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        processor = CLIPImageProcessor()

        return t2i_model, i2i_model, safety_checker, processor

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

    def is_alive(self, synapse: IsAlive) -> IsAlive:
        timeout = synapse.timeout
        start_time = time.perf_counter()
        bt.logging.info("IsAlive")
        synapse.completion = "True"
        if time.perf_counter() - start_time > timeout:
            self.stats.timeouts += 1
        return synapse

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """
        Image generation logic shared between both text-to-image and image-to-image
        """
        timeout = synapse.timeout
        self.stats.total_requests += 1
        start_time = time.perf_counter()

        ### Set up args
        local_args = copy.deepcopy(self.mapping[synapse.generation_type]["args"])
        local_args["prompt"] = [clean_nsfw_from_prompt(synapse.prompt)]
        local_args["target_size"] = (synapse.height, synapse.width)

        ### Get the model
        model = self.mapping[synapse.generation_type]["model"]

        if synapse.generation_type == "image_to_image":
            local_args["image"] = T.transforms.ToPILImage()(
                bt.Tensor.deserialize(synapse.prompt_image)
            )
            del local_args["num_inference_steps"]

        ### Output logs
        do_logs(self, synapse, local_args)

        ### Generate images & serialize

        for attempt in range(3):
            try:
                images = model(**local_args).images
                synapse.images = [
                    bt.Tensor.serialize(transform(image)) for image in images
                ]
                output_log(
                    f"{sh('Generating')} -> Succesful image generation after {attempt+1} attempt(s)"
                )
                break
            except Exception as e:
                bt.logging.error(
                    f"errror in attempt number {attempt+1} to generate an image"
                )
                asyncio.sleep(5)
                if attempt == 2:
                    images = []
                    synapse.images = []

        if time.perf_counter() - start_time > timeout:
            self.stats.timeouts += 1

        ### Log NSFW images
        if any(nsfw_image_filter(self, images)):
            bt.logging.debug(f"NSFW image detected in outputs")
            synapse.images = []

        ### Log to wandb
        if self.wandb:
            ### Store the images and prompts for uploading to wandb
            self.wandb._add_images(synapse)

            #### Log to Wandb
            self.wandb._log()

        #### Log to console
        output_log(
            f"{sh('Time')} -> {time.perf_counter() - start_time:.2f}s.", color_key="y"
        )
        return synapse

    def _base_priority(self, synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    def _base_blacklist(
        self, synapse, vpermit_tao_limit=1024
    ) -> typing.Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            caller_stake = get_caller_stake(self, synapse)

            if hotkey in self.hotkey_whitelist:
                bt.logging.trace(f"Whitelisting hotkey {synapse.dendrite.hotkey}")
                return False, "Whitelisted hotkey recognized"

            if caller_stake is None:
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey: {synapse.dendrite.hotkey}"
                )
                return (
                    True,
                    f"Blacklisted a non-registered hotkey's {synapse_type} request from {hotkey}",
                )

            # Check stake if uid is recognized
            if caller_stake < vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted a low stake {synapse_type} request: {caller_stake} < {vpermit_tao_limit} from {hotkey}",
                )

            bt.logging.trace(f"Allowing recognized hotkey {synapse.dendrite.hotkey}")
            return False, "Hotkey recognized"

        except Exception as e:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")

    def blacklist_is_alive(self, synapse: IsAlive) -> typing.Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def blacklist_image_generation(
        self, synapse: ImageGeneration
    ) -> typing.Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def priority_is_alive(self, synapse: IsAlive) -> float:
        return self._base_priority(synapse)

    def priority_image_generation(self, synapse: ImageGeneration) -> float:
        return self._base_priority(synapse)

    def loop(self):
        output_log("Starting miner loop.", "g", type="debug")
        step = 0
        while True:
            #### Check the miner is still registered
            is_registered = self.check_still_registered()

            if not is_registered:
                output_log("The miner is not currently registered.", "r")
                time.sleep(120)

                ### Ensure the metagraph is synced before the next registration check
                self.metagraph.sync(lite=True)
                continue

            #### Output current statistics and set weights
            try:
                if step % 5 == 0:
                    #### Output metrics
                    log = (
                        f"Step: {step} | "
                        f"Block: {self.metagraph.block.item()} | "
                        f"Stake: {self.metagraph.S[self.miner_index]:.2f} | "
                        f"Rank: {self.metagraph.R[self.miner_index]:.2f} | "
                        f"Trust: {self.metagraph.T[self.miner_index]:.2f} | "
                        f"Consensus: {self.metagraph.C[self.miner_index]:.2f} | "
                        f"Incentive: {self.metagraph.I[self.miner_index]:.2f} | "
                        f"Emission: {self.metagraph.E[self.miner_index]:.2f}"
                    )
                    output_log(log, "g")

                step += 1
                time.sleep(60)

            #### If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            #### In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue
