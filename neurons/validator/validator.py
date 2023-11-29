# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Bittensor Validator Template:
# TODO(developer): Rewrite based on protocol defintion.

import argparse
import asyncio
import copy

# Step 1: Import necessary libraries and modules
import os
import random
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from traceback import print_exception
from typing import List

# import this repo
import torch
from datasets import load_dataset
from forward import run_step
from openai import OpenAI
from reward import (
    BlacklistFilter,
    DiversityRewardModel,
    ImageRewardModel,
    NSFWRewardModel,
)
from template.protocol import IsAlive
from transformers import pipeline
from utils import (
    generate_followup_prompt_gpt,
    generate_random_prompt,
    generate_random_prompt_gpt,
    get_random_uids,
    init_wandb,
    ttl_get_block,
)
from weights import set_weights, should_set_weights

import bittensor as bt
from config import add_args, check_config, config


class neuron:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    def __init__(self):
        # Init config
        self.config = neuron.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.neuron.full_path)
        bt.logging.info("neuron.__init__()")

        # Init device.
        bt.logging.debug("loading", "device")
        self.device = torch.device(self.config.neuron.device)
        bt.logging.debug(str(self.device))

        # Init seed
        bt.logging.debug("setting", "seed")
        self.seed = random.randint(0, 1000000)
        bt.logging.debug(str(self.seed))

        # Init dataset
        bt.logging.debug("loading", "dataset")
        self.dataset = iter(
            load_dataset("poloclub/diffusiondb")["train"]
            .shuffle(seed=self.seed)
            .to_iterable_dataset()
        )

        # Init prompt generation model
        bt.logging.debug("loading", "prompt generation model")
        self.prompt_generation_pipeline = pipeline(
            "text-generation", model="succinctly/text2image-prompt-generator"
        )
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Init subtensor
        bt.logging.debug("loading", "subtensor")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(str(self.subtensor))

        # Init wallet.
        bt.logging.debug("loading", "wallet")
        self.wallet = bt.wallet(config=self.config)
        self.wallet.create_if_non_existent()
        if not self.config.wallet._mock:
            if not self.subtensor.is_hotkey_registered_on_subnet(
                hotkey_ss58=self.wallet.hotkey.ss58_address, netuid=self.config.netuid
            ):
                raise Exception(
                    f"Wallet not currently registered on netuid {self.config.netuid}, please first register wallet before running"
                )
        bt.logging.debug(str(self.wallet))

        # Init subtensor
        bt.logging.debug("loading", "subtensor")
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(str(self.subtensor))

        # Dendrite pool for querying the network during  training.
        bt.logging.debug("loading", "dendrite_pool")
        if self.config.neuron.mock_dendrite_pool:
            # self.dendrite = MockDendrite()
            pass
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.debug(str(self.dendrite))

        # Init metagraph.
        bt.logging.debug("loading", "metagraph")
        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )  # Make sure not to sync without passing subtensor
        self.metagraph.sync(subtensor=self.subtensor)  # Sync metagraph with subtensor.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        bt.logging.debug(str(self.metagraph))

        # Init Weights.
        bt.logging.debug("loading", "moving_averaged_scores")
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.device)
        bt.logging.debug(str(self.moving_averaged_scores))

        # Each validator gets a unique identity (UID) in the network for differentiation.
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

        # Init weights
        self.weights = torch.ones_like(self.metagraph.uids, dtype=torch.float32).to(
            self.device
        )

        # Init bloack and step
        self.prev_block = ttl_get_block(self)
        self.step = 0

        # Init reward function
        self.reward_functions = [ImageRewardModel(), DiversityRewardModel()]

        # Init reward function
        self.reward_weights = torch.tensor(
            [
                self.config.reward.image_model_weight,
                self.config.reward.diversity_model_weight,
                self.config.reward.human_model_weight
                if not self.config.neuron.disable_manual_validator
                else 0.0,
            ],
            dtype=torch.float32,
        ).to(self.device)
        self.reward_weights / self.reward_weights.sum(dim=-1).unsqueeze(-1)

        # Init masking function
        self.masking_functions = [BlacklistFilter(), NSFWRewardModel()]

        #  Init the event loop
        self.loop = asyncio.get_event_loop()

        # Init wandb.
        if not self.config.wandb.off:
            bt.logging.debug("loading", "wandb")
            init_wandb(self)

        # Init manual validator
        if not self.config.neuron.disable_log_rewards:
            bt.logging.debug("loading", "streamlit validator")
            process = subprocess.Popen(
                [
                    "streamlit",
                    "run",
                    os.path.join(os.getcwd(), "neurons", "validator", "app.py"),
                ]
            )

    def run(self):
        # Step 11: The Main Validation Loop
        bt.logging.info("Starting validator loop.")
        step = 0
        while True:
            try:
                # Get a random number of uids
                uids = get_random_uids(
                    self, self.dendrite, k=self.config.neuron.followup_sample_size
                ).to(self.device)
                axons = [self.metagraph.axons[uid] for uid in uids]

                # Text to Image Run
                prompt = generate_random_prompt_gpt(self)
                t2i_event = run_step(
                    self, prompt, axons, uids, task_type="text_to_image"
                )

                # Image to Image Run
                followup_prompt = generate_followup_prompt_gpt(self, prompt)
                # breakpoint()
                followup_image = [image for image in t2i_event["images"]][
                    torch.tensor(t2i_event["rewards"]).argmax()
                ]
                _ = self.run_step(
                    self, followup_prompt, axons, uids, "image_to_image", followup_image
                )

                # Set the weights on chain.
                if should_set_weights(self):
                    set_weights(self)

                # End the current step and prepare for the next iteration.
                self.prev_block = ttl_get_block(self)
                self.step += 1

            # If we encounter an unexpected error, log it for debugging.
            except Exception as err:
                bt.logging.error("Error in training loop", str(err))
                bt.logging.debug(print_exception(type(err), err, err.__traceback__))

            # If the user interrupts the program, gracefully exit.
            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()


def main():
    neuron().run()


if __name__ == "__main__":
    main()
