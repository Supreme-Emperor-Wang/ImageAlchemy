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
from time import sleep
from traceback import print_exception
from typing import List

# import this repo
import torch
from datasets import load_dataset
from openai import OpenAI
from template.protocol import IsAlive
from template.validator.config import add_args, check_config, config
from template.validator.forward import run_step
from template.validator.reward import (
    BlacklistFilter,
    DiversityRewardModel,
    ImageRewardModel,
    NSFWRewardModel,
)
from template.validator.utils import (
    generate_followup_prompt_gpt,
    generate_random_prompt,
    generate_random_prompt_gpt,
    get_promptdb_backup,
    get_random_uids,
    init_wandb,
    ttl_get_block,
)
from template.validator.weights import set_weights
from transformers import pipeline

import bittensor as bt


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

        # Init prompt backup db
        self.prompt_history_db = get_promptdb_backup()
        self.prompt_generation_failures = 0

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
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
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

        # Init prev_block and step
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

        # Init sync with the network. Updates the metagraph.
        self.sync()

        #  Init the event loop
        self.loop = asyncio.get_event_loop()

        # Init wandb.
        bt.logging.debug("loading", "wandb")
        init_wandb(self)

        # Init manual validator
        if not self.config.neuron.disable_manual_validator:
            bt.logging.debug("loading", "streamlit validator")
            process = subprocess.Popen(
                [
                    "streamlit",
                    "run",
                    os.path.join(os.getcwd(), "template", "validator", "app.py"),
                ]
            )

    def run(self):
        # Main Validation Loop
        bt.logging.info("Starting validator loop.")
        step = 0
        while True:
            try:
                # Reduce calls to miner to be approximately 1 per 5 minutes
                while (ttl_get_block(self) - self.prev_block) < 25:
                    sleep(10)
                    bt.logging.info(
                        "waiting for 5 minutes before queriying miners again"
                    )

                # Get a random number of uids
                uids = get_random_uids(
                    self, self.dendrite, k=self.config.neuron.followup_sample_size
                ).to(self.device)
                axons = [self.metagraph.axons[uid] for uid in uids]

                # Generate prompt + followup_prompt
                prompt = generate_random_prompt_gpt(self)
                followup_prompt = generate_followup_prompt_gpt(self, prompt)
                if (prompt is None) or (followup_prompt is None):
                    if (self.prompt_generation_failures != 0) and (
                        (self.prompt_generation_failures / len(self.prompt_history_db))
                        > 0.2
                    ):
                        self.prompt_history_db = get_promptdb_backup(
                            self.prompt_history_db
                        )
                    prompt, followup_prompt = random.choice(self.prompt_history_db)
                    self.prompt_history_db.remove((prompt, followup_prompt))
                    self.prompt_generation_failures += 1

                # Text to Image Run
                t2i_event = run_step(
                    self, prompt, axons, uids, task_type="text_to_image"
                )
                # Image to Image Run
                followup_image = [image for image in t2i_event["images"]][
                    torch.tensor(t2i_event["rewards"]).argmax()
                ]
                if (
                    (followup_prompt is not None)
                    and (followup_image is not None)
                    and (followup_image != [])
                ):
                    _ = run_step(
                        self,
                        followup_prompt,
                        axons,
                        uids,
                        "image_to_image",
                        followup_image,
                    )
                # Re-sync with the network. Updates the metagraph.
                self.sync()

                # End the current step and prepare for the next iteration.
                self.step += 1

            # If we encounter an unexpected error, log it for debugging.
            except Exception as err:
                bt.logging.error("Error in training loop", str(err))
                bt.logging.debug(print_exception(type(err), err, err.__traceback__))

            # If the user interrupts the program, gracefully exit.
            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            set_weights(self)
            self.prev_block = ttl_get_block(self)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            ttl_get_block(self) - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False
        return (
            ttl_get_block(self) % self.prev_block
        ) >= self.config.neuron.epoch_length


def main():
    neuron().run()


if __name__ == "__main__":
    main()
