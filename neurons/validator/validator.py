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
from typing import List

# import this repo
import template
import torch
import torchvision.transforms as T
from datasets import load_dataset
from event import EventSchema
from loguru import logger
from reward import (
    BlacklistFilter,
    DiversityRewardModel,
    ImageRewardModel,
    NSFWRewardModel,
)
from template.protocol import IsAlive
from transformers import pipeline
from utils import generate_random_prompt, generate_random_prompt_gpt, get_random_uids, init_wandb, ttl_get_block, generate_followup_prompt_gpt

import bittensor as bt
import wandb
from config import add_args, check_config, config
from openai import OpenAI

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
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
                raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.openai_client = OpenAI(api_key = openai_api_key)

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

        # Each validator gets a unique identity (UID) in the network for differentiation.
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

        # Init weights
        self.weights = torch.ones_like(self.metagraph.uids, dtype=torch.float32).to(
            self.device
        )

        # Set current and last updated blocks
        self.current_block = self.subtensor.block
        self.last_updated_block = self.subtensor.block

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
                timeout = 100

                # Get a random number of uids
                uids = get_random_uids(
                    self, self.dendrite, k=self.config.neuron.followup_sample_size
                ).to(self.device)
                axons = [self.metagraph.axons[uid] for uid in uids]
                prompt = generate_random_prompt_gpt(self)

                responses = self.loop.run_until_complete(
                    self.dendrite(
                        axons,
                        template.protocol.ImageGeneration(
                            generation_type="text_to_image", prompt=prompt
                        ),
                        timeout=timeout,
                    )
                )
                event = {"task_type": "text_to_image"}

                start_time = time.time()

                # Log the results for monitoring purposes.
                bt.logging.info(f"Received response: {responses}")

                # Save images
                bt.logging.info(f"Saving images")
                i = 0
                for r in responses:
                    for image in r.images:
                        T.transforms.ToPILImage()(bt.Tensor.deserialize(image)).save(
                            f"neurons/validator/images/{i}.png"
                        )
                        i = i + 1

                bt.logging.info(f"Saving prompt")
                with open("neurons/validator/images/prompt.txt", "w") as f:
                    f.write(prompt)

                # Initialise rewards tensor
                rewards: torch.FloatTensor = torch.ones(
                    len(responses), dtype=torch.float32
                ).to(self.device)
                for masking_fn_i in self.masking_functions:
                    mask_i, mask_i_normalized = masking_fn_i.apply(responses, rewards)
                    rewards *= mask_i_normalized.to(self.device)
                    if not self.config.neuron.disable_log_rewards:
                        event[masking_fn_i.name] = mask_i.tolist()
                        event[
                            masking_fn_i.name + "_normalized"
                        ] = mask_i_normalized.tolist()
                    bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

                for weight_i, reward_fn_i in zip(
                    self.reward_weights, self.reward_functions
                ):
                    reward_i, reward_i_normalized = reward_fn_i.apply(
                        responses, rewards
                    )
                    rewards += weight_i * reward_i_normalized.to(self.device)
                    if not self.config.neuron.disable_log_rewards:
                        event[reward_fn_i.name] = reward_i.tolist()
                        event[
                            reward_fn_i.name + "_normalized"
                        ] = reward_i_normalized.tolist()
                    bt.logging.trace(
                        str(reward_fn_i.name), reward_i_normalized.tolist()
                    )

                if not self.config.neuron.disable_manual_validator:
                    bt.logging.info(f"Waiting for manual vote")
                    start_time = time.perf_counter()

                    while (time.perf_counter() - start_time) < 10:
                        if os.path.exists("neurons/validator/images/vote.txt"):
                            # loop until vote is successfully saved
                            while (
                                open("neurons/validator/images/vote.txt", "r").read()
                                == ""
                            ):
                                continue

                            reward_i = open(
                                "neurons/validator/images/vote.txt", "r"
                            ).read()
                            bt.logging.info("Received manual vote")
                            bt.logging.info("MANUAL VOTE = " + reward_i)
                            reward_i_normalized: torch.FloatTensor = torch.zeros(
                                len(rewards), dtype=torch.float32
                            ).to(self.device)
                            reward_i_normalized[int(reward_i) - 1] = 1.0

                            rewards += self.reward_weights[-1] * reward_i_normalized.to(
                                self.device
                            )

                            if not self.config.neuron.disable_log_rewards:
                                event[
                                    "human_reward_model"
                                ] = reward_i_normalized.tolist()
                                event[
                                    "human_reward_model_normalized"
                                ] = reward_i_normalized.tolist()

                            break
                    else:
                        bt.logging.info("No manual vote received")

                    # Delete contents of images folder except for black image
                    for file in os.listdir("neurons/validator/images"):
                        os.remove(
                            f"neurons/validator/images/{file}"
                        ) if file != "black.png" else "_"

                # TODO Image to Image
                followup_prompt = generate_random_prompt_gpt(self)
                followup_image = [response.images for response in responses][int(rewards.argmax())][0]
                responses = self.loop.run_until_complete(
                    self.dendrite(
                        axons,
                        template.protocol.ImageGeneration(
                            generation_type="image_to_image", prompt=prompt, prompt_image=followup_image
                        ),
                        timeout=timeout,
                    )
                )
                event = {"task_type": "image_to_image"}

                start_time = time.time()

                # Log the results for monitoring purposes.
                bt.logging.info(f"Received response: {responses}")

                # Save images
                bt.logging.info(f"Saving images")
                i = 0
                for r in responses:
                    for image in r.images:
                        T.transforms.ToPILImage()(bt.Tensor.deserialize(image)).save(
                            f"neurons/validator/images/{i}.png"
                        )
                        i = i + 1

                bt.logging.info(f"Saving prompt")
                with open("neurons/validator/images/prompt.txt", "w") as f:
                    f.write(prompt)

                # Initialise rewards tensor
                rewards: torch.FloatTensor = torch.ones(
                    len(responses), dtype=torch.float32
                ).to(self.device)
                for masking_fn_i in self.masking_functions:
                    mask_i, mask_i_normalized = masking_fn_i.apply(responses, rewards)
                    rewards *= mask_i_normalized.to(self.device)
                    if not self.config.neuron.disable_log_rewards:
                        event[masking_fn_i.name] = mask_i.tolist()
                        event[
                            masking_fn_i.name + "_normalized"
                        ] = mask_i_normalized.tolist()
                    bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

                for weight_i, reward_fn_i in zip(
                    self.reward_weights, self.reward_functions
                ):
                    reward_i, reward_i_normalized = reward_fn_i.apply(
                        responses, rewards
                    )
                    rewards += weight_i * reward_i_normalized.to(self.device)
                    if not self.config.neuron.disable_log_rewards:
                        event[reward_fn_i.name] = reward_i.tolist()
                        event[
                            reward_fn_i.name + "_normalized"
                        ] = reward_i_normalized.tolist()
                    bt.logging.trace(
                        str(reward_fn_i.name), reward_i_normalized.tolist()
                    )

                if not self.config.neuron.disable_manual_validator:
                    bt.logging.info(f"Waiting for manual vote")
                    start_time = time.perf_counter()

                    while (time.perf_counter() - start_time) < 10:
                        if os.path.exists("neurons/validator/images/vote.txt"):
                            # loop until vote is successfully saved
                            while (
                                open("neurons/validator/images/vote.txt", "r").read()
                                == ""
                            ):
                                continue

                            reward_i = open(
                                "neurons/validator/images/vote.txt", "r"
                            ).read()
                            bt.logging.info("Received manual vote")
                            bt.logging.info("MANUAL VOTE = " + reward_i)
                            reward_i_normalized: torch.FloatTensor = torch.zeros(
                                len(rewards), dtype=torch.float32
                            ).to(self.device)
                            reward_i_normalized[int(reward_i) - 1] = 1.0

                            rewards += self.reward_weights[-1] * reward_i_normalized.to(
                                self.device
                            )

                            if not self.config.neuron.disable_log_rewards:
                                event[
                                    "human_reward_model"
                                ] = reward_i_normalized.tolist()
                                event[
                                    "human_reward_model_normalized"
                                ] = reward_i_normalized.tolist()

                            break
                    else:
                        bt.logging.info("No manual vote received")

                    # Delete contents of images folder except for black image
                    for file in os.listdir("neurons/validator/images"):
                        os.remove(
                            f"neurons/validator/images/{file}"
                        ) if file != "black.png" else "_"

                # TODO Add Moving Average Score
                for i in range(len(rewards)):
                    self.weights[uids[i]] = self.weights[uids[i]] + (
                        self.config.neuron.alpha * rewards[i]
                    )
                self.weights = torch.nn.functional.normalize(self.weights, p=1.0, dim=0)
                # Normalize weights.
                bt.logging.trace("Weights:")
                bt.logging.trace(self.weights)

                self.current_block = self.subtensor.block
                if self.current_block - self.last_updated_block >= 100:
                    bt.logging.trace(f"Setting weights")
                    (
                        uids,
                        processed_weights,
                    ) = bt.utils.weight_utils.process_weights_for_netuid(
                        uids=self.metagraph.uids.to("cpu"),
                        weights=self.weights.to("cpu"),
                        netuid=self.config.netuid,
                        subtensor=self.subtensor,
                    )
                    result = self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.config.netuid,
                        weights=processed_weights,
                        uids=uids,
                    )
                    self.last_updated_block = self.current_block

                    if result:
                        bt.logging.success("Successfully set weights.")
                    else:
                        bt.logging.error("Failed to set weights.")

                try:
                    # Log the step event.
                    event.update(
                        {
                            "block": ttl_get_block(self),
                            "step_length": time.time() - start_time,
                            "prompt": prompt,
                            "uids": uids.tolist(),
                            "hotkeys": [
                                self.metagraph.axons[uid].hotkey for uid in uids
                            ],
                            "images": [
                                wandb.Image(
                                    bt.Tensor.deserialize(r.images[0])[0],
                                    caption=prompt,
                                )
                                if r.images != []
                                else wandb.Image(
                                    torch.full([3, 1024, 1024], 255, dtype=torch.float),
                                    caption=prompt,
                                )
                                for r in responses
                            ],
                            "rewards": rewards.tolist(),
                        }
                    )
                except:
                    breakpoint()

                bt.logging.debug("event:", str(event))
                if not self.config.neuron.dont_save_events:
                    logger.log("EVENTS", "events", **event)

                # Log the event to wandb.
                if not self.config.wandb.off:
                    wandb_event = EventSchema.from_dict(
                        event, self.config.neuron.disable_log_rewards
                    )
                    self.wandb.log(asdict(wandb_event))

                # End the current step and prepare for the next iteration.
                step += 1
                # Resync our local state with the latest state from the blockchain.
                self.metagraph = self.subtensor.metagraph(self.config.netuid)
                # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
                time.sleep(bt.__blocktime__)

            # If we encounter an unexpected error, log it for debugging.
            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            # If the user interrupts the program, gracefully exit.
            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()


def main():
    neuron().run()


if __name__ == "__main__":
    main()
