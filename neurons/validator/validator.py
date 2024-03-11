import argparse
import asyncio
import copy
import os
import random
import subprocess
import time
from time import sleep
from traceback import print_exception
from typing import List

import streamlit
import torch
from datasets import load_dataset
from neurons.constants import ENABLE_IMAGE2IMAGE, EPOCH_LENGTH, N_NEURONS
from neurons.utils import BackgroundTimer, background_loop, get_defaults
from neurons.validator.config import add_args, check_config, config
from neurons.validator.forward import run_step
from neurons.validator.reward import (
    BlacklistFilter,
    DiversityRewardModel,
    ImageRewardModel,
    NSFWRewardModel,
)
from neurons.validator.utils import (
    generate_followup_prompt_gpt,
    generate_random_prompt_gpt,
    get_promptdb_backup,
    get_random_uids,
    init_wandb,
    reinit_wandb,
    ttl_get_block,
)
from neurons.validator.weights import set_weights
from openai import OpenAI
from passwordgenerator import pwgenerator
from transformers import pipeline

import bittensor as bt
import wandb


class StableValidator:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)
    
    def loop_until_registered(self):
        index = None
        while True:
            try:
                index = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            except:
                pass
            if index is not None:
                bt.logging.info(
                    f"Validator {self.config.wallet.hotkey} is registered with uid {self.metagraph.uids[index]}.",
                    "g",
                )
                break
            bt.logging.info(
                f"Validator {self.config.wallet.hotkey} is not registered. Sleeping for 120 seconds...",
                "r",
            )
            time.sleep(120)
            self.metagraph.sync(subtensor=self.subtensor)

    def __init__(self):
        # Init config
        self.config = StableValidator.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.alchemy.full_path)

        # Init device.
        self.device = torch.device(self.config.alchemy.device)

        self.openai_client = None

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.corcel_api_key = os.environ.get("CORCEL_API_KEY")

        # if not self.corcel_api_key:
        #     bt.logging.warning("Please set the CORCEL_API_KEY environment variable.")

        if not openai_api_key:
            bt.logging.warning("Please set the OPENAI_API_KEY environment variable.")
        else:
            self.openai_client = OpenAI(api_key=openai_api_key)

        if not self.corcel_api_key and not openai_api_key:
            raise ValueError(
                "You must set either the CORCEL_API_KEY or OPENAI_API_KEY environment variables. It is preferable to use both."
            )

        wandb.login(anonymous="must")

        # Init prompt backup db
        try:
            self.prompt_history_db = get_promptdb_backup(self.config.netuid)
        except Exception as e:
            bt.logging.warning(
                f"Unexpected error occurred loading the backup prompts: {e}"
            )
            self.prompt_history_db = []
        self.prompt_generation_failures = 0

        # Init subtensor
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(f"Loaded subtensor: {self.subtensor}")

        # Init wallet.
        self.wallet = bt.wallet(config=self.config)
        self.wallet.create_if_non_existent()

        # Dendrite pool for querying the network during training.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.debug(f"Loaded dendrite pool: {self.dendrite}")

        # Init metagraph.
        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )  # Make sure not to sync without passing subtensor
        self.metagraph.sync(subtensor=self.subtensor)  # Sync metagraph with subtensor.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        if not self.config.wallet._mock:
            #### Wait until the miner is registered
            self.loop_until_registered()


        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.debug("Loaded metagraph")

        self.scores = torch.zeros_like(self.metagraph.stake, dtype=torch.float32)

        # Init Weights.
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.device)
        bt.logging.debug(
            f"Loaded moving_averaged_scores: {str(self.moving_averaged_scores)}"
        )

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
        self.reward_functions = [ImageRewardModel()]

        # Init manual validator
        if not self.config.alchemy.disable_manual_validator:
            try:
                if 'ImageAlchemy' not in os.getcwd():
                    raise Exception("Unable to load manual validator please cd into the ImageAlchemy folder before running the validator")
                bt.logging.debug("Setting streamlit credentials")
                if not os.path.exists('streamlit_credentials.txt'):
                    username = self.wallet.hotkey.ss58_address
                    password = pwgenerator.generate()
                    with open('streamlit_credentials.txt', 'w') as f: f.write(f"username={username}\npassword={password}")
                    # Sleep until the credentials file is written
                    sleep(5)
                bt.logging.debug("Loading Manual Validator")
                process = subprocess.Popen(
                    [
                        "streamlit",
                        "run",
                        os.path.join(os.getcwd(), "neurons", "validator", "app.py"),
                        "--server.port" if self.config.alchemy.streamlit_port is not None else "", 
                        f"{self.config.alchemy.streamlit_port}" if self.config.alchemy.streamlit_port is not None else ""
                    ]
                )
            except Exception as e:
                bt.logging.error(f"Failed to Load Manual Validator due to error: {e}")
                self.config.alchemy.disable_manual_validator = True

        # Init reward function
        self.reward_weights = torch.tensor(
            [
                1.0,
                1/3 if not self.config.alchemy.disable_manual_validator else 0.0,
            ],
            dtype=torch.float32,
        ).to(self.device)

        self.reward_weights = self.reward_weights / self.reward_weights.sum(dim=-1).unsqueeze(-1)

        self.reward_names = ["image_reward_model", "manual_reward_model"]

        # Init masking function
        self.masking_functions = [BlacklistFilter(), NSFWRewardModel()]

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        self.serve_axon()

        # Init the event loop
        self.loop = asyncio.get_event_loop()

        # Init wandb.
        init_wandb(self)
        bt.logging.debug("Loaded wandb")

        # Init blacklists and whitelists
        self.hotkey_blacklist = set()
        self.coldkey_blacklist = set()
        self.hotkey_whitelist = set()
        self.coldkey_whitelist = set()

        # Init stats
        self.stats = get_defaults(self)

        # Get vali index
        self.validator_index = self.get_validator_index()

        # Set validator request frequency
        self.request_frequency = 120
        self.query_timeout = 20

        # Start the generic background loop
        self.storage_client = None
        self.background_steps = 1
        self.background_timer = BackgroundTimer(300, background_loop, [self, True])
        self.background_timer.daemon = True
        self.background_timer.start()

        # Create a Dict for storing miner query histroy
        self.miner_query_history_duration = {self.metagraph.axons[uid].hotkey:float('inf') for uid in range(self.metagraph.n.item())}
        self.miner_query_history_count = {self.metagraph.axons[uid].hotkey:0 for uid in range(self.metagraph.n.item())}


    async def run(self):
        # Main Validation Loop
        bt.logging.info("Starting validator loop.")
        # Load Previous Sates
        self.load_state()
        self.step = 0
        while True:
            try:
                # Reduce calls to miner to be approximately 1 per 5 minutes
                if self.step > 0:
                    bt.logging.info(
                        f"Waiting for {self.request_frequency} seconds before querying miners again..."
                    )
                    sleep(self.request_frequency)

                # Get a random number of uids
                uids = await get_random_uids(self, self.dendrite, k=N_NEURONS)
                uids = uids.to(self.device)

                axons = [self.metagraph.axons[uid] for uid in uids]

                # Generate prompt + followup_prompt
                prompt = generate_random_prompt_gpt(self)

                if prompt is None:
                    bt.logging.warning(f"The prompt was not generated successfully.")

                    ### Prevent loop from forming if the prompt error occurs on the first step
                    if self.step == 0:
                        self.step += 1

                    continue

                # followup_prompt = generate_followup_prompt_gpt(self, prompt)
                # if prompt is None:  # or (followup_prompt is None):
                #     if (self.prompt_generation_failures != 0) and (
                #         (self.prompt_generation_failures / len(self.prompt_history_db))
                #         > 0.2
                #     ):
                #         try:
                #             self.prompt_history_db = get_promptdb_backup(
                #                 self.config.netuid, self.prompt_history_db
                #             )
                #         except Exception as e:
                #             bt.logging.warning(
                #                 f"Unexpected error occurred loading the backup prompts: {e}"
                #             )
                #             self.prompt_history_db = []
                #             bt.logging.debug("Resetting loop.")
                #             continue

                #     prompt, followup_prompt = random.choice(self.prompt_history_db)
                #     self.prompt_history_db.remove((prompt, followup_prompt))
                #     self.prompt_generation_failures += 1

                # Text to Image Run
                t2i_event = run_step(
                    self, prompt, axons, uids, task_type="text_to_image"
                )
                # if ENABLE_IMAGE2IMAGE:
                #     # Image to Image Run
                #     followup_image = [image for image in t2i_event["images"]][
                #         torch.tensor(t2i_event["rewards"]).argmax()
                #     ]
                #     if (
                #         (followup_prompt is not None)
                #         and (followup_image is not None)
                #         and (followup_image != [])
                #     ):
                #         _ = run_step(
                #             self,
                #             followup_prompt,
                #             axons,
                #             uids,
                #             "image_to_image",
                #             followup_image,
                #         )
                # Re-sync with the network. Updates the metagraph.
                try:
                    self.sync()
                except Exception as e:
                    bt.logging.warning(f"An unexpected error occurred trying to sync the metagraph: {e}")

                # Load Previous Sates
                self.save_state()

                # End the current step and prepare for the next iteration.
                self.step += 1

                # Assuming each step is 3 minutes restart wandb run ever 3 hours to avoid overloading a validators storgage space
                if self.step % 360 == 0 and self.step != 0:
                    bt.logging.info("Re-initializing wandb run...")
                    try:
                        reinit_wandb(self)
                    except Exception as e:
                        bt.logging.error(f"An unexpected error occurred reinitializing wandb: {e}")

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

    def get_validator_index(self):
        """
        Retrieve the given miner's index in the metagraph.
        """
        index = None
        try:
            index = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            pass
        return index

    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.validator_index],
            "rank": self.metagraph.ranks[self.validator_index],
            "vtrust": self.metagraph.validator_trust[self.validator_index],
            "dividends": self.metagraph.dividends[self.validator_index],
            "emissions": self.metagraph.emission[self.validator_index],
        }

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""

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
                f" Please register the hotkey before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            ttl_get_block(self) - self.metagraph.last_update[self.uid]
        ) > EPOCH_LENGTH

    def should_set_weights(self) -> bool:
        # Check if all moving_averages_socres are the 0s or 1s
        ma_scores = self.moving_averaged_scores
        ma_scores_sum = sum(ma_scores)
        if any([ma_scores_sum == len(ma_scores), ma_scores_sum == 0]):
            return False
        else:
            # Check if enough epoch blocks have elapsed since the last epoch.
            return (ttl_get_block(self) % self.prev_block) >= EPOCH_LENGTH
        
    def save_state(self):
        r"""Save hotkeys, neuron model and moving average scores to filesystem."""
        bt.logging.info("save_state()")
        try:
            neuron_state_dict = {
                "neuron_weights": self.moving_averaged_scores.to("cpu").tolist(),
            }
            torch.save(neuron_state_dict, f"{self.config.alchemy.full_path}/model.torch")
            bt.logging.success(
                prefix="Saved model",
                sufix=f"<blue>{ self.config.alchemy.full_path }/model.torch</blue>",
            )
            torch.save(self.miner_query_history_duration, f"{self.config.alchemy.full_path}/history_duration.torch")
            bt.logging.success(
                prefix="Saved model",
                sufix=f"<blue>{ self.config.alchemy.full_path }/history_duration.torch</blue>",
            )
            torch.save(self.miner_query_history_count, f"{self.config.alchemy.full_path}/history_count.torch")
            bt.logging.success(
                prefix="Saved model",
                sufix=f"<blue>{ self.config.alchemy.full_path }/history_count.torch</blue>",
            )
        except Exception as e:
            bt.logging.warning(f"Failed to save model with error: {e}")

        # empty cache
        torch.cuda.empty_cache()

    def load_state(self):
        r"""Load hotkeys and moving average scores from filesystem."""
        bt.logging.info("load_state()")
        try:
            state_dict = torch.load(f"{self.config.alchemy.full_path}/model.torch")
            neuron_weights = torch.tensor(state_dict["neuron_weights"])

            has_nans = torch.isnan(neuron_weights).any()
            has_infs = torch.isinf(neuron_weights).any()

            if has_nans:
                bt.logging.warning(f"Nans found in the model state: {has_nans}")

            if has_infs:
                bt.logging.warning(f"Infs found in the model state: {has_infs}")

            # Check to ensure that the size of the neruon weights matches the metagraph size.
            if neuron_weights.shape != (self.metagraph.n,):
                bt.logging.warning(
                    f"Neuron weights shape {neuron_weights.shape} does not match metagraph n {self.metagraph.n}"
                    "Populating new moving_averaged_scores IDs with zeros"
                )
                self.moving_averaged_scores[: len(neuron_weights)] = neuron_weights.to(
                    self.device
                )
            # Check for nans in saved state dict
            elif not any([has_nans, has_infs]):
                self.moving_averaged_scores = neuron_weights.to(self.device)
                bt.logging.trace(f"MA scores: {self.moving_averaged_scores}")
            else:
                bt.logging.warning("Loaded MA scores from scratch.")

            bt.logging.success(
                prefix="Reloaded model",
                sufix=f"<blue>{ self.config.alchemy.full_path }/model.torch</blue>",
            )
            
            if os.path.isfile(f"{self.config.alchemy.full_path}/history_duration.torch)"):
                # Load saved history duration dict
                breakpoint()
                history_duration_dict = torch.load(f"{self.config.alchemy.full_path}/history_duration.torch")
                for key in self.miner_query_history_duration.keys():
                    self.miner_query_history_duration[key] = history_duration_dict[key]

                bt.logging.success(
                    prefix="Reloaded model",
                    sufix=f"<blue>{ self.config.alchemy.full_path }/history_duration.torch</blue>",
                )

            if os.path.isfile(f"{self.config.alchemy.full_path}/history_count.torch"):
            # Load saved history count dict

                history_count_dict = torch.load(f"{self.config.alchemy.full_path}/history_count.torch")
                for key in self.miner_query_history_count.keys():
                    self.miner_query_history_count[key] = history_count_dict[key]

                bt.logging.success(
                    prefix="Reloaded model",
                    sufix=f"<blue>{ self.config.alchemy.full_path }/history_count.torch</blue>",
                )

        except Exception as e:
            bt.logging.warning(f"Failed to load model with error: {e}")

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(
                wallet=self.wallet,
                ip=bt.utils.networking.get_external_ip(),
                external_ip=bt.utils.networking.get_external_ip(),
                config=self.config
            )

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass
