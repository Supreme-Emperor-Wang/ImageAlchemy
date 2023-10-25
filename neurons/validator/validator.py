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

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import traceback
import bittensor as bt
import random

# import this repo
import template

from utils import check_uid_availability
from typing import List
import asyncio
from reward  import BlacklistFilter, NSFWRewardModel, ImageRewardModel, DiversityRewardModel

# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha", default=0.9, type=float, help="The weight moving average scoring."
    )
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/validator.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config

def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = torch.tensor(random.sample(available_uids, k))
    return uids

class neuron:
    def __init__(self):
        # Set up logging with the provided configuration and directory.
        self.config = get_config()
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint} with config:"
        )
        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Step 4: Build Bittensor validator objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the validator.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # The subtensor is our connection to the Bittensor blockchain.
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Step 5: Connect the validator to the network
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again."
            )
            exit()

        # Each validator gets a unique identity (UID) in the network for differentiation.
        self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

        # Step 6: Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = torch.ones_like(self.metagraph.S, dtype=torch.float32)
        bt.logging.info(f"Weights: {self.scores}")
        
        # Step 7: Set up reward functions
        self.reward_functions = [
            ImageRewardModel(),
            # DiversityRewardModel()
        ]
        
        # Step 8: Set up masking functions
        self.masking_functions = [
            BlacklistFilter(),
            NSFWRewardModel()
        ]

        # Step 9: Set device
        self.device = "cuda"

        # Step 10: Initialise weights
        self.weights = torch.ones_like(self.metagraph.uids , dtype = torch.float32 )

        # multiply weights by the active tensor
        self.current_block = self.subtensor.block
        self.last_updated_block = self.subtensor.block

        # loop over all last_update, any that are within 600 blocks are set to 1 others are set to 0 
        self.weights = self.weights * self.metagraph.last_update > self.current_block - 600

        # all nodes with more than 1e3 total stake are set to 0 (sets validtors weights to 0)
        self.weights = self.weights * (self.metagraph.total_stake < 1.024e3) 

        # set all nodes without ips set to 0
        self.weights = self.weights * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in self.metagraph.uids]) * 0.5

        # move weights to device
        self.weights = self.weights.to(self.device)

        # Step 11: Initialize loop
        self.loop = asyncio.get_event_loop()

    def run(self):
        # Step 11: The Main Validation Loop
        bt.logging.info("Starting validator loop.")
        step = 0
        while True:
            try:
                # k = 12
                # uids = get_random_uids(self, k=k).to(self.device)
                device = "cuda"
                timeout = 100
                
                # Call the dentrite 
                responses = self.loop.run_until_complete( self.dendrite(self.metagraph.axons, template.protocol.Dummy(dummy_input=step), timeout = timeout))
                
                # Log the results for monitoring purposes.
                bt.logging.info(f"Received response: {responses}")

                # Initialise rewards tensor
                rewards: torch.FloatTensor = torch.ones(len(responses), dtype=torch.float32).to(
                    device
                )
                for masking_fn_i in self.masking_functions:
                    mask_i, mask_i_normalized = masking_fn_i.apply(responses, )
                    rewards *= mask_i_normalized.to(device)
                    # TODO add wandb tracking
                    # if not self.config.neuron.disable_log_rewards:
                    #     event[masking_fn_i.name] = mask_i.tolist()
                    #     event[masking_fn_i.name + "_normalized"] = mask_i_normalized.tolist()
                    bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

                for weight_i, reward_fn_i in zip([0.95], self.reward_functions):
                    reward_i, reward_i_normalized = reward_fn_i.apply(responses)
                    rewards += weight_i * reward_i_normalized.to(device)
                    # TODO add wandb tracking
                    # if not self.config.neuron.disable_log_rewards:
                    #     event[reward_fn_i.name] = reward_i.tolist()
                    #     event[reward_fn_i.name + "_normalized"] = reward_i_normalized.tolist()
                    bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())
                
                bt.logging.info(f"Rewards: {rewards}")

                self.weights = self.weights + (self.config.alpha * rewards)

                self.current_block = self.subtensor.block
                if self.current_block - self.last_updated_block  >= 100:
                    bt.logging.trace(f"Setting weights")

                    # Normalize weights.
                    self.weights = self.weights / torch.sum( self.weights )
                    bt.logging.trace("Weights:")
                    bt.logging.trace(self.weights)

                    uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
                        uids = self.metagraph.uids.to("cpu"),
                        weights = self.weights.to("cpu"),
                        netuid = self.config.netuid,
                        subtensor = self.subtensor,
                    )
                    result = self.subtensor.set_weights(
                        wallet = self.wallet,
                        netuid = self.config.netuid,
                        weights = processed_weights,
                        uids = uids,
                    )
                    self.last_updated_block = self.current_block

                    if result:
                        bt.logging.success("Successfully set weights.")
                    else:
                        bt.logging.error("Failed to set weights.")

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