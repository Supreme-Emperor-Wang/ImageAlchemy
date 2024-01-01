import _thread
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from threading import Timer

import torch
from google.cloud import storage
from neurons.constants import (
    IA_BUCKET_NAME,
    IA_MINER_BLACKLIST,
    IA_MINER_WHITELIST,
    IA_VALIDATOR_BLACKLIST,
    IA_VALIDATOR_WEIGHT_FILES,
    IA_VALIDATOR_WHITELIST,
    WANDB_MINER_PATH,
    WANDB_VALIDATOR_PATH,
)

import bittensor as bt


@dataclass
class Stats:
    start_time: datetime
    start_dt: datetime
    total_requests: int
    timeouts: int
    nsfw_count: int


#### Colors to use in the logs
COLORS = {
    "r": "\033[1;31;40m",
    "g": "\033[1;32;40m",
    "b": "\033[1;34;40m",
    "y": "\033[1;33;40m",
    "m": "\033[1;35;40m",
    "c": "\033[1;36;40m",
    "w": "\033[1;37;40m",
}


#### Utility function for coloring logs
def output_log(message: str, color_key: str = "w", type: str = "info") -> None:
    log = bt.logging.info
    if type == "debug":
        log = bt.logging.debug

    if color_key == "na":
        log(f"{message}")
    else:
        log(f"{COLORS[color_key]}{message}{COLORS['w']}")


def sh(message: str):
    return f"{message: <12}"


### Get default stats
def get_defaults(self):
    now = datetime.now()
    stats = Stats(
        start_time=now,
        start_dt=datetime.strftime(now, "%Y/%m/%d %H:%M"),
        total_requests=0,
        nsfw_count=0,
        timeouts=0,
    )
    return stats


#### Background Loop
class BackgroundTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def background_loop(self, is_validator):
    """
    Handles terminating the miner after deregistration and updating the blacklist and whitelist.
    """
    neuron_type = "Validator" if is_validator else "Miner"
    whitelist_type = IA_VALIDATOR_WHITELIST if is_validator else IA_MINER_WHITELIST
    blacklist_type = IA_VALIDATOR_BLACKLIST if is_validator else IA_MINER_BLACKLIST

    #### Terminate the miner / validator after deregistration
    #### Each step is 5 minutes
    if self.background_steps % 1 == 0 and self.background_steps > 1:
        try:
            self.metagraph.sync(lite=True)
            if not self.wallet.hotkey.ss58_address in self.metagraph.hotkeys:
                bt.logging.debug(f">>> {neuron_type} has deregistered... terminating.")
                try:
                    _thread.interrupt_main()
                except Exception as e:
                    bt.logging.error(
                        f"An error occurred trying to terminate the main thread: {e}."
                    )
                try:
                    os._exit(0)
                except Exception as e:
                    bt.logging.error(
                        f"An error occurred trying to use os._exit(): {e}."
                    )
                sys.exit(0)
        except Exception as e:
            bt.logging.error(
                f">>> An unexpected error occurred syncing the metagraph: {e}"
            )

    #### Update the whitelists and blacklists
    if self.background_steps % 1 == 0:
        try:
            if not self.storage_client:
                self.storage_client = storage.Client.create_anonymous_client()
                bt.logging.debug("Created anonymous storage client.")

            blacklist_for_neuron = retrieve_public_file(
                self.storage_client, IA_BUCKET_NAME, blacklist_type
            )
            if blacklist_for_neuron:
                self.hotkey_blacklist = set(
                    [
                        k
                        for k, v in blacklist_for_neuron.items()
                        if v["type"] == "hotkey"
                    ]
                )
                self.coldkey_blacklist = set(
                    [
                        k
                        for k, v in blacklist_for_neuron.items()
                        if v["type"] == "coldkey"
                    ]
                )
                bt.logging.debug("Updated the key blacklists.")

            whitelist_for_neuron = retrieve_public_file(
                self.storage_client, IA_BUCKET_NAME, whitelist_type
            )
            if whitelist_for_neuron:
                self.hotkey_whitelist = set(
                    [
                        k
                        for k, v in whitelist_for_neuron.items()
                        if v["type"] == "hotkey"
                    ]
                )
                self.coldkey_whitelist = set(
                    [
                        k
                        for k, v in whitelist_for_neuron.items()
                        if v["type"] == "coldkey"
                    ]
                )
                bt.logging.debug("Updated the key whitelists.")

            if is_validator:
                validator_weights = retrieve_public_file(
                    self.storage_client, IA_BUCKET_NAME, IA_VALIDATOR_WEIGHT_FILES
                )
                self.reward_weights = torch.tensor(
                    [v for k, v in validator_weights.items() if "manual" not in k],
                    dtype=torch.float32,
                ).to(self.device)
                bt.logging.debug("Updated the validator weights.")
        except Exception as e:
            bt.logging.error(
                f"An error occurred trying to update the blacklists and whitelists: {e}."
            )

    #### Clean up the wandb runs and cache folders
    if self.background_steps == 1 or self.background_steps % 300 == 0:
        wandb_path = WANDB_VALIDATOR_PATH if is_validator else WANDB_MINER_PATH
        try:
            if os.path.exists(wandb_path):
                cleanup_runs_process = subprocess.Popen(
                    [f"echo y | wandb sync --clean {wandb_path}"], shell=True
                )
                bt.logging.debug("Cleaned all synced wandb runs.")
                cleanup_cache_process = subprocess.Popen(
                    ["wandb artifact cache cleanup 5GB"], shell=True
                )
                bt.logging.debug("Cleaned all wandb cache data > 5GB.")
            else:
                bt.logging.debug(f"The path {wandb_path} doesn't exist yet.")
        except Exception as e:
            bt.logging.error(
                f"An error occurred trying to clean wandb artifacts and runs: {e}."
            )

    self.background_steps += 1


def retrieve_public_file(client, bucket_name, source_name):
    file = None
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_name)
        file = blob.download_as_text()

        bt.logging.debug(
            f"Successfully downloaded file: {source_name} of type {type(file)} from {bucket_name}"
        )

        file = json.loads(file)
    except Exception as e:
        bt.logging.error(f"An error occurred downloading from Google Cloud: {e}")

    return file
