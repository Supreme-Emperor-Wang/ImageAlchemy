import _thread
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import Timer

import requests
import torch
from google.cloud import storage
from neurons.constants import (
    DEV_URL,
    IA_BUCKET_NAME,
    IA_MINER_BLACKLIST,
    IA_MINER_WARNINGLIST,
    IA_MINER_WHITELIST,
    IA_TEST_BUCKET_NAME,
    IA_VALIDATOR_BLACKLIST,
    IA_VALIDATOR_SETTINGS_FILE,
    IA_VALIDATOR_WEIGHT_FILES,
    IA_VALIDATOR_WHITELIST,
    PROD_URL,
    VALIDATOR_DEFAULT_QUERY_TIMEOUT,
    VALIDATOR_DEFAULT_REQUEST_FREQUENCY,
    WANDB_MINER_PATH,
    WANDB_VALIDATOR_PATH,
)
from neurons.validator.utils import init_wandb

import bittensor as bt


class ModelType(Enum):
   alchemy = "alchemy"
   custom = "custom"
   
@dataclass
class Stats:
    start_time: datetime
    start_dt: datetime
    total_requests: int
    timeouts: int
    nsfw_count: int
    generation_time: int


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
    log = print
    if type == "debug":
        log = print

    if color_key == "na":
        log(f"[{type.upper()}] {message}")
    else:
        log(f"{COLORS[color_key]}[{type.upper()}] {message}{COLORS['w']}")


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
        generation_time=0,
    )
    return stats


#### Background Loop
class BackgroundTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def get_coldkey_for_hotkey(self, hotkey):
    """
    Look up the coldkey of the caller.
    """
    if hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(hotkey)
        return self.metagraph.coldkeys[index]
    return None


def background_loop(self, is_validator):
    """
    Handles terminating the miner after deregistration and updating the blacklist and whitelist.
    """
    neuron_type = "Validator" if is_validator else "Miner"
    whitelist_type = IA_VALIDATOR_WHITELIST if is_validator else IA_MINER_WHITELIST
    blacklist_type = IA_VALIDATOR_BLACKLIST if is_validator else IA_MINER_BLACKLIST
    warninglist_type = IA_MINER_WARNINGLIST

    bucket_name = (
        IA_TEST_BUCKET_NAME if self.subtensor.network == "test" else IA_BUCKET_NAME
    )

    #### Terminate the miner / validator after deregistration
    if self.background_steps % 5 == 0 and self.background_steps > 1:
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            if not self.wallet.hotkey.ss58_address in self.metagraph.hotkeys:
                print(f">>> {neuron_type} has deregistered... terminating.")
                try:
                    _thread.interrupt_main()
                except Exception as e:
                    print(
                        f"An error occurred trying to terminate the main thread: {e}."
                    )
                try:
                    os._exit(0)
                except Exception as e:
                    print(f"An error occurred trying to use os._exit(): {e}.")
                sys.exit(0)
        except Exception as e:
            print(f">>> An unexpected error occurred syncing the metagraph: {e}")

    #### Send new batches to the Human Validation Bot
    try:
        if (self.background_steps % 1 == 0) and is_validator and (self.batches != []):
            print(f"Number of batches in queue: {len(self.batches)}")
            max_retries = 3
            backoff = 2
            batches_for_deletion = []
            invalid_batches = []
            for batch in self.batches:
                for attempt in range(0, max_retries):
                    try:
                        response = requests.post(
                            f"{self.api_url}/batch",
                            data=json.dumps(batch),
                            headers={"Content-Type": "application/json"},
                            timeout=30,
                        )
                        if response.status_code == 200:
                            print(f"Successfully posted batch {batch['batch_id']}")
                            batches_for_deletion.append(batch)
                            break
                        else:
                            response_data = response.json()
                            error = response_data.get("error")
                            if (
                                error
                                and "Submitted compute count must be greater than"
                                in error
                            ):
                                invalid_batches.append(batch)

                            print(f"{response_data=}")
                            raise Exception(
                                f"Failed to post batch. Status code: {response.status_code}"
                            )
                    except Exception as e:
                        if attempt != max_retries:
                            print(
                                f"Attempt number {attempt+1} failed to send batch {batch['batch_id']}. Retrying in {backoff} seconds. Error: {e}"
                            )
                            time.sleep(backoff)
                            continue
                        else:
                            print(
                                f"Attempted to post batch {batch['batch_id']} {attempt+1} times unsuccessfully. Skipping this batch and moving to the next batch. Error: {e}"
                            )
                            break

            ### Delete any invalid batches
            for batch in invalid_batches:
                print(f"Removing invalid batch: {batch['batch_id']}")
                self.batches.remove(batch)

            ### Delete any successful batches
            for batch in batches_for_deletion:
                print(f"Removing successful batch: {batch['batch_id']}")
                self.batches.remove(batch)

    except Exception as e:
        print(f"An error occurred trying to submit a batch: {e}")

    #### Update the whitelists and blacklists
    if self.background_steps % 5 == 0:
        try:
            ### Create client if needed
            if not self.storage_client:
                self.storage_client = storage.Client.create_anonymous_client()
                print("Created anonymous storage client.")

            ### Update the blacklists
            blacklist_for_neuron = retrieve_public_file(
                self.storage_client, bucket_name, blacklist_type
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
                print("Retrieved the latest blacklists.")

            ### Update the whitelists
            whitelist_for_neuron = retrieve_public_file(
                self.storage_client, bucket_name, whitelist_type
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
                print("Retrieved the latest whitelists.")

            ### Update the warning list
            warninglist_for_neuron = retrieve_public_file(
                self.storage_client, bucket_name, warninglist_type
            )
            if warninglist_for_neuron:
                self.hotkey_warninglist = {
                    k: [v["reason"], v["resolve_by"]]
                    for k, v in warninglist_for_neuron.items()
                    if v["type"] == "hotkey"
                }
                self.coldkey_warninglist = {
                    k: [v["reason"], v["resolve_by"]]
                    for k, v in warninglist_for_neuron.items()
                    if v["type"] == "coldkey"
                }
                print("Retrieved the latest warninglists.")
                if self.wallet.hotkey.ss58_address in self.hotkey_warninglist.keys():
                    output_log(
                        f"This hotkey is on the warning list: {self.hotkey_warninglist[self.wallet.hotkey.ss58_address][0]} | Date for rectification: {self.hotkey_warninglist[self.wallet.hotkey.ss58_address][1]}",
                        color_key="r",
                    )
                coldkey = get_coldkey_for_hotkey(self, self.wallet.hotkey.ss58_address)
                if coldkey in self.coldkey_warninglist.keys():
                    output_log(
                        f"This coldkey is on the warning list: {self.coldkey_warninglist[coldkey][0]} | Date for rectification: {self.coldkey_warninglist[coldkey][1]}",
                        color_key="r",
                    )

            ### Validator only
            if is_validator:
                ### Update weights
                validator_weights = retrieve_public_file(
                    self.storage_client, bucket_name, IA_VALIDATOR_WEIGHT_FILES
                )
                if (
                    "manual_reward_model" in validator_weights
                    and self.config.alchemy.disable_manual_validator
                ):
                    validator_weights["manual_reward_model"] = 0.0

                if validator_weights:
                    weights_to_add = []
                    for rw_name in self.reward_names:
                        if rw_name in validator_weights:
                            weights_to_add.append(validator_weights[rw_name])

                    print(f"Raw model weights: {weights_to_add}")

                    if weights_to_add:
                        ### Normalize weights
                        if sum(weights_to_add) != 1:
                            weights_to_add = normalize_weights(weights_to_add)
                            print(f"Normalized model weights: {weights_to_add}")

                        self.reward_weights = torch.tensor(
                            weights_to_add, dtype=torch.float32
                        ).to(self.device)
                        print(
                            f"Retrieved the latest validator weights: {self.reward_weights}"
                        )

                    # self.reward_weights = torch.tensor(
                    #     [v for k, v in validator_weights.items() if "manual" not in k],
                    #     dtype=torch.float32,
                    # ).to(self.device)

                ### Update settings
                validator_settings: dict = retrieve_public_file(
                    self.storage_client, bucket_name, IA_VALIDATOR_SETTINGS_FILE
                )

                if validator_settings:
                    self.request_frequency = validator_settings.get(
                        "request_frequency", self.request_frequency
                    )

                    self.query_timeout = validator_settings.get(
                        "query_timeout", self.query_timeout
                    )

                    self.manual_validator_timeout = validator_settings.get(
                        "manual_validator_timeout", self.manual_validator_timeout
                    )

                    self.async_timeout = validator_settings.get(
                        "async_timeout", self.async_timeout
                    )

                    self.epoch_length = validator_settings.get(
                        "epoch_length", self.epoch_length
                    )

                    if self.config.alchemy.disable_manual_validator:
                        self.request_frequency += self.manual_validator_timeout

                    print(
                        f"Retrieved the latest validator settings: {validator_settings}"
                    )

        except Exception as e:
            print(f"An error occurred trying to update settings from the cloud: {e}.")

    #### Clean up the wandb runs and cache folders
    if self.background_steps == 1 or self.background_steps % 180 == 0:
        print("Trying to clean wandb directoy...")
        wandb_path = WANDB_VALIDATOR_PATH if is_validator else WANDB_MINER_PATH
        try:
            if os.path.exists(wandb_path):
                ### Write a condition to skip this if there are no runs to clean
                # os.path.basename(path).split("run-")[1].split("-")[0], "%Y%m%d_%H%M%S"
                runs = [
                    x
                    for x in os.listdir(f"{wandb_path}/wandb")
                    if "run-" in x and not "latest-run" in x
                ]
                if len(runs) > 0:
                    cleanup_runs_process = subprocess.call(
                        f"cd {wandb_path} && echo 'y' | wandb sync --clean --clean-old-hours 3",
                        shell=True,
                    )
                    print("Cleaned all synced wandb runs.")
                    cleanup_cache_process = subprocess.Popen(
                        ["wandb artifact cache cleanup 5GB"], shell=True
                    )
                    print("Cleaned all wandb cache data > 5GB.")
            else:
                print(f"The path {wandb_path} doesn't exist yet.")
        except Exception as e:
            print(f"An error occurred trying to clean wandb artifacts and runs: {e}.")

    #### Attempt to init wandb if it wasn't sucessfully originally
    if (self.background_steps % 5 == 0) and is_validator and not self.wandb_loaded:
        try:
            init_wandb(self)
            print("Loaded wandb")
            self.wandb_loaded = True
        except Exception as e:
            self.wandb_loaded = False
            print("Unable to load wandb. Retrying in 5 minutes.")

    self.background_steps += 1


def normalize_weights(weights):
    sum_weights = float(sum(weights))
    normalizer = 1 / sum_weights
    weights = [weight * normalizer for weight in weights]
    if sum(weights) < 1:
        diff = 1 - sum(weights)
        weights[0] += diff

    return weights


def retrieve_public_file(client, bucket_name, source_name):
    file = None
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_name)
        try:
            file = blob.download_as_text()
            file = json.loads(file)
            print(f"Successfully downloaded {source_name} from {bucket_name}")
        except Exception as e:
            print(f"Failed to download {source_name} from {bucket_name}: {e}")

    except Exception as e:
        print(f"An error occurred downloading from Google Cloud: {e}")

    return file
