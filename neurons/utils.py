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
    IA_MINER_WARNINGLIST,
    IA_MINER_WHITELIST,
    IA_TEST_BUCKET_NAME,
    IA_VALIDATOR_BLACKLIST,
    IA_VALIDATOR_SETTINGS_FILE,
    IA_VALIDATOR_WEIGHT_FILES,
    IA_VALIDATOR_WHITELIST,
    MANUAL_VALIDATOR_TIMEOUT,
    VALIDATOR_DEFAULT_QUERY_TIMEOUT,
    VALIDATOR_DEFAULT_REQUEST_FREQUENCY,
    WANDB_MINER_PATH,
    WANDB_VALIDATOR_PATH,
)
from neurons.validator.utils import init_wandb

import bittensor as bt


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

    bucket_name = IA_TEST_BUCKET_NAME if self.subtensor.network == "test" else IA_BUCKET_NAME

    #### Terminate the miner / validator after deregistration
    if self.background_steps % 1 == 0 and self.background_steps > 1:
        try:
            # if is_validator:
            self.metagraph.sync(subtensor=self.subtensor)
            # else:
                # self.metagraph.sync()
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
            ### Create client if needed
            if not self.storage_client:
                self.storage_client = storage.Client.create_anonymous_client()
                bt.logging.info("Created anonymous storage client.")

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
                bt.logging.info("Retrieved the latest blacklists.")

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
                bt.logging.info("Retrieved the latest whitelists.") 

            ### Update the warning list
            warninglist_for_neuron = retrieve_public_file(
                self.storage_client, bucket_name, warninglist_type
            )
            if warninglist_for_neuron:
                self.hotkey_warninglist =   {
                        k :[v['reason'],v['resolve_by']]
                        for k, v in warninglist_for_neuron.items()
                        if v["type"] == "hotkey"
                }
                self.coldkey_warninglist = {
                        k :[v['reason'],v['resolve_by']]
                        for k, v in warninglist_for_neuron.items()
                        if v["type"] == "coldkey"
                }
                bt.logging.info("Retrieved the latest warninglists.")
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
                if "manual_reward_model" in validator_weights and self.config.alchemy.disable_manual_validator:
                    validator_weights["manual_reward_model"] = 0.0

                if validator_weights:
                    weights_to_add = []
                    for rw_name in self.reward_names:
                        if rw_name in validator_weights:
                            weights_to_add.append(validator_weights[rw_name])


                    bt.logging.trace(f"Raw model weights: {weights_to_add}")

                    if weights_to_add:
                        ### Normalize weights
                        if sum(weights_to_add) != 1:
                            weights_to_add = normalize_weights(weights_to_add)
                            bt.logging.trace(f"Normalized model weights: {weights_to_add}")

                        self.reward_weights = torch.tensor(weights_to_add, dtype=torch.float32).to(self.device)
                        bt.logging.info(
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
                        "request_frequency", VALIDATOR_DEFAULT_REQUEST_FREQUENCY
                    )
                    if self.config.alchemy.disable_manual_validator:
                        self.request_frequency += MANUAL_VALIDATOR_TIMEOUT
                    self.query_timeout = validator_settings.get(
                        "query_timeout", VALIDATOR_DEFAULT_QUERY_TIMEOUT
                    )

                    bt.logging.info(
                        f"Retrieved the latest validator settings: {validator_settings}"
                    )
        

        except Exception as e:
            bt.logging.error(
                f"An error occurred trying to update settings from the cloud: {e}."
            )

    #### Clean up the wandb runs and cache folders
    if self.background_steps == 1 or self.background_steps % 36 == 0:
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
                    cleanup_runs_process = subprocess.call(f"cd {wandb_path} && echo 'y' | wandb sync --clean --clean-old-hours 3", shell=True)
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

    # Attempt to init wandb if it wasn't sucessfully originally
    if (self.background_steps % 1 == 0) and (self.wandb_loaded == False):
        try:
            init_wandb(self)
            bt.logging.debug("Loaded wandb")
        except Exception as e:
            self.wandb_loaded = False
            bt.logging.debug("Unable to load wandb. Retrying in 5 minnutes.")

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
            bt.logging.debug(f"Successfully downloaded {source_name} from {bucket_name}")
        except Exception as e:
            bt.logging.warning(f"Failed to download {source_name} from {bucket_name}: {e}")


    except Exception as e:
        bt.logging.error(f"An error occurred downloading from Google Cloud: {e}")

    return file
