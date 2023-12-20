import _thread
import subprocess
from dataclasses import dataclass
from datetime import datetime
from threading import Timer

from neurons.constants import IA_BUCKET_NAME, IA_MINER_BLACKLIST, IA_MINER_WHITELIST

import bittensor as bt


@dataclass
class Stats:
    start_time: datetime
    start_dt: datetime
    total_requests: int
    timeouts: int
    response_times: list

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
        timeouts=0,
        response_times=[],
    )
    return stats

#### Background Loop
class BackgroundTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def background_loop(self):
    """
    Handles terminating the miner after deregistration and updating the blacklist and whitelist.
    """
    bt.logging.debug("Debug Background Timer")
    #### Terminate the miner after deregistration
    #### Each step is 5 minutes
    if self.background_steps % 1 == 0:
        self.metagraph.sync(lite=True)
        if not self.wallet.hotkey.ss58_address in self.metagraph.hotkeys:
            bt.logging.debug(">>> Miner has deregistered... terminating...")
            try:
                _thread.interrupt_main()
            except Exception as e:
                print(f"An error occurred trying to terminate the main thread: {e}")
            try:
                os._exit(0)
            except Exception as e:
                print(f"An error occurred trying to use os._exit(): {e}")
            sys.exit(0)

    #### Update the whitelists and blacklists
    if self.background_steps % 2 == 0:
        if not self.storage_client:
            self.storage_client = storage.Client.create_anonymous_client()
            bt.logging.debug("Created anonymous storage client")

        blacklist_for_miners = retrieve_public_file(
            self.storage_client, IA_BUCKET_NAME, IA_MINER_BLACKLIST
        )
        if blacklist_for_miners and blacklist_for_miners != '{}':
            if type(blacklist_for_miners) == str:
                blacklist_for_miners = eval(blacklist_for_miners)
            self.hotkey_blacklist = set(
                [k for k, v in blacklist_for_miners.items() if v["type"] == "hotkey"]
            )
            self.coldkey_blacklist = set(
                [k for k, v in blacklist_for_miners.items() if v["type"] == "coldkey"]
            )
            bt.logging.debug("Updated the blacklist")

        whitelist_for_miners = retrieve_public_file(
            self.storage_client, IA_BUCKET_NAME, IA_MINER_WHITELIST
        )
        if whitelist_for_miners and whitelist_for_miners != '{}':
            if type(whitelist_for_miners) == str:
                whitelist_for_miners = eval(whitelist_for_miners)
            self.hotkey_whitelist = set(
                [k for k, v in whitelist_for_miners.items() if v["type"] == "hotkey"]
            )
            bt.logging.debug("Updated the hotkey whitelist")

    #### Clean up the wandb runs and cache folders
    if self.background_steps % 300 == 0:
        cleanup_runs_process = subprocess.Popen(
            ["echo y | wandb sync --clean"], shell = True
        )
        bt.logging.debug("Cleaned all synced wanbd runs")
        cleanup_cache_process = subprocess.Popen(
            ["wandb artifact cache cleanup 5GB"], shell = True
        )
        bt.logging.debug("Cleaned all wanbd cache data > 5GB")

    self.background_steps += 1

def retrieve_public_file(client, bucket_name, source_name):
    file = None
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_name)
        file = blob.download_as_text()

        print(
            f"Successfully downloaded file: {source_name} of type {type(file)} from {bucket_name}"
        )
    except Exception as e:
        bt.logging.error(f"An error occurred downloading from Google Cloud: {e}")

    return file