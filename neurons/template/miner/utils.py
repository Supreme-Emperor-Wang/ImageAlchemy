import asyncio
import copy
import os
import sys
from threading import Timer
import _thread
import time
import traceback
from datetime import datetime
from typing import Dict, List
from google.cloud import storage

from template.miner.constants import (
    IA_BUCKET_NAME,
    IA_MINER_BLACKLIST,
    IA_MINER_WHITELIST,
)

import torchvision.transforms as transforms
import torchvision.transforms as T

import bittensor as bt


#### Wrapper for the raw images
class Images:
    def __init__(self, images):
        self.images = images


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

NSFW_WORDS = [
    "hentai",
    "loli",
    "lolita",
    "naked",
    "undress",
    "undressed",
    "nude",
    "sexy",
    "sex",
    "porn",
    "orgasm",
    "cum",
    "cumming",
    "penis",
    "cock",
    "dick",
    "vagina",
    "pussy",
    "anus",
    "ass",
    "asshole",
    "tits",
]

transform = transforms.Compose([transforms.PILToTensor()])


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


def get_caller_stake(self, synapse):
    """
    Look up the stake of the requesting validator.
    """
    if synapse.dendrite.hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return self.metagraph.S[index].item()
    return None


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

        if blacklist_for_miners:
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

        if whitelist_for_miners:
            self.hotkey_whitelist = set(
                [k for k, v in whitelist_for_miners.items() if v["type"] == "hotkey"]
            )
            bt.logging.debug("Updated the hotkey whitelist")

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


def do_logs(self, synapse, local_args):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.stats.start_time
    num_images = 1

    output_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f} | Model {self.config.miner.model} | Seed {self.config.miner.seed}.",
        color_key="g",
    )
    output_log(
        f"{sh('Request')} -> Type: {synapse.generation_type} | Total requests {self.stats.total_requests:,} | Timeouts {self.stats.timeouts:,}",
        color_key="y",
    )
    args_list = [
        f"{k.capitalize()}: {f'{v:.2f}' if isinstance(v, float) else v}"
        for k, v in local_args.items()
    ]
    output_log(f"{sh('Args')} -> {' | '.join(args_list)}", color_key="m")

    miner_info = self.get_miner_info()
    output_log(
        f"{sh('Stats')} -> Block: {miner_info['block']} | Stake: {miner_info['stake']:.2f} | Incentive: {miner_info['incentive']:.2f} | Trust: {miner_info['trust']:.2f} | Consensus: {miner_info['consensus']:.2f}",
        color_key="c",
    )
    requester_stake = get_caller_stake(self, synapse)
    if not requester_stake:
        requester_stake = -1
    output_log(
        f"{sh('Caller')} -> Stake {int(requester_stake):,} | Hotkey {synapse.dendrite.hotkey}",
        color_key="y",
    )
    output_log(f"{sh('Generating')} -> {num_images} image(s)", color_key="c")


### mapping["text_to_image"]["args"]


def warm_up(model, local_args):
    """
    Warm the model up if using optimization.
    """
    images = model(**local_args).images
    bt.logging.debug("Warm up is complete...")


async def generate(self, synapse, timeout=10):
    """
    Image generation logic shared between both text-to-image and image-to-image
    """
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
            synapse.images = [bt.Tensor.serialize(transform(image)) for image in images]
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


def nsfw_image_filter(self, images):
    clip_input = self.processor(
        [transform(image) for image in images], return_tensors="pt"
    ).to(self.config.miner.device)
    images, nsfw = self.safety_checker.forward(
        images=images, clip_input=clip_input.pixel_values.to(self.config.miner.device)
    )

    return nsfw


def clean_nsfw_from_prompt(prompt):
    for word in NSFW_WORDS:
        if word in prompt:
            prompt = prompt.replace(f" {word}", "")
            bt.logging.debug(f"Removed NSFW word {word.strip()} from prompt...")
    return prompt
