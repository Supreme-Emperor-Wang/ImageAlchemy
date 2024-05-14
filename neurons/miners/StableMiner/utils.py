import asyncio
import copy
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import regex as re
from google.cloud import storage
from loguru import logger

from neurons.utils import COLORS, colored_log, sh

import bittensor as bt


#### Wrapper for the raw images
class Images:
    def __init__(self, images):
        self.images = images


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


def get_caller_stake(self, synapse):
    """
    Look up the stake of the requesting validator.
    """
    if synapse.dendrite.hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return self.metagraph.S[index].item()
    return None


def get_coldkey_for_hotkey(self, hotkey):
    """
    Look up the coldkey of the caller.
    """
    if hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(hotkey)
        return self.metagraph.coldkeys[index]
    return None


def do_logs(self, synapse, local_args):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.stats.start_time
    hotkey = synapse.dendrite.hotkey

    colored_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f} | Model {self.config.miner.model} | Default seed {self.config.miner.seed}.",
        color="green",
    )
    colored_log(
        f"{sh('Request')} -> Type: {synapse.generation_type} | Request seed: {synapse.seed} | Total requests {self.stats.total_requests:,} | Timeouts {self.stats.timeouts:,}.",
        color="yellow",
    )

    args_list = [
        f"{k.capitalize()}: {f'{v:.2f}' if isinstance(v, float) else v}"
        for k, v in local_args.items()
    ]
    colored_log(f"{sh('Args')} -> {' | '.join(args_list)}.", color="magenta")

    miner_info = self.get_miner_info()
    colored_log(
        f"{sh('Stats')} -> Block: {miner_info['block']} | Stake: {miner_info['stake']:.4f} | Incentive: {miner_info['incentive']:.4f} | Trust: {miner_info['trust']:.4f} | Consensus: {miner_info['consensus']:.4f}.",
        color="cyan",
    )

    ### Output stake
    requester_stake = get_caller_stake(self, synapse)
    if requester_stake is None:
        requester_stake = -1

    ### Retrieve the coldkey of the caller
    caller_coldkey = get_coldkey_for_hotkey(self, hotkey)

    temp_string = f"Stake {int(requester_stake):,}"

    if hotkey in self.hotkey_whitelist or caller_coldkey in self.coldkey_whitelist:
        temp_string = "Whitelisted key"

    colored_log(
        f"{sh('Caller')} -> {temp_string} | Hotkey {hotkey}.",
        color="yellow",
    )


def warm_up(model, local_args):
    """
    Warm the model up if using optimization.
    """
    start = time.perf_counter()
    c_args = copy.deepcopy(local_args)
    c_args["prompt"] = "An alchemist brewing a vibrant glowing potion."
    images = model(**c_args).images
    logger.info(f"Warm up is complete after {time.perf_counter() - start}")


def nsfw_image_filter(self, images):
    clip_input = self.processor(
        [self.transform(image) for image in images], return_tensors="pt"
    ).to(self.config.miner.device)
    images, nsfw = self.safety_checker.forward(
        images=images, clip_input=clip_input.pixel_values.to(self.config.miner.device)
    )

    return nsfw


def clean_nsfw_from_prompt(prompt):
    for word in NSFW_WORDS:
        if re.search(r"\b{}\b".format(word), prompt):
            prompt = re.sub(r"\b{}\b".format(word), "", prompt).strip()
            logger.warning(f"Removed NSFW word {word.strip()} from prompt...")
    return prompt
