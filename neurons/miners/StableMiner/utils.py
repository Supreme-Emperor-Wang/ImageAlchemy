import asyncio
import copy
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import regex as re
from google.cloud import storage
from neurons.utils import COLORS, output_log, sh

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


def get_coldkey_for_hotkey(self, synapse):
    """
    Look up the coldkey of the caller.
    """
    if synapse.dendrite.hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return self.metagraph.coldkeys[index]
    return None


def do_logs(self, synapse, local_args):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.stats.start_time

    output_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f} | Model {self.config.miner.model} | Seed {self.config.miner.seed}.",
        color_key="g",
    )
    output_log(
        f"{sh('Request')} -> Type: {synapse.generation_type} | Total requests {self.stats.total_requests:,} | Timeouts {self.stats.timeouts:,}.",
        color_key="y",
    )

    args_list = [
        f"{k.capitalize()}: {f'{v:.2f}' if isinstance(v, float) else v}"
        for k, v in local_args.items()
    ]
    output_log(f"{sh('Args')} -> {' | '.join(args_list)}.", color_key="m")

    miner_info = self.get_miner_info()
    output_log(
        f"{sh('Stats')} -> Block: {miner_info['block']} | Stake: {miner_info['stake']:.2f} | Incentive: {miner_info['incentive']:.2f} | Trust: {miner_info['trust']:.2f} | Consensus: {miner_info['consensus']:.2f}.",
        color_key="c",
    )
    requester_stake = get_caller_stake(self, synapse)
    if requester_stake is None:
        requester_stake = -1
    output_log(
        f"{sh('Caller')} -> Stake {int(requester_stake):,} | Hotkey {synapse.dendrite.hotkey}.",
        color_key="y",
    )
    output_log(f"{sh('Generating')} -> 1 image.", color_key="c")


### mapping["text_to_image"]["args"]


def warm_up(model, local_args):
    """
    Warm the model up if using optimization.
    """
    images = model(**local_args).images
    bt.logging.debug("Warm up is complete...")


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
            bt.logging.debug(f"Removed NSFW word {word.strip()} from prompt...")
    return prompt
