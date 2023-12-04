import copy
import time
from datetime import datetime
from typing import Dict, List

import torch
import torchvision.transforms as transforms

import bittensor as bt
import wandb

transform = transforms.Compose([transforms.PILToTensor()])


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


def do_logs(self, synapse):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.stats.start_time

    if synapse.generation_type == "text_to_image":
        num_images = self.t2i_args["num_images_per_prompt"]
    elif synapse.generation_type == "image_to_image":
        num_images = self.i2i_args["num_images_per_prompt"]
    else:
        bt.logging.debug(
            f"Generation type should be one of either text_to_image or image_to_image."
        )

    output_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f} | Model {self.config.miner.model} | Seed {self.config.miner.seed}."
    )
    output_log(
        f"{sh('Stats')} -> Total requests {self.stats.total_requests} | Timeouts {self.stats.timeouts}."
    )
    requester_stake = get_caller_stake(self, synapse)
    if not requester_stake:
        requester_stake = -1
    output_log(
        f"{sh('Caller')} -> Stake {int(requester_stake):,} | Hotkey {synapse.dendrite.hotkey}"
    )
    output_log(f"{sh('Generating')} -> {num_images} images.")


def generate(self, synapse, timeout=10):
    """
    Image generation logic shared between both text-to-image and image-to-image
    """
    # Increment total requests state by 1

    self.stats.total_requests += 1
    do_logs(self, synapse)
    start_time = time.perf_counter()
    # breakpoint()
    # Set up arguments
    if synapse.generation_type == "text_to_image":
        local_args = copy.copy(self.t2i_args)
        model = self.t2i_model
    elif synapse.generation_type == "image_to_image":
        local_args = copy.copy(self.i2i_args)
        local_args["image"] = bt.Tensor.deserialize(synapse.prompt_image)
        model = self.i2i_model
    else:
        bt.logging.debug(
            f"Generation type should be one of either text_to_image or image_to_image."
        )

    local_args["prompt"] = synapse.prompt
    local_args["target_size"] = (synapse.height, synapse.width)
    images = model(**local_args).images

    if (time.perf_counter() - start_time) > timeout:
        self.stats.total_requests += 1

    synapse.images = [bt.Tensor.serialize(transform(image)) for image in images]
    self.event.update(
        {
            "images": [
                wandb.Image(bt.Tensor.deserialize(image), caption=synapse.prompt)
                if image != []
                else wandb.Image(
                    torch.full([3, 1024, 1024], 255, dtype=torch.float),
                    caption=synapse.prompt,
                )
                for image in synapse.images
            ],
        }
    )
    #### Log to Wanbd
    self.wandb._log()
    # breakpoint()
    #### Log to console
    output_log(f"{sh('Time')} -> {time.perf_counter() - start_time:.2f}s.")
