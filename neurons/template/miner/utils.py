import copy
import time
from datetime import datetime
from typing import Dict, List

import torch
import torchvision.transforms as transforms
import torchvision.transforms as T

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

WHITELISTED_HOTKEYS = ["5C5PXHeYLV5fAx31HkosfCkv8ark3QjbABbjEusiD3HXH2Ta"]


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


def do_logs(self, synapse, local_args):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.stats.start_time
    num_images = local_args["num_images_per_prompt"]

    output_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f} | Model {self.config.miner.model} | Seed {self.config.miner.seed}.",
        color_key="g",
    )
    output_log(
        f"{sh('Stats')} -> Type: {synapse.generation_type} | Total requests {self.stats.total_requests} | Timeouts {self.stats.timeouts}.",
        color_key="y",
    )
    requester_stake = get_caller_stake(self, synapse)
    if not requester_stake:
        requester_stake = -1
    output_log(
        f"{sh('Caller')} -> Stake {int(requester_stake):,} | Hotkey {synapse.dendrite.hotkey}",
        color_key="c",
    )
    output_log(f"{sh('Generating')} -> {num_images} image(s)")


### mapping["text_to_image"]["args"]


def generate(self, synapse, timeout=10):
    """
    Image generation logic shared between both text-to-image and image-to-image
    """
    self.stats.total_requests += 1
    start_time = time.perf_counter()

    ### Set up args
    local_args = copy.copy(self.mapping[synapse.generation_type]["args"])
    if synapse.generation_type == "image_to_image":
        local_args["image"] = T.transforms.ToPILImage()(
            bt.Tensor.deserialize(synapse.prompt_image)
        )
        del local_args["num_inference_steps"]
        T.transforms.ToPILImage()(bt.Tensor.deserialize(synapse.prompt_image)).save(
            "test.png"
        )
    bt.logging.info(synapse.generation_type)
    local_args["prompt"] = [synapse.prompt]
    local_args["target_size"] = (synapse.height, synapse.width)

    ### Output logs
    do_logs(self, synapse, local_args)

    ### Get the model
    model = self.mapping[synapse.generation_type]["model"]

    ### Generate images
    # breakpoint()
    # local_args_2 = {'image':init_image, 'guidance_scale': 7.5, 'num_inference_steps': 1, 'num_images_per_prompt': 1, 'prompt': 'Add a gentle stream flowing near the base of the lone tree, reflecting the surrounding flowers.', 'target_size': (1024, 1024)}
    # local_args_2 = {'image':local_args_2["image"], 'prompt': local_args_2["prompt"], 'target_size': (1024, 1024), 'guidance_scale': 7.5, 'num_inference_steps': 2}
    # model(**local_args_2)
    # model.num_inference_steps
    # local_args_2["prompt"]
    # model(**local_args_2)
    # pipe(**local_args_2)
    # 'generator': <torch._C.Generator object at 0x7ff7f1716c70>, 'image': <PIL.Image.Image image mode=RGB size=1024x1024 at 0x7FF7B45CE150>,
    images = model(**local_args).images

    if time.perf_counter() - start_time > timeout:
        self.stats.timeouts += 1

    ### Seralize the images
    synapse.images = [bt.Tensor.serialize(transform(image)) for image in images]
    images[0].save("test.png")

    if self.wandb:
        ### Store the images and prompts for uploading to wandb
        self.wandb._add_images(synapse)

        #### Log to Wandb
        self.wandb._log()

    #### Log to console
    output_log(
        f"{sh('Time')} -> {time.perf_counter() - start_time:.2f}s.", color_key="y"
    )
