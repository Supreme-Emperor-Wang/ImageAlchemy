import copy
import time
from datetime import datetime
from threading import Timer
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


#### Wandb functions
class WandbTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class WandbUtils:
    def __init__(self, miner, metagraph, config, wallet, event):
        # breakpoint()
        self.miner = miner
        self.metagraph = metagraph
        self.config = config
        self.wallet = wallet
        self.wandb = None
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.event = event
        output_log(
            f"Wandb starting run with project {self.config.wandb.project} and entity {self.config.wandb.entity}."
        )
        # self.timer = WandbTimer(600, self._log, [self])
        # self.timer.start()

    def _start_run(self):
        if self.wandb:
            self._stop_run()

        #### Start new run

        config = {}
        config.update(self.config)
        config["model"] = self.config.model
        self.wandb = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=config,
        )

        #### Take the first two random words plus the name of the wallet, hotkey name and uid
        self.wandb.name = (
            "-".join(self.wandb.name.split("-")[:2])
            + f"-{self.wallet.name}-{self.wallet.hotkey_str}-{self.uid}"
        )
        output_log(f"Started new run: {self.wandb.name}", "c")

    def _stop_run(self):
        self.wandb.finish()

    def _log(self):
        if not self.wandb:
            self._start_run()
            return
        # breakpoint()
        #### Log incentive, trust, emissions, total requests, timeouts
        self.event.update(self.miner.get_miner_info())
        self.event.update(
            {
                "total_requests": self.miner.stats.total_requests,
                "timeouts": self.miner.stats.timeouts,
            }
        )
        self.wandb.log(self.event)


def generate(model, args: Dict, synapse) -> List:
    local_args = copy.copy(args)
    local_args["prompt"] = synapse.prompt
    local_args["target_size"] = (synapse.height, synapse.width)
    # breakpoint()
    images = model(**local_args).images
    return images


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


def shared_logic(self, synapse, timeout=10):
    """
    Forward logic shared between both text-to-image and image-to-image
    """
    # Increment total requests state by 1

    self.stats.total_requests += 1
    # breakpoint()
    do_logs(self, synapse)
    # breakpoint()
    start_time = time.perf_counter()
    if synapse.generation_type == "text_to_image":
        images = generate(self.t2i_model, self.t2i_args, synapse)
    elif synapse.generation_type == "image_to_image":
        images = generate(self.i2i_model, self.i2i_args, synapse)
    else:
        bt.logging.debug(
            f"Generation type should be one of either text_to_image or image_to_image."
        )
    # breakpoint()
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

    #### Log to console
    output_log(f"{sh('Time')} -> {time.perf_counter() - start_time:.2f}s.")
