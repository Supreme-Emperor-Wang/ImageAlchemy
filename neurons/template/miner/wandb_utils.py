import copy
from threading import Timer

from template.miner.utils import output_log

import wandb
import torch
import bittensor as bt


#### Wandb functions
class WandbTimer(Timer):
    def run(self):
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class WandbUtils:
    def __init__(self, miner, metagraph, config, wallet, event):
        self.miner = miner
        self.metagraph = metagraph
        self.config = config
        self.wallet = wallet
        self.wandb = None
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.event = event
        self.timer = WandbTimer(600, self._loop, [])
        self.timer.start()

    def _loop(self):
        if not self.wandb:
            self._start_run()
        else:
            self._log()

    def _start_run(self):
        if self.wandb:
            self._stop_run()

        output_log(
            f"Wandb starting run with project {self.config.wandb.project} and entity {self.config.wandb.entity}."
        )

        #### Start new run
        config = copy.deepcopy(self.config)
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

    def _add_images(self, synapse):
        ### Store the images and prompts for uploading to wandb
        if self.config.wandb.compress:
            file_type = "jpg"
        else:
            file_type = "png"

        self.event.update(
            {
                "images": [
                    wandb.Image(
                        bt.Tensor.deserialize(image),
                        caption=synapse.prompt,
                        file_type=file_type,
                    )
                    if image != []
                    else wandb.Image(
                        torch.full([3, 1024, 1024], 255, dtype=torch.float),
                        caption=synapse.prompt,
                        file_type=file_type,
                    )
                    for image in synapse.images
                ],
            }
        )

    def _stop_run(self):
        self.wandb.finish()

    def _log(self):
        #### Log incentive, trust, emissions, total requests, timeouts
        self.event.update(self.miner.get_miner_info())
        self.event.update(
            {
                "total_requests": self.miner.stats.total_requests,
                "timeouts": self.miner.stats.timeouts,
            }
        )
        self.wandb.log(self.event)
