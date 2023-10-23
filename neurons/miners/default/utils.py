from threading import Timer
from typing import List
import bittensor as bt
import wandb


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
    def __init__(self, miner):
        self.miner = miner
        self.wandb = None
        output_log(
            f"Wandb starting run with project {self.miner.config.wandb.project} and entity {self.miner.config.wandb.entity}."
        )
        self.timer = WandbTimer(600, self._log, [self])
        self.timer.start()

    def _start_run(self):
        if self.wandb:
            self._stop_run()

        #### Start new run

        config = {}
        config.update(self.args)
        config["model"] = self.miner.config.model
        self.wandb = wandb.init(
            project=self.wandb.project, entity=self.wandb.entity, config=config
        )

        #### Take the first two random words plus the name of the wallet, hotkey name and uid
        self.wandb.name = (
            self.wandb.name.split("-")[:2]
            + f"-{self.wallet.name}-{self.wallet.hotkey_str}-{self.args.uid}"
        )
        output_log(f"Started new run: {self.wandb.name}", "c")

    def _stop_run(self):
        self.wandb.finish()

    def _log(self):
        if not self.wandb:
            self._start_run()
            return
        #### Log incentive, trust, emissions, total requests, timeouts
        info = self.miner.get_miner_info()
        info.update(
            {
                "total_requests": self.miner.stats.total_requests,
                "timeouts": self.miner.stats.timeouts,
            }
        )
        self.wandb.log(info)
