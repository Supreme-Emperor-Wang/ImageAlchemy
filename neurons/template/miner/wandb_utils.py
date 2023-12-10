from threading import Timer

from template.miner.utils import output_log

import wandb


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

        wandb.login(self.config.wandb.api_key)

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
