import argparse
import os
import time
import traceback
import bittensor as bt

# from config import get_config
from synapses import Synapses
from utils import output_log


class BaseMiner:
    def get_config(self) -> "bt.config":
        argp = argparse.ArgumentParser(description="Miner Configs")

        #### Add any args from the parent class
        self.add_args(argp)

        argp.add_argument("--netuid", type=int, default=1)
        argp.add_argument("--wandb.project", type=str, default="")
        argp.add_argument("--wandb.entity", type=str, default="")
        argp.add_argument("--miner.device", type=str, default="cuda:0")

        bt.subtensor.add_args(argp)
        bt.logging.add_args(argp)
        bt.wallet.add_args(argp)
        bt.axon.add_args(argp)

        config = bt.config(argp)

        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "miner",
            )
        )
        #### Ensure the directory for logging exists
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path, exist_ok=True)

        return config

    def add_args(cls, argp: argparse.ArgumentParser):
        pass

    def loop_until_registered(self):
        index = None
        while True:
            index = self.get_miner_index()
            if index is not None:
                self.miner_index = index
                output_log(
                    f"Miner {self.wallet.config.hotkey} is registered on uid {self.metagraph.uids[self.miner_index]}.",
                    "g",
                )
                break
            output_log(
                f"Miner {self.wallet.config.hotkey} is not registered. Sleeping for 30 seconds...",
                "r",
            )
            time.sleep(30)

    def __init__(self):
        #### Parse the config
        self.config = self.get_config()

        #### Output the config
        output_log("Outputting miner config:", "c")
        output_log(f"{self.config}")

        #### Initialize the synapse classes
        self.synapses = Synapses()

        #### Establish subtensor connection
        output_log("Establishing subtensor connection.", "g", type="debug")
        self.subtensor = bt.subtensor(config=self.config)

        #### Create the metagraph
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)

        #### Configure the wallet
        self.wallet = bt.wallet(config=self.config)

        #### Wait until the miner is registered
        self.loop_until_registered()

        #### Serve the axon
        output_log(f"Serving axon on port {self.config.axon.port}.", "g", type="debug")
        # self.axon = (
        #     bt.axon(
        #         wallet=self.wallet,
        #         config=self.config,
        #         ip="127.0.0.1",
        #         external_ip=bt.utils.networking.get_external_ip(),
        #     )
        #     .attach(self.synapses.text_to_image.forward_fn, self.synapses.text_to_image.blacklist_fn)
        #     .attach(self.synapses.image_to_image.forward_fn, self.synapses.image_to_image.blacklist_fn)
        #     .start()
        # )

        #### Start the weight setting loop
        output_log("Starting weight setting loop.", "g", type="debug")
        self.loop()

    def get_miner_index(self):
        """
        Retrieve the given miner's index in the metagraph.
        """
        index = None
        try:
            index = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            pass
        return index

    def check_still_registered(self):
        self.miner_index = self.get_miner_index()
        return True if self.miner_index is not None else False

    def get_incentive(self):
        return (
            self.metagraph.I[self.miner_index] * 100_000
            if self.miner_index is not None
            else 0
        )

    def get_trust(self):
        return (
            self.metagraph.T[self.miner_index] * 100
            if self.miner_index is not None
            else 0
        )

    def get_consensus(self):
        return (
            self.metagraph.C[self.miner_index] * 100_000
            if self.miner_index is not None
            else 0
        )

    def loop(self):
        step = 0
        while True:
            ### Check the miner is still registered
            is_registered = self.check_still_registered()

            if not is_registered:
                output_log("The miner is not currently registered.", "r")
                time.sleep(30)
                continue

            #### Output current statistics and set weights

            try:
                if step % 5 == 0:
                    ### Output metrics
                    log = (
                        f"Step:{step} | "
                        f"Block:{self.metagraph.block.item()} | "
                        f"Stake:{self.metagraph.S[self.miner_index]} | "
                        f"Rank:{self.metagraph.R[self.miner_index]} | "
                        f"Trust:{self.metagraph.T[self.miner_index]} | "
                        f"Consensus:{self.metagraph.C[self.miner_index] } | "
                        f"Incentive:{self.metagraph.I[self.miner_index]} | "
                        f"Emission:{self.metagraph.E[self.miner_index]}"
                    )
                    output_log(log, "g")

                    ### Set weights (WIP)
                    output_log("Settings weights.")

                    weights = [0.0] * len(self.metagraph.uids)
                    weights[self.miner_index] = 1.0

                    uids = self.metagraph.uids

                    self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.config.netuid,
                        weights=weights,
                        uids=uids,
                    )
                    output_log("Weights set.")

                step += 1
                time.sleep(60)

            #### If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            #### In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

            time.sleep(30)
