import time
import traceback
import typing

import torch
from template.miner.base import BaseMiner, Stats
from template.miner.utils import WHITELISTED_HOTKEYS, generate, output_log
from template.miner.wandb_utils import WandbUtils
from template.protocol import ImageGeneration, IsAlive

import bittensor as bt


class StableMiner(BaseMiner):
    def __init__(self):
        #### Parse the config
        self.config = self.get_config()

        if self.config.logging.debug:
            output_log("Enabling debug mode...", type="debug")
            bt.debug()

        #### Output the config
        output_log("Outputting miner config:", "c")
        output_log(f"{self.config}", color_key="na")

        #### Build args
        self.t2i_args, self.i2i_args = self.get_args()

        #### Initialise event dict
        self.event = {}

        #### Establish subtensor connection
        output_log("Establishing subtensor connection.", "g", type="debug")
        self.subtensor = bt.subtensor(config=self.config)

        #### Create the metagraph
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)

        #### Configure the wallet
        self.wallet = bt.wallet(config=self.config)

        #### Wait until the miner is registered
        self.loop_until_registered()

        ### Defaults
        self.stats = self.get_defaults()

        ### Start the wandb logging thread if both project and entity have been provided
        if all([self.config.wandb.project, self.config.wandb.entity]):
            self.wandb = WandbUtils(
                self, self.metagraph, self.config, self.wallet, self.event
            )
            self.wandb._start_run()

        #### Load the model
        self.t2i_model, self.i2i_model = self.load_models()

        #### Optimize model
        if self.config.miner.optimize:
            self.t2i_model.unet = torch.compile(
                self.t2i_model.unet, mode="reduce-overhead", fullgraph=True
            )

            #### Warm up model
            output_log("Warming up model with compile...")
            generate(self.t2i_model, self.t2i_args)

        #### Load the safety checker (WIP)

        #### Serve the axon
        output_log(f"Serving axon on port {self.config.axon.port}.", "g", type="debug")
        self.axon = (
            bt.axon(
                wallet=self.wallet,
                external_ip=bt.utils.networking.get_external_ip(),
                port=self.config.axon.port,
            )
            .attach(
                self.generate_image,
            )
            .attach(
                self.is_alive,
            )
            .attach(
                self.blacklist,
            )
            .start()
        )
        output_log(f"Axon created: {self.axon}", "g", type="debug")

        self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)
        # self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        #### Start the weight setting loop
        output_log("Starting weight setting loop.", "g", type="debug")
        self.loop()

    def is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("answered to be active")
        synapse.completion = "True"
        return synapse

    def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        generate(self, synapse)
        return synapse

    def blacklist(self, synapse: ImageGeneration) -> typing.Tuple[bool, str]:
        if (synapse.dendrite.hotkey not in self.metagraph.hotkeys) and (
            synapse.dendrite.hotkey not in WHITELISTED_HOTKEYS
        ):
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    def loop(self):
        step = 0
        while True:
            #### Check the miner is still registered
            is_registered = self.check_still_registered()

            if not is_registered:
                output_log("The miner is not currently registered.", "r")
                time.sleep(120)

                ### Ensure the metagraph is synced before the next registration check
                self.metagraph.sync(lite=True)
                continue

            #### Output current statistics and set weights
            try:
                if step % 5 == 0:
                    #### Output metrics
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


if __name__ == "__main__":
    with StableMiner():
        while True:
            time.sleep(1)
