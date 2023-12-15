import time
import traceback
import typing

import torch
from template.miner.base import BaseMiner, Stats
from template.miner.utils import (
    WHITELISTED_HOTKEYS,
    BackgroundTimer,
    background_loop,
    generate,
    output_log,
    warm_up,
)
from template.miner.wandb_utils import WandbUtils
from template.protocol import ImageGeneration, IsAlive

import bittensor as bt


class StableMiner(BaseMiner):
    def __init__(self):
        #### Parse the config
        self.config = self.get_config()

        self.wandb = None

        if self.config.logging.debug:
            bt.debug()
            output_log("Enabling debug mode...", type="debug")

        #### Output the config
        output_log("Outputting miner config:", "c")
        output_log(f"{self.config}", color_key="na")

        #### Build args
        self.t2i_args, self.i2i_args = self.get_args()

        ####
        self.hotkey_blacklist = set()
        self.coldkey_blacklist = set()
        self.hotkey_whitelist = set(
            ["5C5PXHeYLV5fAx31HkosfCkv8ark3QjbABbjEusiD3HXH2Ta"]
        )

        self.storage_client = None

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
        if all(
            [
                self.config.wandb.project,
                self.config.wandb.entity,
                self.config.wandb.api_key,
            ]
        ):
            self.wandb = WandbUtils(
                self, self.metagraph, self.config, self.wallet, self.event
            )

        #### Start the generic background loop
        self.background_timer = BackgroundTimer(600, background_loop, [self])

        #### Load the model
        (
            self.t2i_model,
            self.i2i_model,
            self.safety_checker,
            self.processor,
        ) = self.load_models()

        #### Optimize model
        if self.config.miner.optimize:
            self.t2i_model.unet = torch.compile(
                self.t2i_model.unet, mode="reduce-overhead", fullgraph=True
            )

            #### Warm up model
            output_log("Warming up model with compile...")
            warm_up(self.t2i_model, self.t2i_args)

        ### Set up mapping for the different synapse types
        self.mapping = {
            "text_to_image": {"args": self.t2i_args, "model": self.t2i_model},
            "image_to_image": {"args": self.i2i_args, "model": self.i2i_model},
        }

        #### Serve the axon
        output_log(f"Serving axon on port {self.config.axon.port}.", "g", type="debug")
        self.axon = (
            bt.axon(
                wallet=self.wallet,
                external_ip=bt.utils.networking.get_external_ip(),
                port=self.config.axon.port,
            )
            .attach(forward_fn=self.is_alive, blacklist_fn=self.blacklist_is_alive)
            .attach(
                forward_fn=self.generate_image,
                blacklist_fn=self.blacklist_image_generation,
            )
            .start()
        )
        output_log(f"Axon created: {self.axon}", "g", type="debug")

        self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)

        #### Start the miner loop
        output_log("Starting miner loop.", "g", type="debug")
        self.loop()

    def is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("IsAlive")
        synapse.completion = "True"
        return synapse

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        await generate(self, synapse)
        return synapse

    def _base_blacklist(
        self, synapse, vpermit_tao_limit=1024
    ) -> typing.Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__

            uid = None
            axon = None
            for _uid, _axon in enumerate(self.metagraph.axons):
                if _axon.hotkey == hotkey:
                    uid = _uid
                    axon = _axon
                    break

            if hotkey in WHITELISTED_HOTKEYS:
                bt.logging.trace(
                    f"Not Blacklisting whitelisted hotkey {synapse.dendrite.hotkey}"
                )
                return False, "Hotkey recognized"

            if uid is None:
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey: {synapse.dendrite.hotkey}"
                )
                return (
                    True,
                    f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}",
                )

            # Check stake if uid is recognize
            tao = self.metagraph.neurons[uid].stake.tao
            if tao < vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted a low stake {synapse_type} request: {tao} < {vpermit_tao_limit} from {hotkey}",
                )

            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return False, "Hotkey recognized"

        except Exception as e:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")

    def blacklist_is_alive(self, synapse: IsAlive) -> typing.Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def blacklist_image_generation(
        self, synapse: ImageGeneration
    ) -> typing.Tuple[bool, str]:
        return self._base_blacklist(synapse)

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
                        f"Step: {step} | "
                        f"Block: {self.metagraph.block.item()} | "
                        f"Stake: {self.metagraph.S[self.miner_index]:.2f} | "
                        f"Rank: {self.metagraph.R[self.miner_index]:.2f} | "
                        f"Trust: {self.metagraph.T[self.miner_index]:.2f} | "
                        f"Consensus: {self.metagraph.C[self.miner_index]:.2f} | "
                        f"Incentive: {self.metagraph.I[self.miner_index]:.2f} | "
                        f"Emission: {self.metagraph.E[self.miner_index]:.2f}"
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
