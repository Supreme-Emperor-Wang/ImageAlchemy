import argparse
import asyncio
import copy
import os
import random
import time
import traceback
import typing
from abc import ABC
from typing import Dict

import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from neurons.constants import VPERMIT_TAO
from neurons.protocol import ImageGeneration, IsAlive
from neurons.utils import BackgroundTimer, background_loop, get_defaults
from utils import (
    clean_nsfw_from_prompt,
    do_logs,
    get_caller_stake,
    get_coldkey_for_hotkey,
    nsfw_image_filter,
    output_log,
    sh,
)
from wandb_utils import WandbUtils

import bittensor as bt


class BaseMiner(ABC):
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

        #### Init blacklists and whitelists
        self.hotkey_blacklist = set()
        self.coldkey_blacklist = set()
        self.coldkey_whitelist = set(
            ["5F1FFTkJYyceVGE4DCVN5SxfEQQGJNJQ9CVFVZ3KpihXLxYo"]
        )
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
        self.stats = get_defaults(self)

        ### Set up transform function
        self.transform = transforms.Compose([transforms.PILToTensor()])

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
        self.background_steps = 1
        self.background_timer = BackgroundTimer(300, background_loop, [self, False])
        self.background_timer.daemon = True
        self.background_timer.start()

        ### Init history dict
        self.request_dict = {}

    def start_axon(self):
        #### Serve the axon
        output_log(f"Serving axon on port {self.config.axon.port}.", "g", type="debug")
        self.axon = (
            bt.axon(
                wallet=self.wallet,
                ip=bt.utils.networking.get_external_ip(),
                external_ip=bt.utils.networking.get_external_ip(),
                config=self.config
            )
            .attach(
                forward_fn=self.is_alive,
                blacklist_fn=self.blacklist_is_alive,
                priority_fn=self.priority_is_alive,
            )
            .attach(
                forward_fn=self.generate_image,
                blacklist_fn=self.blacklist_image_generation,
                priority_fn=self.priority_image_generation,
            )
            .start()
        )
        output_log(f"Axon created: {self.axon}", "g", type="debug")

        self.subtensor.serve_axon(axon=self.axon, netuid=self.config.netuid)

    def get_args(self) -> Dict:
        return {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
        }, {"guidance_scale": 5, "strength": 0.6}

    def get_config(self) -> "bt.config":
        argp = argparse.ArgumentParser(description="Miner Configs")

        #### Add any args from the parent class
        self.add_args(argp)

        argp.add_argument("--netuid", type=int, default=1)
        argp.add_argument("--wandb.project", type=str, default="")
        argp.add_argument("--wandb.entity", type=str, default="")
        argp.add_argument("--wandb.api_key", type=str, default="")
        argp.add_argument("--miner.device", type=str, default="cuda:0")
        argp.add_argument("--miner.optimize", action="store_true")

        seed = random.randint(0, 100_000_000_000)
        argp.add_argument("--miner.seed", type=int, default=seed)

        argp.add_argument(
            "--miner.model",
            type=str,
            default="stabilityai/stable-diffusion-xl-base-1.0",
        )

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
                    f"Miner {self.config.wallet.hotkey} is registered with uid {self.metagraph.uids[self.miner_index]}.",
                    "g",
                )
                break
            output_log(
                f"Miner {self.config.wallet.hotkey} is not registered. Sleeping for 120 seconds...",
                "r",
            )
            time.sleep(120)
            self.metagraph.sync(subtensor=self.subtensor)

    def get_miner_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.miner_index],
            "trust": self.metagraph.trust[self.miner_index],
            "consensus": self.metagraph.consensus[self.miner_index],
            "incentive": self.metagraph.incentive[self.miner_index],
            "emissions": self.metagraph.emission[self.miner_index],
        }

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

    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        bt.logging.info("IsAlive")
        synapse.completion = "True"
        return synapse

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """
        Image generation logic shared between both text-to-image and image-to-image
        """

        ### Misc
        timeout = synapse.timeout
        self.stats.total_requests += 1
        start_time = time.perf_counter()

        ### Set up args
        local_args = copy.deepcopy(self.mapping[synapse.generation_type]["args"])
        local_args["prompt"] = [clean_nsfw_from_prompt(synapse.prompt)]
        local_args["target_size"] = (synapse.height, synapse.width)
        try:
            local_args["guidance_scale"] = synapse.guidance_scale
            local_args["negative_prompt"] = synapse.negative_prompt
        except:
            bt.logging.info("Validator hasn't provided a guidance_scale or negative_prompt")

        ### Get the model
        model = self.mapping[synapse.generation_type]["model"]

        if synapse.generation_type == "image_to_image":
            local_args["image"] = T.transforms.ToPILImage()(
                bt.Tensor.deserialize(synapse.prompt_image)
            )

        ### Output logs
        do_logs(self, synapse, local_args)

        ### Generate images & serialize
        for attempt in range(3):
            try:
                seed = synapse.seed if synapse.seed != -1 else self.config.miner.seed
                local_args["generator"] = [
                    torch.Generator(device=self.config.miner.device).manual_seed(seed)
                ]
                images = model(
                    **local_args,
                ).images
                synapse.images = [
                    bt.Tensor.serialize(self.transform(image)) for image in images
                ]
                output_log(
                    f"{sh('Generating')} -> Succesful image generation after {attempt+1} attempt(s).",
                    color_key="c",
                )
                break
            except Exception as e:
                bt.logging.error(
                    f"Error in attempt number {attempt+1} to generate an image: {e}"
                )
                asyncio.sleep(5)
                if attempt == 2:
                    images = []
                    synapse.images = []
                    bt.logging.error(
                        f"Failed to generate any images after {attempt+1} attempts."
                    )

        ### Count timeouts
        if time.perf_counter() - start_time > timeout:
            self.stats.timeouts += 1

        ### Log NSFW images
        if any(nsfw_image_filter(self, images)):
            bt.logging.debug(f"An image was flagged as NSFW: discarding image.")
            self.stats.nsfw_count += 1
            synapse.images = []

        ### Log to wandb
        try:
            if self.wandb:
                ### Store the images and prompts for uploading to wandb
                self.wandb._add_images(synapse)

                #### Log to Wandb
                self.wandb._log()

        except Exception as e:
            bt.logging.error(f"Error trying to log events to wandb.")

        #### Log time to generate image
        generation_time = time.perf_counter() - start_time
        self.stats.generation_time += generation_time
        output_log(
            f"{sh('Time')} -> {generation_time:.2f}s | Average: {self.stats.generation_time / self.stats.total_requests:.2f}s",
            color_key="y",
        )
        return synapse

    def _base_priority(self, synapse) -> float:
        ### If hotkey or coldkey is whitelisted and not found on the metagraph, give a priority of 5,000
        ### Caller hotkey
        caller_hotkey = synapse.dendrite.hotkey

        ### Retrieve the coldkey of the caller
        caller_coldkey = get_coldkey_for_hotkey(self, caller_hotkey)

        if (
            caller_coldkey in self.coldkey_whitelist
            or caller_hotkey in self.hotkey_whitelist
        ):
            priority = 5000
            bt.logging.trace(
                f"Prioritizing whitelisted key {synapse.dendrite.hotkey} with default value: {priority}."
            )

        try:
            caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            priority = float(self.metagraph.S[caller_uid])
            bt.logging.trace(
                f"Prioritizing key {synapse.dendrite.hotkey} with value: {priority}."
            )
        except:
            pass

        return priority

    def _base_blacklist(
        self, synapse, vpermit_tao_limit=VPERMIT_TAO, rate_limit=1
    ) -> typing.Tuple[bool, str]:
        try:
            ### Get the name of the synapse
            synapse_type = type(synapse).__name__

            ### Caller hotkey
            caller_hotkey = synapse.dendrite.hotkey

            ### Retrieve the coldkey of the caller
            caller_coldkey = get_coldkey_for_hotkey(self, caller_hotkey)

            ### Retrieve the stake of the caller
            caller_stake = get_caller_stake(self, synapse)

            ### Count the request frequencies
            exceeded_rate_limit = False
            if synapse_type == "ImageGeneration":
                ### Apply a rate limit from the same caller
                if caller_hotkey in self.request_dict.keys():
                    now = time.perf_counter()

                    ### The difference in seconds between the current request and the previous one
                    delta = now - self.request_dict[caller_hotkey]["history"][-1]

                    ### E.g., 0.3 < 1.0
                    if delta < rate_limit:
                        ### Count number of rate limited calls from caller's hotkey
                        self.request_dict[caller_hotkey]["rate_limited_count"] += 1
                        exceeded_rate_limit = True

                    ### Store the data
                    self.request_dict[caller_hotkey]["history"].append(now)
                    self.request_dict[caller_hotkey]["delta"].append(delta)
                    self.request_dict[caller_hotkey]["count"] += 1

                else:
                    ### For the first request, initialize the dictionary
                    self.request_dict[caller_hotkey] = {
                        "history": [time.perf_counter()],
                        "delta": [0],
                        "count": 0,
                        "rate_limited_count": 0,
                    }

            ### Allow through any whitelisted keys unconditionally
            ### Note that blocking these keys will result in a ban from the network
            if caller_coldkey in self.coldkey_whitelist:
                output_log(
                    f"Whitelisting coldkey's {synapse_type} request from {caller_hotkey}.",
                    color_key="g",
                )
                return False, "Whitelisted coldkey recognized."

            if caller_hotkey in self.hotkey_whitelist:
                output_log(
                    f"Whitelisting hotkey's {synapse_type} request from {caller_hotkey}.",
                    color_key="g",
                )
                return False, "Whitelisted hotkey recognized."

            ### Reject request if rate limit was exceeded and key wasn't whitelisted
            if exceeded_rate_limit:
                output_log(
                    f"Blacklisted a {synapse_type} request from {caller_hotkey}. Rate limit ({rate_limit:.2f}) exceeded. Delta: {delta:.2f}s.",
                    color_key="r",
                    type="debug",
                )
                return (
                    True,
                    f"Blacklisted a {synapse_type} request from {caller_hotkey}. Rate limit ({rate_limit:.2f}) exceeded. Delta: {delta:.2f}s.",
                )

            ### Blacklist requests from validators that aren't registered
            if caller_stake is None:
                output_log(
                    f"Blacklisted a non-registered hotkey's {synapse_type} request from {caller_hotkey}.",
                    color_key="r",
                    type="debug",
                )
                return (
                    True,
                    f"Blacklisted a non-registered hotkey's {synapse_type} request from {caller_hotkey}.",
                )

            ### Check that the caller has sufficient stake
            if caller_stake < vpermit_tao_limit:
                # output_log(
                #     f"Blacklisted a {synapse_type} request from {caller_hotkey} due to low stake: {caller_stake:.2f} < {vpermit_tao_limit}.",
                #     color_key="r",
                #     type="debug",
                # )
                return (
                    True,
                    f"Blacklisted a {synapse_type} request from {caller_hotkey} due to low stake: {caller_stake:.2f} < {vpermit_tao_limit}",
                )

            bt.logging.debug(f"Allowing recognized hotkey {caller_hotkey}")
            return False, "Hotkey recognized"

        except Exception as e:
            bt.logging.error(f"Error in blacklist: {traceback.format_exc()}")

    def blacklist_is_alive(self, synapse: IsAlive) -> typing.Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def blacklist_image_generation(
        self, synapse: ImageGeneration
    ) -> typing.Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def priority_is_alive(self, synapse: IsAlive) -> float:
        return self._base_priority(synapse)

    def priority_image_generation(self, synapse: ImageGeneration) -> float:
        return self._base_priority(synapse)

    def loop(self):
        output_log("Starting miner loop.", "g", type="debug")
        step = 0
        while True:
            #### Check the miner is still registered
            is_registered = self.check_still_registered()

            if not is_registered:
                output_log("The miner is not currently registered.", "r")
                time.sleep(120)

                ### Ensure the metagraph is synced before the next registration check
                self.metagraph.sync(subtensor=self.subtensor)
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

                    ### Show the top 10 requestors by calls along with their delta
                    ### Hotkey, count, delta, rate limited count
                    top_requestors = [
                        (k, v["count"], v["delta"], v["rate_limited_count"])
                        for k, v in self.request_dict.items()
                    ]

                    ### Retrieve total number of requests
                    total_requests_counted = sum([x[1] for x in top_requestors])

                    try:
                        ### Sort by count
                        top_requestors = sorted(
                            top_requestors, key=lambda x: x[1], reverse=True
                        )[:10]

                        if len(top_requestors) > 0:
                            formatted_str = "\n".join(
                                [
                                    f"Hotkey: {x[0]}, Count: {x[1]} ({((x[1] / total_requests_counted)*100) if total_requests_counted > 0 else 0:.2f}%), Average delta: {sum(x[2]) / len(x[2]) if len(x[2]) > 0 else 0:.2f}, Rate limited count: {x[3]}"
                                    for x in top_requestors
                                ]
                            )
                            formatted_str = f"{formatted_str}"

                            output_log(
                                f"{sh('Top Callers')} -> Metrics\n{formatted_str}",
                                color_key="c",
                            )
                    except:
                        pass

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
