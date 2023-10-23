from datetime import datetime
import time
from typing import Dict, List, Tuple
import bittensor as bt
from generate import generate
from utils import output_log, sh, Images


def generate(model, args: Dict) -> List:
    images = model(**args).images
    return images


def get_caller_stake(self, synapse):
    """
    Look up the stake of the requesting validator.
    """
    if synapse.dendrite.hotkey in self.miner.metagraph.hotkeys:
        index = self.miner.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return self.miner.metagraph.S[index].item()
    return None


def do_logs(self, synapse):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.miner.stats.start_time
    output_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.miner.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.miner.stats.total_requests/(time_elapsed.total_seconds()/60):.2f} | Model {self.miner.model['model_name']} | Seed {self.miner.seed}."
    )
    output_log(
        f"{sh('Stats')} -> Total requests {self.miner.stats.total_requests} | Timeouts {self.miner.stats.timeouts}."
    )
    requester_stake = get_caller_stake(self, synapse)
    if not requester_stake:
        requester_stake = -1
    output_log(
        f"{sh('Caller')} -> Stake {int(requester_stake):,} | Hotkey {synapse.dendrite.hotkey}"
    )
    num_images = self.miner.args["num_images_per_prompt"]
    output_log(f"{sh('Generating')} -> {num_images} images.")


def shared_logic(self, synapse, t2i=True):
    """
    Forward logic shared between both text-to-image and image-to-image
    """
    do_logs(self, synapse)

    start_time = time.perf_counter()
    if t2i:
        images = generate(self.miner.t2i_model, self.miner.t2i_args)
    else:
        images = generate(self.miner.i2i_model, self.miner.i2i_args)
    output_log(f"{sh('Time')} -> {time.perf_counter() - start_time:.2f}s.")
    synapse.images = images


class Synapses:
    class TextToImage:
        def forward_fn(self, synapse):
            shared_logic(self, synapse)

            return synapse

        def blacklist_fn(self, synapse) -> Tuple[bool, str]:
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                #### Ignore requests from non-registered entities
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}."
                )
                return True, "Unrecognized hotkey."

            #### Get index of caller uid
            uid_index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            if not self.metagraph.validator_permit[uid_index]:
                return True, "No validator permit."

            if self.metagraph.S[uid_index] < 1024:
                return True, "Insufficient stake."

            return False, "Hotkey recognized."

        def priority_fn(self, synapse) -> float:
            #### Get index of requestor
            uid_index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            #### Return stake as priority
            return float(self.metagraph.S[uid_index])

    class ImageToImage:
        def forward_fn(self, synapse):
            shared_logic(self, synapse, t2i=False)
            return synapse

        def blacklist_fn(self, synapse) -> Tuple[bool, str]:
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                #### Ignore requests from non-registered entities
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey."

            #### Get index of caller uid
            uid_index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            if not self.metagraph.validator_permit[uid_index]:
                return True, "No validator permit."

            if self.metagraph.S[uid_index] < 1024:
                return True, "Insufficient stake."

            return False, "Hotkey recognized."

        def priority_fn(self, synapse) -> float:
            #### Get index of requestor
            uid_index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            #### Return stake as priority
            return float(self.metagraph.S[uid_index])

    def __init__(self, miner):
        self.text_to_image = self.TextToImage()
        self.image_to_image = self.ImageToImage()
        self.miner = miner
