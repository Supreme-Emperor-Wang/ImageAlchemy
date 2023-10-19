import time
from typing import Tuple
import bittensor as bt
from generate import generate
from utils import output_log


class Synapses:
    class TextToImage:
        def forward_fn(self, synapse):
            output_log(f"Generating {self.miner.args['num_images_per_prompt']} images.")
            start_time = time.perf_counter()
            images = generate(self.miner.t2i_model, self.miner.t2i_args)
            end_time = time.perf_counter() - start_time

            synapse.images = images

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
            pass

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
