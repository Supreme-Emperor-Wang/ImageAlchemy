from datetime import datetime
import time
from typing import Dict, List, Tuple
import bittensor as bt
from utils import output_log, sh, Images
import template
import copy
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.PILToTensor()
])

def generate(model, args: Dict, synapse) -> List:
    d = copy.copy(args)
    d["prompt"] = synapse.prompt
    d["target_size"] = (synapse.height, synapse.width)
    images = model(**d).images
    return images


def get_caller_stake(self, synapse):
    """
    Look up the stake of the requesting validator.
    """
    if synapse.dendrite.hotkey in self.miner.metagraph.hotkeys:
        index = self.miner.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return self.miner.metagraph.S[index].item()
    return None


def do_logs(self, synapse, t2i):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.miner.stats.start_time
    
    if synapse.generation_type == "text_to_image":
        num_images = self.miner.t2i_args["num_images_per_prompt"]
    elif synapse.generation_type == "image_to_image":
        num_images = self.miner.i2i_args["num_images_per_prompt"]
    else:
        bt.logging.debug(
            f"Generation type should be one of either text_to_image or image_to_image."
        )

    output_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.miner.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.miner.stats.total_requests/(time_elapsed.total_seconds()/60):.2f} | Model {self.miner.config.miner.model} | Seed {self.miner.config.miner.seed}."
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
    output_log(f"{sh('Generating')} -> {num_images} images.")


def shared_logic(self, synapse, t2i=True):
    """
    Forward logic shared between both text-to-image and image-to-image
    """
    do_logs(self, synapse, t2i)

    start_time = time.perf_counter()
    if t2i:
        images = generate(self.miner.t2i_model, self.miner.t2i_args, synapse)
    else:
        images = generate(self.miner.i2i_model, self.miner.i2i_args, synapse)

    synapse.images = [bt.Tensor.serialize( transform(image) ) for image in images]
    
    output_log(f"{sh('Time')} -> {time.perf_counter() - start_time:.2f}s.")



class Synapses:
    class TextToImage:
        def __init__(self, miner):
            self.miner = miner

        def forward_fn(self, synapse: template.protocol.ImageGeneration):
            shared_logic(self, synapse)

            return synapse

        def blacklist_fn(self, synapse: template.protocol.ImageGeneration) -> Tuple[bool, str]:
            if synapse.dendrite.hotkey not in self.miner.metagraph.hotkeys:
                #### Ignore requests from non-registered entities
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}."
                )
                return True, "Unrecognized hotkey."

            #### Get index of caller uid
            uid_index = self.miner.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            if not self.miner.metagraph.validator_permit[uid_index]:
                return True, "No validator permit."

            if self.miner.metagraph.S[uid_index] < 1024:
                return True, "Insufficient stake."

            return False, "Hotkey recognized."

        def priority_fn(self, synapse: template.protocol.ImageGeneration) -> float:
            #### Get index of requestor
            uid_index = self.miner.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            #### Return stake as priority
            return float(self.miner.metagraph.S[uid_index])

    class ImageToImage:
        def __init__(self, miner):
            self.miner = miner

        def forward_fn(self, synapse: template.protocol.ImageGeneration):
            shared_logic(self, synapse, t2i=False)
            return synapse

        def blacklist_fn(self, synapse: template.protocol.ImageGeneration) -> Tuple[bool, str]:
            if synapse.dendrite.hotkey not in self.miner.metagraph.hotkeys:
                #### Ignore requests from non-registered entities
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey."

            #### Get index of caller uid
            uid_index = self.miner.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            if not self.miner.metagraph.validator_permit[uid_index]:
                return True, "No validator permit."

            if self.miner.metagraph.S[uid_index] < 1024:
                return True, "Insufficient stake."

            return False, "Hotkey recognized."

        def priority_fn(self, synapse: template.protocol.ImageGeneration) -> float:
            #### Get index of requestor
            uid_index = self.miner.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            #### Return stake as priority
            return float(self.miner.metagraph.S[uid_index])

    def __init__(self, miner):
        self.miner = miner
        self.text_to_image = self.TextToImage(self.miner)
        self.image_to_image = self.ImageToImage(self.miner)
