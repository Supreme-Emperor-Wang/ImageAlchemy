import bittensor as bt
import torch.nn as nn
import torch

# Utils for checkpointing and saving the model.
import torch
import wandb
import copy
import bittensor as bt
import random
from typing import List
# import prompting.validators as validators
# from prompting.validators.misc import ttl_get_block

# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import math
import hashlib as rpccheckhealth
from math import floor
from typing import Callable, Any
from functools import lru_cache, update_wrapper


# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    return self.subtensor.get_current_block()

def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, 400
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # breakpoint()
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        # available_uids += random.sample(
        #     [uid for uid in avail_uids if uid not in candidate_uids],
        #     k - len(candidate_uids),
        # )
        uids = torch.tensor(available_uids)
    else:
        uids = torch.tensor(random.sample(available_uids, k))
    return uids

def generate_random_prompt(self):
    
    # Pull a random prompt from the dataset and cut to 1-7 words
    prompt_trim_length = random.randint(1, 7)
    old_prompt = " ".join(next(self.dataset)['prompt'].split(" ")[:prompt_trim_length])

    # Generate a new prompt from the truncated prompt using the prompt generation pipeline
    new_prompt = self.prompt_generation_pipeline(old_prompt, min_length=10 )[0]['generated_text']

    return new_prompt


def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity.item()

def compare_to_set(image_array, target_size=(224, 224)):
    # convert image array to index, image tuple pairs
    image_array = [(i, image) for i, image in enumerate(image_array)]

    # if there are no images, return an empty matrix
    if len(image_array) == 0:
        return []

    # only process images that are not None
    style_vectors = extract_style_vectors([image for _, image in image_array if image is not None], target_size)
    # add back in the None images as zero vectors
    for i, image in image_array:
        if image is None:
            # style_vectors = torch.cat((style_vectors[:i], torch.zeros(1, style_vectors.size(1)), style_vectors[i:]))
            # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_CUDA_cat)
            # Fixed version:
            style_vectors = torch.cat((style_vectors[:i], torch.zeros(1, style_vectors.size(1)).to(style_vectors.device), style_vectors[i:]))

    similarity_matrix = torch.zeros(len(image_array), len(image_array))
    for i in range(style_vectors.size(0)):
        for j in range(style_vectors.size(0)):
            if image_array[i] is not None and image_array[j] is not None:
                similarity = cosine_similarity(style_vectors[i], style_vectors[j])
                likeness = 1.0 - similarity  # Invert the likeness to get dissimilarity
                likeness = min(1,max(0, likeness))  # Clip the likeness to [0,1]
                if likeness < 0.01:
                    likeness = 0
                similarity_matrix[i][j] = likeness

    return similarity_matrix.tolist()

def calculate_mean_dissimilarity(dissimilarity_matrix):
    num_images = len(dissimilarity_matrix)
    mean_dissimilarities = []

    for i in range(num_images):
        dissimilarity_values = [dissimilarity_matrix[i][j] for j in range(num_images) if i != j]
        # error: list index out of range
        if len(dissimilarity_values) == 0 or sum(dissimilarity_values) == 0:
            mean_dissimilarities.append(0)
            continue
        # divide by amount of non zero values
        non_zero_values = [value for value in dissimilarity_values if value != 0]
        mean_dissimilarity = sum(dissimilarity_values) / len(non_zero_values)
        mean_dissimilarities.append(mean_dissimilarity)

     # Min-max normalization
    non_zero_values = [value for value in mean_dissimilarities if value != 0]

    if(len(non_zero_values) == 0):
        return [0.5] * num_images

    min_value = min(non_zero_values)
    max_value = max(mean_dissimilarities)
    range_value = max_value - min_value
    if range_value != 0:
        mean_dissimilarities = [(value - min_value) / range_value for value in mean_dissimilarities]
    else:
        # All elements are the same (no range), set all values to 0.5
        mean_dissimilarities = [0.5] * num_images
    # clamp to [0,1]
    mean_dissimilarities = [min(1,max(0, value)) for value in mean_dissimilarities]

    # Ensure sum of values is 1 (normalize)
    # sum_values = sum(mean_dissimilarities)
    # if sum_values != 0:
    #     mean_dissimilarities = [value / sum_values for value in mean_dissimilarities]

    return mean_dissimilarities

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return (
        not self.config.wandb.off
        and self.step
        and self.step % self.config.wandb.run_step_length == 0
    )


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""
    tags = [
        self.wallet.hotkey.ss58_address,
        # validators.__version__,
        # str(validators.__spec_version__),
        f"netuid_{self.metagraph.netuid}",
    ]

    if self.config.mock:
        tags.append("mock")
    if self.config.neuron.use_custom_gating_model:
        tags.append("custom_gating_model")
    for fn in self.reward_functions:
        if not self.config.neuron.mock_reward_models:
            tags.append(str(fn.name))
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")
    if self.config.neuron.disable_log_rewards:
        tags.append("disable_log_rewards")

    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "reward", "netuid", "wandb")
    }
    wandb_config["neuron"].pop("full_path", None)

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=self.config.wandb.project_name,
        entity=self.config.wandb.entity,
        config=wandb_config,
        mode="offline" if self.config.wandb.offline else "online",
        dir=self.config.neuron.full_path,
        tags=tags,
        notes=self.config.wandb.notes,
    )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    self.wandb.finish()
    init_wandb(self, reinit=True)


def should_checkpoint(self):
    # Check if enough epoch blocks have elapsed since the last checkpoint.
    return (
        ttl_get_block(self) % self.config.neuron.checkpoint_block_length
        < self.prev_block % self.config.neuron.checkpoint_block_length
    )


def checkpoint(self):
    """Checkpoints the training process."""
    bt.logging.info("checkpoint()")
    resync_metagraph(self)
    save_state(self)


def resync_metagraph(self: "validators.neuron.neuron"):
    """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
    bt.logging.info("resync_metagraph()")

    # Copies state of metagraph before syncing.
    previous_metagraph = copy.deepcopy(self.metagraph)

    # Sync the metagraph.
    self.metagraph.sync(subtensor=self.subtensor)

    # Check if the metagraph axon info has changed.
    metagraph_axon_info_updated = previous_metagraph.axons != self.metagraph.axons

    if metagraph_axon_info_updated:
        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )

        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.moving_averaged_scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.moving_averaged_scores))
            new_moving_average[:min_len] = self.moving_averaged_scores[:min_len]
            self.moving_averaged_scores = new_moving_average

        # Resize the gating model.
        bt.logging.info("Re-syncing gating model")
        self.gating_model.resync(previous_metagraph, self.metagraph)

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)


def resync_linear_layer(
    linear_layer: torch.nn.Module,
    previous_metagraph: "bt.metagraph.Metagraph",
    metagraph: "bt.metagraph.Metagraph",
):
    """Resync the linear layer with the latest state of the network
    Args:
         linear_layer (:obj: torch.nn.Module): Linear layer to be resynced
         previous_metagraph (:obj: bt.metagraph.Metagraph):
             Previous state of metagraph before updated resync
         metagraph (:obj: bt.metagraph.Metagraph):
             Latest state of the metagraph with updated uids and hotkeys
    """
    uids_hotkeys_state_dict = dict(
        zip(previous_metagraph.uids.tolist(), previous_metagraph.hotkeys)
    )
    latest_uids_hotkeys_state_dict = dict(
        zip(metagraph.uids.tolist(), metagraph.hotkeys)
    )

    updated_uids_indices = []
    for uid, latest_hotkey in latest_uids_hotkeys_state_dict.items():
        if uids_hotkeys_state_dict.get(uid) != latest_hotkey:
            updated_uids_indices.append(uid)

    for index in updated_uids_indices:
        # Reinitialize the bias of the selected index of the linear layer
        torch.nn.init.zeros_(linear_layer.bias[index])
        # Clone the weights of the selected index of the linear layer
        weights = linear_layer.weight[index].clone()
        # Adds a dimension to the weights tensor to make it compatible with the xavier_uniform_ function
        torch.nn.init.xavier_uniform_(weights.unsqueeze(0))
        reinitialized_weights = weights.squeeze(0)
        # Copy the reinitialized weights back to the selected index of the linear layer
        linear_layer.weight[index].data.copy_(reinitialized_weights)


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def save_state(self):
    r"""Save hotkeys, gating model, neuron model and moving average scores to filesystem."""
    bt.logging.info("save_state()")
    try:
        neuron_state_dict = {
            "neuron_weights": self.moving_averaged_scores.to("cpu").tolist(),
            "neuron_hotkeys": self.hotkeys,
        }
        torch.save(neuron_state_dict, f"{self.config.neuron.full_path}/model.torch")
        bt.logging.success(
            prefix="Saved model",
            sufix=f"<blue>{ self.config.neuron.full_path }/model.torch</blue>",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to save model with error: {e}")

    try:
        # Save the gating model.
        gating_model_linear_layer_dict = self.gating_model.linear.state_dict()
        gating_model_name = self.config.gating.model_name.replace("/", "_")
        gating_model_file_path = f"{self.config.neuron.full_path}/{gating_model_name}_gating_linear_layer.pth"
        torch.save(gating_model_linear_layer_dict, gating_model_file_path)

        if not self.config.wandb.off:
            wandb.log(
                {"step": self.step, "block": ttl_get_block(self), **neuron_state_dict}
            )
        if not self.config.wandb.off and self.config.wandb.track_gating_model:
            model_artifact = wandb.Artifact(
                f"{gating_model_name}_gating_linear_layer", type="model"
            )
            model_artifact.add_file(gating_model_file_path)
            self.wandb.log_artifact(model_artifact)

        bt.logging.success(
            prefix="Saved gating model", sufix=f"<blue>{gating_model_file_path}</blue>"
        )
    except Exception as e:
        bt.logging.warning(f"Failed to save gating model with error: {e}")

    try:
        # Save diversity model.
        diversity_model_dict = {
            "historic_embeddings": self.diversity_model.historic_embeddings.to("cpu")
        }
        diversity_model_file_path = (
            f"{self.config.neuron.full_path}/diversity_model.pth"
        )
        torch.save(diversity_model_dict, diversity_model_file_path)
        bt.logging.success(
            prefix="Saved diversity model",
            sufix=f"<blue>{diversity_model_file_path}</blue> {list(self.diversity_model.historic_embeddings.shape)}",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to save diversity model with error: {e}")

    # empty cache
    torch.cuda.empty_cache()


def load_state(self):
    r"""Load hotkeys and moving average scores from filesystem."""
    bt.logging.info("load_state()")
    try:
        state_dict = torch.load(f"{self.config.neuron.full_path}/model.torch")
        neuron_weights = torch.tensor(state_dict["neuron_weights"])
        # Check to ensure that the size of the neruon weights matches the metagraph size.
        if neuron_weights.shape != (self.metagraph.n,):
            bt.logging.warning(
                f"Neuron weights shape {neuron_weights.shape} does not match metagraph n {self.metagraph.n}"
                "Populating new moving_averaged_scores IDs with zeros"
            )
            self.moving_averaged_scores[: len(neuron_weights)] = neuron_weights.to(
                self.device
            )
        # Check for nans in saved state dict
        elif not torch.isnan(neuron_weights).any():
            self.moving_averaged_scores = neuron_weights.to(self.device)
        self.hotkeys = state_dict["neuron_hotkeys"]
        bt.logging.success(
            prefix="Reloaded model",
            sufix=f"<blue>{ self.config.neuron.full_path }/model.torch</blue>",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to load model with error: {e}")

    try:
        # Load diversity model.
        diversity_model_file_path = (
            f"{self.config.neuron.full_path}/diversity_model.pth"
        )
        diversity_model_dict = torch.load(diversity_model_file_path)
        self.diversity_model.historic_embeddings = diversity_model_dict[
            "historic_embeddings"
        ].to(self.device)
        bt.logging.success(
            prefix="Reloaded diversity model",
            sufix=f"<blue>{diversity_model_file_path}</blue> {list(self.diversity_model.historic_embeddings.shape)}",
        )
    except Exception as e:
        bt.logging.warning(f"Failed to load diversity model with error: {e}")