# Utils for checkpointing and saving the model.
import asyncio
import copy
import os
import random
import time
import traceback
from functools import lru_cache, update_wrapper
from math import floor
from typing import Any, Callable, List
from neurons.constants import VPERMIT_TAO, WANDB_RUN_STEP_LENGTH, WANDB_VALIDATOR_PATH

import neurons.validator as validator
import pandas as pd
import torch
import torch.nn as nn
from neurons.protocol import IsAlive

import bittensor as bt
import wandb


def _ttl_hash_gen(seconds: int):
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


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


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    return self.subtensor.get_current_block()


def check_uid(dendrite, axon, uid):
    try:
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            dendrite(axon, IsAlive(), deserialize=False, timeout=2.3)
        )
        if response.is_success:
            bt.logging.trace(f"UID {uid} is active.")
            # loop.close()
            return True
        else:
            bt.logging.trace(f"UID {uid} is not active.")
            # loop.close()
            return False
    except Exception as e:
        bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        # loop.close()
        return False


def check_uid_availability(
    dendrite, metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
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
    # Filter for miners that are processing other responses
    if not check_uid(dendrite, metagraph.axons[uid], uid):
        return False
    # Available otherwise.
    return True


def get_random_uids(
    self, dendrite, k: int, exclude: List[int] = None
) -> torch.LongTensor:
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
            dendrite, self.metagraph, uid, VPERMIT_TAO
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if (
            uid_is_available
            and (self.metagraph.axons[uid].hotkey not in self.hotkey_blacklist)
            and (self.metagraph.axons[uid].coldkey not in self.coldkey_blacklist)
        ):
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all available uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        uids = torch.tensor(available_uids)
    else:
        uids = torch.tensor(random.sample(available_uids, k))
    return uids


def calculate_mean_dissimilarity(dissimilarity_matrix):
    num_images = len(dissimilarity_matrix)
    mean_dissimilarities = []

    for i in range(num_images):
        dissimilarity_values = [
            dissimilarity_matrix[i][j] for j in range(num_images) if i != j
        ]
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

    if len(non_zero_values) == 0:
        return [0.5] * num_images

    min_value = min(non_zero_values)
    max_value = max(mean_dissimilarities)
    range_value = max_value - min_value
    if range_value != 0:
        mean_dissimilarities = [
            (value - min_value) / range_value for value in mean_dissimilarities
        ]
    else:
        # All elements are the same (no range), set all values to 0.5
        mean_dissimilarities = [0.5] * num_images
    # clamp to [0,1]
    mean_dissimilarities = [min(1, max(0, value)) for value in mean_dissimilarities]

    # Ensure sum of values is 1 (normalize)
    # sum_values = sum(mean_dissimilarities)
    # if sum_values != 0:
    #     mean_dissimilarities = [value / sum_values for value in mean_dissimilarities]

    return mean_dissimilarities


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return self.step and self.step % WANDB_RUN_STEP_LENGTH == 0


def generate_random_prompt(self):
    # Pull a random prompt from the dataset and cut to 1-7 words
    prompt_trim_length = random.randint(1, 7)
    old_prompt = " ".join(next(self.dataset)["prompt"].split(" ")[:prompt_trim_length])

    # Generate a new prompt from the truncated prompt using the prompt generation pipeline
    new_prompt = self.prompt_generation_pipeline(old_prompt, min_length=10)[0][
        "generated_text"
    ]

    return new_prompt


def generate_random_prompt_gpt(self):
    for _ in range(2):
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an image prompt generator. Your purpose is to generate creative one sentence prompts that can be fed into Dalle-3.",
                    },
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            new_prompt = response.choices[0].message.content
            bt.logging.trace(f"T2I prompt is {new_prompt}")
            return new_prompt

        except Exception as e:
            bt.logging.info(f"Error when calling OpenAI: {e}")
            time.sleep(0.5)

    return None


def generate_followup_prompt_gpt(self, prompt):
    messages = [
        {"role": "system", "content": "You are an image prompt generator."},
        {"role": "assistant", "content": f"{prompt}"},
        {
            "role": "user",
            "content": "An image has now been generated from your first prompt. What is a second instruction that can be applied to this generated image?",
        },
    ]

    for _ in range(2):
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            new_prompt = response.choices[0].message.content
            bt.logging.trace(f"I2I prompt is {new_prompt}")
            return new_prompt

        except Exception as e:
            bt.logging.info(f"Error when calling OpenAI: {e}")
            time.sleep(0.5)

    return None


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""
    tags = [
        self.wallet.hotkey.ss58_address,
        str(validator.__version__),
        f"netuid_{self.metagraph.netuid}",
    ]

    if self.config.mock:
        tags.append("mock")

    for fn in self.reward_functions:
        if not self.config.neuron.mock_reward_models:
            tags.append(str(fn.name))

    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "reward", "netuid", "wandb")
    }
    wandb_config["neuron"].pop("full_path", None)

    if not os.path.exists(WANDB_VALIDATOR_PATH):
        os.makedirs(WANDB_VALIDATOR_PATH, exist_ok=True)

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project="ImageAlchemy",
        entity="tensoralchemy",
        config=wandb_config,
        dir=WANDB_VALIDATOR_PATH,
        tags=tags,
    )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    self.wandb.finish()
    init_wandb(self, reinit=True)


def get_promptdb_backup(prompt_history=[]):
    api = wandb.Api()
    runs = api.runs(f"tensoralchemy/ImageAlchemy")
    for run in runs:
        if run.historyLineCount >= 100:
            history = run.history()
            if ("prompt_t2i" not in history.columns) or (
                "prompt_i2i" not in history.columns
            ):
                continue
            for i in range(0, len(history) - 1, 2):
                if (
                    pd.isna(history.loc[i, "prompt_t2i"])
                    or (history.loc[i, "prompt_t2i"] is None)
                    or (i == len(history))
                    or (history.loc[i + 1, "prompt_i2i"] is None)
                    or pd.isna(history.loc[i + 1, "prompt_i2i"])
                ):
                    continue
                else:
                    prompt_tuple = (
                        history.loc[i, "prompt_t2i"],
                        history.loc[i + 1, "prompt_i2i"],
                    )
                    if prompt_tuple in prompt_history:
                        continue
                    else:
                        prompt_history.append(prompt_tuple)

    return prompt_history
