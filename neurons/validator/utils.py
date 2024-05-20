# Utils for checkpointing and saving the model.
import asyncio
import copy
import os
import random
import time
import traceback
from functools import lru_cache, update_wrapper
from math import floor
from typing import Any, Callable, List, Tuple

import neurons.validator as validator
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from loguru import logger
from neurons.constants import N_NEURONS_TO_QUERY, VPERMIT_TAO, WANDB_VALIDATOR_PATH
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


async def check_uid(dendrite, self, uid, response_times):
    try:
        t1 = time.perf_counter()
        response = await dendrite(
            self.metagraph.axons[uid],
            IsAlive(),
            deserialize=False,
            timeout=self.async_timeout,
        )
        if response.is_success:
            response_times.append(time.perf_counter() - t1)
            self.isalive_dict[uid] = 0
            return True
        else:
            try:
                self.isalive_failed_count[uid] += 1
                key = self.metagraph.axons[uid].hotkey
                self.miner_query_history_fail_count[key] += 1
                # If miner doesn't respond for 3 iterations rest it's count to the average to avoid spamming
                if self.miner_query_history_fail_count[key] >= 3:
                    self.miner_query_history_duration[key] = time.perf_counter()
                    self.miner_query_history_count[key] = int(
                        np.array(list(self.miner_query_history_count.values())).mean()
                    )
            except:
                pass
            return False
    except Exception as e:
        logger.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        return False


def check_uid_availability(
    dendrite,
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
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
    # # Filter for miners that are processing other responses
    # if not await check_uid(dendrite, metagraph.axons[uid], uid):
    #     return False
    # # Available otherwise.
    return True


async def get_random_uids(
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

    # Filter for only serving miners
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

    # # Sort candidate UIDs by their count history
    # # This prioritises miners that have been queried less than average
    # candidate_uids = [i for i,_ in sorted(zip(candidate_uids, [self.miner_query_history_count[self.metagraph.axons[uid].hotkey] for uid in candidate_uids]))]

    ### Random sort candidate_uids
    random.shuffle(candidate_uids)

    # Find the first K uids that respond with IsAlive
    final_uids = []
    t0 = time.perf_counter()
    attempt_counter = 0
    avg_num_list = []
    for uid in range(0, len(candidate_uids), N_NEURONS_TO_QUERY):
        tasks = []

        logger.info(f"UIDs in pool: {final_uids}")
        logger.info(f"Querying uids: {candidate_uids[uid:uid+N_NEURONS_TO_QUERY]}")

        t1 = time.perf_counter()

        times_list = []

        for u in candidate_uids[uid : uid + N_NEURONS_TO_QUERY]:
            tasks.append(check_uid(dendrite, self, u, times_list))

        responses = await asyncio.gather(*tasks)
        attempt_counter += 1

        logger.info(f"Time to get responses: {time.perf_counter() - t1:.2f}s")

        list_slice = times_list[-25:]
        time_sum = sum(list_slice)

        logger.info(
            f"Number of times stored: {len(times_list)} | Average successful response across {len(list_slice)} samples: {time_sum / len(list_slice) if len(list_slice) > 0 else 0:.2f}"
        )

        if True in responses:
            t2 = time.perf_counter()

            temp_list = []

            for i, response in enumerate(responses):
                if response and (len(final_uids) < k):
                    final_uids.append(candidate_uids[uid + i])

                    temp_list.append(candidate_uids[uid + i])

                elif len(final_uids) >= k:
                    break
                else:
                    continue

            logger.info(f"Added uids: {temp_list} in {time.perf_counter() - t2:.2f}s")

            avg_num_list.append(len(temp_list))

            if len(final_uids) >= k:
                break

        else:
            continue

    sum_avg = 0
    try:
        sum_avg = sum(avg_num_list) / attempt_counter
    except:
        pass

    logger.info(
        f"Time to find all {len(final_uids)} uids: {time.perf_counter() - t0:.2f}s in {attempt_counter} attempts | Avg active UIDs per attempt: {sum_avg:.2f}"
    )

    # print({f"UID_{candidate_uid}": "Active" if candidate_uid in final_uids else "Inactive" for i, candidate_uid in enumerate(candidate_uids)})

    uids = (
        torch.tensor(final_uids)
        if len(final_uids) < k
        else torch.tensor(random.sample(final_uids, k))
    )

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


def corcel_parse_response(text):
    split = text.split('"')
    if len(split) == 3:
        ### Has quotes
        split = [x for x in split if x]

        if split:
            split = split[0]
        else:
            logger.info(f"Returning (X1) default text: {text}")
            return text
    elif len(split) == 1:
        split = split[0]
    elif len(split) > 3:
        split = [x for x in split if x]
        if len(split) > 0:
            split = split[0]
    else:
        logger.info(f"Split: {split}")
        logger.info(f"Returning (X2) default text: {text}")
        return text

    logger.info(f"Returning parsed text: {split}")
    return split


def call_openai(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    logger.info(f"OpenAI response object: {response}")
    response = response.choices[0].message.content
    if response:
        logger.info(f"Prompt generated with OpenAI: {response}")
    return response


def call_corcel(self, prompt):
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"{self.corcel_api_key}",
    }
    JSON = {
        "miners_to_query": 1,
        "top_k_miners_to_query": 160,
        "ensure_responses": True,
        "miner_uids": [],
        "messages": [
            {
                "role": "system",
                "content": prompt,
            }
        ],
        "model": "cortext-ultra",
        "stream": False,
        "top_p": 1.0,
        "temperature": 1,
        "max_tokens": 250,
        "seed": random.randint(0, 1_000_000),
    }

    logger.info(f"Using args: {JSON}")

    response = None

    try:
        response = requests.post(
            "https://api.corcel.io/cortext/text", json=JSON, headers=HEADERS, timeout=15
        )
        response = response.json()[0]["choices"][0]["delta"]["content"]
    except requests.exceptions.ReadTimeout as e:
        logger.info(
            f"Corcel request timed out after 15 seconds... falling back to OpenAI..."
        )

    if response:
        logger.info(f"Prompt generated with Corcel: {response}")

    return response


def generate_random_prompt_gpt(
    self,
    model="gpt-4",
    prompt="You are an image prompt generator. Your purpose is to generate a single one sentence prompt that can be fed into Dalle-3.",
):
    response = None

    ### Generate the prompt from corcel if we have an API key
    if self.corcel_api_key:
        try:
            response = call_corcel(self, prompt)
            if response:
                ### Parse response to remove quotes and also adapt the bug with corcel where the output is repeated N times
                response = corcel_parse_response(response)
                if response.startswith("{"):
                    response = None
        except Exception as e:
            logger.error(f"An unexpected error occurred calling corcel: {e}")
            logger.error(f"Falling back to OpenAI if available...")

    if not response:
        if self.openai_client:
            for _ in range(2):
                try:
                    response = call_openai(self.openai_client, model, prompt)
                except Exception as e:
                    logger.error(f"An unexpected error occurred calling OpenAI: {e}")
                    logger.error(f"Sleeping for 10 seconds and retrying once...")
                    time.sleep(10)

                if response:
                    break
        else:
            logger.warning(
                "Attempted to use OpenAI as a fallback but the OPENAI_API_KEY is not set."
            )

    ### Remove any double quotes from the output
    if response:
        response = response.replace('"', "")

    return response


def generate_followup_prompt_gpt(
    self,
    prompt,
    model="gpt-4",
    followup_prompt="An image has now been generated from your first prompt. What is a second instruction that can be applied to this generated image?",
):
    ### Update this for next week. Combine this and the method above.
    messages = [
        {"role": "system", "content": "You are an image prompt generator."},
        {"role": "assistant", "content": f"{prompt}"},
        {
            "role": "user",
            "content": f"{followup_prompt}",
        },
    ]

    for _ in range(2):
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            new_prompt = response.choices[0].message.content
            logger.info(f"I2I prompt is {new_prompt}")
            return new_prompt

        except Exception as e:
            logger.error(f"Error when calling OpenAI: {e}")
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
        tags.append(str(fn.name))

    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "alchemy", "reward", "netuid", "wandb")
    }
    wandb_config["alchemy"].pop("full_path", None)

    if not os.path.exists(WANDB_VALIDATOR_PATH):
        os.makedirs(WANDB_VALIDATOR_PATH, exist_ok=True)

    project = "ImageAlchemyTest"

    if self.config.netuid == 26:
        project = "ImageAlchemy"

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=project,
        entity="tensoralchemists",
        config=wandb_config,
        dir=WANDB_VALIDATOR_PATH,
        tags=tags,
    )
    logger.success(f"Started a new wandb run called {self.wandb.name}.")


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    if self.wandb:
        try:
            self.wandb.finish()
        except:
            pass
    init_wandb(self, reinit=True)


def get_promptdb_backup(netuid, prompt_history=[], limit=1):
    api = wandb.Api()
    project = "ImageAlchemy" if netuid == 26 else "ImageAlchemyTest"
    runs = api.runs(f"tensoralchemists/{project}")

    for run in runs:
        if len(prompt_history) >= limit:
            break
        if run.historyLineCount >= 100:
            history = run.history()
            if ("prompt_t2i" not in history.columns) or (
                "prompt_i2i" not in history.columns
            ):
                continue
            for i in range(0, len(history) - 1, 2):
                if len(prompt_history) >= limit:
                    break

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


def get_device_name(device: torch.device):
    """Returns name of GPU model"""
    try:
        if device.type == "cuda":
            # Fetch the device index and then get the device name
            device_name = torch.cuda.get_device_name(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            return device_name
        else:
            # Return 'CPU' as it does not have a specific name like GPUs do
            return "CPU"
    except Exception as e:
        logger.error(f"failed to get device name: {e}")
        return "n/a"
