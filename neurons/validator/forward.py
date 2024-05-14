import base64
import copy
import os
import random
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
import torch
import torchvision.transforms as T
from loguru import logger
from neurons.constants import MOVING_AVERAGE_ALPHA, MOVING_AVERAGE_BETA
from neurons.protocol import ImageGeneration
from neurons.utils import output_log, sh
from neurons.validator.event import EventSchema
from neurons.validator.reward import (
    filter_rewards,
    get_automated_rewards,
    get_human_rewards,
)
from neurons.validator.utils import ttl_get_block

import bittensor as bt
import wandb

transform = T.Compose([T.PILToTensor()])


def update_moving_averages(self, rewards):
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0) 
    self.moving_averaged_scores: torch.FloatTensor = MOVING_AVERAGE_ALPHA * rewards + (
        1 - MOVING_AVERAGE_ALPHA
    ) * self.moving_averaged_scores.to(self.device)


def run_step(self, prompt, axons, uids, task_type="text_to_image", image=None):
    time_elapsed = datetime.now() - self.stats.start_time

    output_log(
        f"{sh('Info')} -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')} | Elapsed {time_elapsed} | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f}",
        color_key="g",
    )
    output_log(
        f"{sh('Request')} -> Type: {task_type} | Total requests sent {self.stats.total_requests:,} | Timeouts {self.stats.timeouts:,}",
        color_key="c",
    )

    ### Set seed to -1 so miners will use a random seed by default
    synapse = (
        ImageGeneration(
            generation_type=task_type,
            prompt=prompt,
            prompt_image=image,
            seed=-1,
        )
        if image is not None
        else ImageGeneration(
            generation_type=task_type,
            prompt=prompt,
            seed=-1,
        )
    )

    output_log(
        f"{sh('Prompt')} -> {synapse.__dict__['prompt']}",
        color_key="y",
    )

    synapse_dict = {
        k: v
        for k, v in synapse.__dict__.items()
        if k
        in [
            "timeout",
            "height",
            "width",
        ]
    }
    args_list = [
        f"{k.capitalize()}: {f'{v:.2f}' if isinstance(v, float) else v}"
        for k, v in synapse_dict.items()
    ]

    responses = self.loop.run_until_complete(
        self.dendrite(
            axons,
            synapse,
            timeout=self.query_timeout,
        )
    )

    # Log query to hisotry
    try:
        for uid in uids:
            self.miner_query_history_duration[
                self.metagraph.axons[uid].hotkey
            ] = time.perf_counter()
        for uid in uids:
            self.miner_query_history_count[self.metagraph.axons[uid].hotkey] += 1
    except:
        print("Failed to log miner counts and histories")

    output_log(
        f"{sh('Miner Counts')} -> Max: {max(self.miner_query_history_count.values()):.2f} | Min: {min(self.miner_query_history_count.values()):.2f} | Mean: {sum(self.miner_query_history_count.values()) / len(self.miner_query_history_count.values()):.2f}",
        color_key="y",
    )

    responses_empty_flag = [1 if not response.images else 0 for response in responses]
    sorted_index = [
        item[0]
        for item in sorted(
            list(zip(range(0, len(responses_empty_flag)), responses_empty_flag)),
            key=lambda x: x[1],
        )
    ]
    uids = torch.tensor([uids[index] for index in sorted_index]).to(self.device)
    responses = [responses[index] for index in sorted_index]

    output_log(f"{sh('Info')} -> {' | '.join(args_list)}", color_key="m")
    output_log(
        f"{sh('UIDs')} -> {' | '.join([str(uid) for uid in uids.tolist()])}",
        color_key="y",
    )

    validator_info = self.get_validator_info()
    output_log(
        f"{sh('Stats')} -> Block: {validator_info['block']} | Stake: {validator_info['stake']:.4f} | Rank: {validator_info['rank']:.4f} | VTrust: {validator_info['vtrust']:.4f} | Dividends: {validator_info['dividends']:.4f} | Emissions: {validator_info['emissions']:.4f}",
        color_key="c",
    )

    self.stats.total_requests += 1

    start_time = time.time()

    # Log the results for monitoring purposes.
    try:
        formatted_responses = [
            {
                "negative_prompt": response.negative_prompt,
                "prompt_image": response.prompt_image,
                "num_images_per_prompt": response.num_images_per_prompt,
                "height": response.height,
                "width": response.width,
                "seed": response.seed,
                "steps": response.steps,
                "guidance_scale": response.guidance_scale,
                "generation_type": response.generation_type,
                "images": [image.shape for image in response.images],
            }
            for response in responses
        ]
        print(
            f"Received {len(responses)} response(s) for the prompt '{prompt}': {formatted_responses}"
        )
    except Exception as e:
        print(f"Failed to log formatted responses: {e}")

    # Save images for manual validator
    if not self.config.alchemy.disable_manual_validator:
        print(f"Saving images")
        for i, r in enumerate(responses):
            for image in r.images:
                T.transforms.ToPILImage()(bt.Tensor.deserialize(image)).save(
                    f"neurons/validator/images/{i}.png"
                )

        print(f"Saving prompt")
        with open("neurons/validator/images/prompt.txt", "w") as f:
            f.write(prompt)

    scattered_rewards, event, rewards = get_automated_rewards(
        self, responses, uids, task_type
    )

    scattered_rewards_adjusted = get_human_rewards(self, scattered_rewards)

    scattered_rewards_adjusted = filter_rewards(self, scattered_rewards_adjusted)

    update_moving_averages(self, scattered_rewards_adjusted)

    try:
        response = requests.post(
            f"{self.api_url}/validator/averages",
            json={
                "averages": {
                    hotkey: moving_average.item()
                    for hotkey, moving_average in zip(
                        self.hotkeys, self.moving_averaged_scores
                    )
                }
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code != 200:
            bt.logging.info("Error logging moving averages to the Averages API")
        else:
            bt.logging.info("Successfully logged moving averages to the Averages API")
    except:
        bt.logging.info("Error logging moving averages to the Averages API")

    try:
        for i, average in enumerate(self.moving_averaged_scores):
            if (self.metagraph.axons[i].hotkey in self.hotkey_blacklist) or (
                self.metagraph.axons[i].coldkey in self.coldkey_blacklist
            ):
                self.moving_averaged_scores[i] = 0

    except Exception as e:
        print(f"An unexpected error occurred (E1): {e}")

    try:
        # Log the step event.
        event.update(
            {
                "block": ttl_get_block(self),
                "step_length": time.time() - start_time,
                "prompt_t2i": prompt if task_type == "text_to_image" else None,
                "prompt_i2i": prompt if task_type == "image_to_image" else None,
                "uids": uids.tolist(),
                "hotkeys": [self.metagraph.axons[uid].hotkey for uid in uids],
                "images": [
                    (
                        response.images[0]
                        if (response.images != []) and (reward != 0)
                        else []
                    )
                    for response, reward in zip(responses, rewards.tolist())
                ],
                "rewards": rewards.tolist(),
                # "moving_averages": self.moving_averaged_scores
            }
        )
        event.update(validator_info)
    except Exception as err:
        print("Error updating event dict", str(err))

    try:
        should_drop_entries = []
        images = []
        for response, reward in zip(responses, rewards.tolist()):
            if (response.images != []) and (reward != 0):
                im_file = BytesIO()
                T.transforms.ToPILImage()(
                    bt.Tensor.deserialize(response.images[0])
                ).save(im_file, format="PNG")
                im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
                im_b64 = base64.b64encode(im_bytes)
                images.append(im_b64.decode())
                should_drop_entries.append(0)
            else:
                im_file = BytesIO()
                T.transforms.ToPILImage()(
                    torch.full([3, 1024, 1024], 255, dtype=torch.float)
                ).save(im_file, format="PNG")
                im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
                im_b64 = base64.b64encode(im_bytes)
                images.append(im_b64.decode())
                should_drop_entries.append(1)

        # Update batches to be sent to the human validation platform
        self.batches.append(
            {
                "batch_id": str(uuid.uuid4()),
                "validator_hotkey": str(self.wallet.hotkey.ss58_address),
                "prompt": prompt,
                "nsfw_scores": event["nsfw_filter"],
                "blacklist_scores": event["blacklist_filter"],
                "miner_hotkeys": [self.metagraph.hotkeys[uid] for uid in uids],
                "miner_coldkeys": [self.metagraph.coldkeys[uid] for uid in uids],
                "computes": images,
                "should_drop_entries": should_drop_entries,
            }
        )
    except Exception as e:
        print(f"An unexpected error occurred appending the batch: {e}")

    print(f"Events: {str(event)}")
    logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    wandb_event = copy.deepcopy(event)
    file_type = "png"

    def gen_caption(prompt, i):
        return f"{prompt}\n({event['uids'][i]} | {event['hotkeys'][i]})"

    for e, image in enumerate(wandb_event["images"]):
        wandb_img = (
            torch.full([3, 1024, 1024], 255, dtype=torch.float)
            if image == []
            else bt.Tensor.deserialize(image)
        )

        wandb_event["images"][e] = wandb.Image(
            wandb_img,
            caption=gen_caption(prompt, e),
            file_type=file_type,
        )
    wandb_event = EventSchema.from_dict(wandb_event)

    try:
        self.wandb.log(asdict(wandb_event))
        print("Logged event to wandb.")
    except Exception as e:
        print(f"Unable to log event to wandb due to the following error: {e}")

    return event
