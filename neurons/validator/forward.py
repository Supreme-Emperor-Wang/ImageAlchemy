import copy
import os
import random
import time
from dataclasses import asdict
from datetime import datetime

import torch
import torchvision.transforms as T
from event import EventSchema
from loguru import logger
from neurons.constants import FOLLOWUP_TIMEOUT, MOVING_AVERAGE_ALPHA
from neurons.protocol import ImageGeneration
from neurons.utils import output_log, sh
from utils import ttl_get_block

import bittensor as bt
import wandb

transform = T.Compose([T.PILToTensor()])


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
    output_log(f"{sh('Info')} -> {' | '.join(args_list)}", color_key="m")
    output_log(
        f"{sh('UIDs')} -> {' | '.join([str(uid) for uid in uids.tolist()])}",
        color_key="y",
    )

    validator_info = self.get_validator_info()
    output_log(
        f"{sh('Stats')} -> Block: {validator_info['block']} | Stake: {validator_info['stake']:.2f} | Rank: {validator_info['rank']:.2f} | VTrust: {validator_info['vtrust']:.2f} | Dividends: {validator_info['dividends']:.2f} | Emissions: {validator_info['emissions']:.2f}",
        color_key="c",
    )
    responses = self.loop.run_until_complete(
        self.dendrite(
            axons,
            synapse,
            timeout=self.query_timeout,
        )
    )

    self.stats.total_requests += 1
    event = {"task_type": task_type}

    start_time = time.time()

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received {len(responses)} response(s): {responses}")
    # Save images for manual validator
    if not self.config.alchemy.disable_manual_validator:
        bt.logging.info(f"Saving images")
        i = 0
        for r in responses:
            for image in r.images:
                T.transforms.ToPILImage()(bt.Tensor.deserialize(image)).save(
                    f"neurons/validator/images/{i}.png"
                )
                time.sleep(5)
                i = i + 1

        bt.logging.info(f"Saving prompt")
        with open("neurons/validator/images/prompt.txt", "w") as f:
            f.write(prompt)
            time.sleep(5)
    # Initialise rewards tensor
    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(
        self.device
    )

    for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
        reward_i, reward_i_normalized = reward_fn_i.apply(responses, rewards)
        rewards += weight_i * reward_i_normalized.to(self.device)
        event[reward_fn_i.name] = reward_i.tolist()
        event[reward_fn_i.name + "_normalized"] = reward_i_normalized.tolist()
        bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

    for masking_fn_i in self.masking_functions:
        mask_i, mask_i_normalized = masking_fn_i.apply(responses, rewards)
        rewards *= mask_i_normalized.to(self.device)
        event[masking_fn_i.name] = mask_i.tolist()
        event[masking_fn_i.name + "_normalized"] = mask_i_normalized.tolist()
        bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())
    
    if not self.config.alchemy.disable_manual_validator:
        bt.logging.info(f"Waiting for manual vote")
        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) < 10:
            # breakpoint()
            if os.path.exists("neurons/validator/images/vote.txt"):
                # loop until vote is successfully saved
                while open("neurons/validator/images/vote.txt", "r").read() == "":
                    continue

                reward_i = open("neurons/validator/images/vote.txt", "r").read()
                bt.logging.info(f"Received manual vote for UID {int(reward_i) - 1}")
                reward_i_normalized: torch.FloatTensor = torch.zeros(
                    len(rewards), dtype=torch.float32
                ).to(self.device)
                reward_i_normalized[int(reward_i) - 1] = 1.0

                rewards += self.reward_weights[-1] * reward_i_normalized.to(self.device)

                if not self.config.alchemy.disable_log_rewards:
                    event["human_reward_model"] = reward_i_normalized.tolist()
                    event[
                        "human_reward_model_normalized"
                    ] = reward_i_normalized.tolist()

                break
        else:
            bt.logging.info("No manual vote received")

    # Delete contents of images folder except for black image
    for file in os.listdir("neurons/validator/images"):
        os.remove(
            f"neurons/validator/images/{file}"
        ) if file != "black.png" else "_"

    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, rewards
    ).to(self.device)

    bt.logging.trace(f"Scattered rewards: {scattered_rewards}")

    try:
        bt.logging.trace(
            f"Before: Moving averaged scores: {self.moving_averaged_scores}"
        )
    except:
        pass

    self.moving_averaged_scores: torch.FloatTensor = (
        MOVING_AVERAGE_ALPHA * scattered_rewards
        + (1 - MOVING_AVERAGE_ALPHA) * self.moving_averaged_scores.to(self.device)
    )
    bt.logging.trace(f"After: Moving averaged scores: {self.moving_averaged_scores}")
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
                    response.images[0]
                    if (response.images != []) and (reward != 0)
                    else []
                    for response, reward in zip(responses, rewards.tolist())
                ],
                "rewards": rewards.tolist(),
            }
        )

    except Exception as err:
        bt.logging.error("Error updating event dict", str(err))

    bt.logging.debug(f"Events: {str(event)}")
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
    self.wandb.log(asdict(wandb_event))
    return event
