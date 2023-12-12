import copy
import os
import time
from dataclasses import asdict

import template
import torch
import torchvision.transforms as T
from loguru import logger
from template.validator.event import EventSchema
from template.validator.utils import ttl_get_block

import bittensor as bt
import wandb

transform = T.Compose([T.PILToTensor()])


def run_step(self, prompt, axons, uids, task_type="text_to_image", image=None):
    responses = self.loop.run_until_complete(
        self.dendrite(
            axons,
            template.protocol.ImageGeneration(
                generation_type=task_type,
                prompt=prompt,
                prompt_image=image,
            )
            if image is not None
            else template.protocol.ImageGeneration(
                generation_type=task_type,
                prompt=prompt,
            ),
            timeout=self.config.neuron.timeout,
        )
    )
    event = {"task_type": task_type}

    start_time = time.time()

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received response: {responses}")

    if not self.config.neuron.disable_manual_validator:
        # Save images for manual validator
        bt.logging.info(f"Saving images")
        i = 0
        for r in responses:
            for image in r.images:
                T.transforms.ToPILImage()(bt.Tensor.deserialize(image)).save(
                    f"neurons/validator/images/{i}.png"
                )
                i = i + 1

        bt.logging.info(f"Saving prompt")
        with open("neurons/validator/images/prompt.txt", "w") as f:
            f.write(prompt)

    # Initialise rewards tensor
    rewards: torch.FloatTensor = torch.ones(len(responses), dtype=torch.float32).to(
        self.device
    )
    for masking_fn_i in self.masking_functions:
        mask_i, mask_i_normalized = masking_fn_i.apply(responses, rewards)
        rewards *= mask_i_normalized.to(self.device)
        if not self.config.neuron.disable_log_rewards:
            event[masking_fn_i.name] = mask_i.tolist()
            event[masking_fn_i.name + "_normalized"] = mask_i_normalized.tolist()
        bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

    for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
        reward_i, reward_i_normalized = reward_fn_i.apply(responses, rewards)
        rewards += weight_i * reward_i_normalized.to(self.device)
        if not self.config.neuron.disable_log_rewards:
            event[reward_fn_i.name] = reward_i.tolist()
            event[reward_fn_i.name + "_normalized"] = reward_i_normalized.tolist()
        bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

    if not self.config.neuron.disable_manual_validator:
        bt.logging.info(f"Waiting for manual vote")
        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) < 10:
            if os.path.exists("neurons/validator/images/vote.txt"):
                # loop until vote is successfully saved
                while open("neurons/validator/images/vote.txt", "r").read() == "":
                    continue

                reward_i = open("neurons/validator/images/vote.txt", "r").read()
                bt.logging.info("Received manual vote")
                bt.logging.info("MANUAL VOTE = " + reward_i)
                reward_i_normalized: torch.FloatTensor = torch.zeros(
                    len(rewards), dtype=torch.float32
                ).to(self.device)
                reward_i_normalized[int(reward_i) - 1] = 1.0

                rewards += self.reward_weights[-1] * reward_i_normalized.to(self.device)

                if not self.config.neuron.disable_log_rewards:
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

    # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
    # shape: [ metagraph.n ]
    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, rewards
    ).to(self.device)

    # Update moving_averaged_scores with rewards produced by this step.
    # shape: [ metagraph.n ]
    alpha: float = self.config.neuron.moving_average_alpha
    self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (
        1 - alpha
    ) * self.moving_averaged_scores.to(self.device)

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

    bt.logging.debug("event:", str(event))
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    if not self.config.wandb.off:
        wandb_event = copy.deepcopy(event)

        if self.config.wandb.compress:
            file_type = "jpg"
        else:
            file_type = "png"

        for e, image in enumerate(wandb_event["images"]):
            if image == []:
                wandb_event["images"][e] = wandb.Image(
                    torch.full([3, 1024, 1024], 255, dtype=torch.float),
                    caption=prompt,
                    file_type=file_type,
                )
            else:
                wandb_event["images"][e] = wandb.Image(
                    bt.Tensor.deserialize(image),
                    caption=prompt,
                    file_type=file_type,
                )

        wandb_event = EventSchema.from_dict(
            wandb_event, self.config.neuron.disable_log_rewards
        )
        self.wandb.log(asdict(wandb_event))
    return event
