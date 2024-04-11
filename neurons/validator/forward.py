import base64
import copy
import os
import random
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from io import BytesIO

import requests
import torch
import torchvision.transforms as T
from event import EventSchema
from loguru import logger
from neurons.constants import HVB_MAINNET_IP, MOVING_AVERAGE_ALPHA, MOVING_AVERAGE_BETA
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
            self.miner_query_history_duration[self.metagraph.axons[uid].hotkey] = (
                time.perf_counter()
            )
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
    event = {"task_type": task_type}

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
    # Initialise rewards tensor
    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(
        self.device
    )

    for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
        reward_i, reward_i_normalized = reward_fn_i.apply(responses, rewards)
        rewards += weight_i * reward_i_normalized.to(self.device)
        event[reward_fn_i.name] = reward_i.tolist()
        event[reward_fn_i.name + "_normalized"] = reward_i_normalized.tolist()
        print(str(reward_fn_i.name), reward_i_normalized.tolist())
    for masking_fn_i in self.masking_functions:
        mask_i, mask_i_normalized = masking_fn_i.apply(responses, rewards)
        rewards *= mask_i_normalized.to(self.device)
        event[masking_fn_i.name] = mask_i.tolist()
        event[masking_fn_i.name + "_normalized"] = mask_i_normalized.tolist()
        print(str(masking_fn_i.name), mask_i_normalized.tolist())

    if not self.config.alchemy.disable_manual_validator:
        print(f"Waiting {self.manual_validator_timeout} seconds for manual vote...")
        start_time = time.perf_counter()

        received_vote = False

        while (time.perf_counter() - start_time) < self.manual_validator_timeout:
            time.sleep(1)
            # If manual vote received
            if os.path.exists("neurons/validator/images/vote.txt"):
                # loop until vote is successfully saved
                while open("neurons/validator/images/vote.txt", "r").read() == "":
                    time.sleep(0.05)
                    continue

                try:
                    reward_i = (
                        int(open("neurons/validator/images/vote.txt", "r").read()) - 1
                    )
                except Exception as e:
                    print(f"An unexpected error occurred parsing the vote: {e}")
                    break

                ### There is a small possibility that not every miner queried will respond.
                ### If 12 are queried, but only 10 respond, we need to handle the error if
                ### the user selects the 11th or 12th image (which don't exist)
                if reward_i >= len(rewards):
                    print(
                        f"Received invalid vote for Image {reward_i+1}: it doesn't exist."
                    )
                    break

                print(f"Received manual vote for Image {reward_i+1}")

                ### Set to true so we don't normalize the rewards later
                received_vote = True

                reward_i_normalized: torch.FloatTensor = torch.zeros(
                    len(rewards), dtype=torch.float32
                ).to(self.device)
                reward_i_normalized[reward_i] = 1.0
                rewards += self.reward_weights[-1] * reward_i_normalized.to(self.device)
                if not self.config.alchemy.disable_log_rewards:
                    event["human_reward_model"] = reward_i_normalized.tolist()
                    event["human_reward_model_normalized"] = (
                        reward_i_normalized.tolist()
                    )

                break

        if not received_vote:
            delta = 1 - self.reward_weights[-1]
            if delta != 0:
                rewards /= delta
            else:
                print("The reward weight difference was 0 which is unexpected.")
            print("No valid vote was received")

        # Delete contents of images folder except for black image
        if os.path.exists("neurons/validator/images"):
            for file in os.listdir("neurons/validator/images"):
                (
                    os.remove(f"neurons/validator/images/{file}")
                    if file != "black.png"
                    else "_"
                )

    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, rewards
    ).to(self.device)

    self.moving_averaged_scores: torch.FloatTensor = (
        MOVING_AVERAGE_ALPHA * scattered_rewards
        + (1 - MOVING_AVERAGE_ALPHA) * self.moving_averaged_scores.to(self.device)
    )

    print(f"{self.moving_averaged_scores}")

    max_retries = 3
    backoff = 2
    print("Querying for human votes...")
    for attempt in range(0, max_retries):
        try:
            api_host = f"{HVB_MAINNET_IP}:5000/api"

            human_voting_scores = requests.get(
                f"http://{api_host}/get_votes", timeout=2
            )

            if (human_voting_scores.status_code != 200) and (attempt == max_retries):

                print(
                    f"Failed to retrieve the human validation bot votes {attempt+1} times. Skipping until the next step."
                )
                break

            elif (human_voting_scores.status_code != 200) and (attempt != max_retries):

                continue

            else:

                human_voting_bot_round_scores = human_voting_scores.json()

                human_voting_bot_scores = {}

                for inner_dict in human_voting_bot_round_scores.values():
                    for key, value in inner_dict.items():
                        if key in human_voting_bot_scores:
                            human_voting_bot_scores[key] += value
                        else:
                            human_voting_bot_scores[key] = value

                human_voting_bot_scores = torch.tensor(
                    [
                        (
                            human_voting_bot_scores[key]
                            if key in human_voting_bot_scores.keys()
                            else 0
                        )
                        for key in self.hotkeys
                    ]
                ).to(self.device)

                if human_voting_bot_scores.sum() == 0:

                    continue

                else:
                    human_voting_bot_scores = torch.nn.functional.normalize(
                        human_voting_bot_scores
                    )

                    self.moving_averaged_scores: (
                        torch.FloatTensor
                    ) = MOVING_AVERAGE_BETA * (0.02 * human_voting_bot_scores) + (
                        1 - MOVING_AVERAGE_BETA
                    ) * self.moving_averaged_scores.to(
                        self.device
                    )
                    break

        except Exception as e:
            print(
                f"Encountered the following error retrieving the manual validator scores: {e}. Retrying in {backoff} seconds."
            )

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
