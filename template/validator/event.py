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

from dataclasses import dataclass
from typing import Any, List, Optional

from .reward import RewardModelType

import bittensor as bt
import wandb


@dataclass
class EventSchema:
    images: List
    # wandb_images: List
    # completion_times: List[float]  # List of completion times for a given prompt
    # completion_status_messages: List[
    #     str
    # ]  # List of completion status messages for a given prompt
    # completion_status_codes: List[
    #     str
    # ]  # List of completion status codes for a given prompt
    task_type: str  # Task type, e.g. 'summary', 'question'
    block: float  # Current block at given step
    # gating_loss: float  # Gating model loss for given step
    uids: List[int]  # Queried uids
    hotkeys: List[str]
    prompt: str  # Prompt text string
    step_length: float  # Elapsed time between the beginning of a run step to the end of a run step
    # best: str  # Best completion for given prompt

    # Reward data
    rewards: List[float]  # Reward vector for given step
    blacklist_filter: Optional[List[float]]  # Output vector of the blacklist filter
    nsfw_filter: Optional[List[float]]  # Output vector of the nsfw filter
    diversity_reward_model: Optional[
        List[float]
    ]  # Output vector of the diversity reward model
    image_reward_model: Optional[List[float]]  # Output vector of the image reward model
    human_reward_model: Optional[List[float]]

    # Weights data
    set_weights: Optional[List[List[float]]]

    @staticmethod
    def from_dict(event_dict: dict, disable_log_rewards: bool) -> "EventSchema":
        """Converts a dictionary to an EventSchema object."""
        rewards = {
            "blacklist_filter": event_dict.get(RewardModelType.blacklist.value),
            "nsfw_filter": event_dict.get(RewardModelType.nsfw.value),
            "diversity_reward_model": event_dict.get(RewardModelType.diversity.value),
            "image_reward_model": event_dict.get(RewardModelType.image.value),
            "human_reward_model": event_dict.get(RewardModelType.human.value),
        }
        return EventSchema(
            task_type=event_dict["task_type"],
            block=event_dict["block"],
            uids=event_dict["uids"],
            hotkeys=event_dict["hotkeys"],
            prompt=event_dict["prompt"],
            step_length=event_dict["step_length"],
            images=event_dict["images"],
            rewards=event_dict["rewards"],
            **rewards,
            set_weights=None,
        )
