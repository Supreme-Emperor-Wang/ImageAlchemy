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

from reward import RewardModelType


@dataclass
class EventSchema:
    images: List
    task_type: str
    block: float
    uids: List[int]
    hotkeys: List[str]
    prompt_t2i: str
    prompt_i2i: str
    step_length: float

    # Reward data
    rewards: List[float]
    blacklist_filter: Optional[List[float]]
    nsfw_filter: Optional[List[float]]
    diversity_reward_model: Optional[List[float]]
    image_reward_model: Optional[List[float]]
    human_reward_model: Optional[List[float]]

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
            prompt_t2i=event_dict["prompt_t2i"],
            prompt_i2i=event_dict["prompt_i2i"],
            step_length=event_dict["step_length"],
            images=event_dict["images"],
            rewards=event_dict["rewards"],
            **rewards,
            set_weights=None,
        )
