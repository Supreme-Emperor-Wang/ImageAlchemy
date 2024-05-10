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
    model_type: str

    # Reward data
    rewards: List[float]
    blacklist_filter: Optional[List[float]]
    nsfw_filter: Optional[List[float]]
    diversity_reward_model: Optional[List[float]]
    image_reward_model: Optional[List[float]]
    human_reward_model: Optional[List[float]]

    # Bittensor data
    stake: List[float]
    rank: List[float]
    vtrust: List[float]
    dividends: List[float]
    emissions: List[float]

    set_weights: Optional[List[List[float]]]

    @staticmethod
    def from_dict(event_dict: dict) -> "EventSchema":
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
            model_type=event_dict["model_type"],
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
            stake=event_dict["stake"],
            rank=event_dict["rank"],
            vtrust=event_dict["vtrust"],
            dividends=event_dict["dividends"],
            emissions=event_dict["emissions"],
            # moving_averages=event_dict["moving_averages"]
        )
