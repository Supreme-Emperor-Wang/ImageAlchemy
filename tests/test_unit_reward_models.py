import torch
import pytest
import bittensor as bt


from neurons.protocol import ImageGeneration
from neurons.validator.reward import (
    BlacklistFilter,
    ModelDiversityRewardModel,
    NSFWRewardModel,
)

diversity_reward_model: ModelDiversityRewardModel = None
blacklist_reward_model: BlacklistFilter = None
nsfw_reward_model: NSFWRewardModel = None


@pytest.fixture(autouse=True, scope="session")
def setup() -> None:
    global diversity_reward_model, blacklist_reward_model, nsfw_reward_model

    diversity_reward_model = ModelDiversityRewardModel()
    blacklist_reward_model = BlacklistFilter()
    nsfw_reward_model = NSFWRewardModel()


def test_black_image():
    responses = [
        ImageGeneration(
            generation_type="text_to_image",
            seed=-1,
            model_type="alchemy",
            images=[
                bt.Tensor.serialize(torch.full([3, 1024, 1024], 254, dtype=torch.float))
            ],
        ),
        ImageGeneration(
            generation_type="text_to_image",
            seed=-1,
            model_type="alchemy",
            images=[
                bt.Tensor.serialize(torch.full([3, 1024, 1024], 0, dtype=torch.float))
            ],
        ),
    ]
    rewards = blacklist_reward_model.get_rewards(
        responses, rewards=torch.ones(len(responses))
    )
    assert rewards[0].item() == 1
    assert rewards[1].item() == 0


def test_incorrect_image_size():
    responses = [
        ImageGeneration(
            generation_type="text_to_image",
            seed=-1,
            model_type="alchemy",
            images=[
                bt.Tensor.serialize(torch.full([3, 1024, 1024], 254, dtype=torch.float))
            ],
        ),
        ImageGeneration(
            generation_type="text_to_image",
            seed=-1,
            model_type="alchemy",
            images=[
                bt.Tensor.serialize(torch.full([3, 100, 1024], 254, dtype=torch.float))
            ],
        ),
    ]
    rewards = blacklist_reward_model.get_rewards(
        responses, rewards=torch.ones(len(responses))
    )
    assert rewards[0].item() == 1
    assert rewards[1].item() == 0


def test_nsfw_image():
    synapse_nsfw = ImageGeneration(
        generation_type="text_to_image",
        seed=-1,
        model_type="alchemy",
        prompt="A girl with no clothes on.",
    )
    synapse_no_nsfw = ImageGeneration(
        generation_type="text_to_image",
        seed=-1,
        model_type="alchemy",
        prompt="A bird flying in the sky.",
    )
    responses = [
        diversity_reward_model.generate_image(synapse_nsfw),
        diversity_reward_model.generate_image(synapse_no_nsfw),
    ]
    rewards = nsfw_reward_model.get_rewards(
        responses, rewards=torch.ones(len(responses))
    )
    assert rewards[0].item() == 0
    assert rewards[1].item() == 1
