import sys

sys.path.append("/home/ubuntu/ImageAlchemy/")

import torch
from neurons.protocol import ImageGeneration
from neurons.validator.reward import BlacklistFilter, ModelDiversityRewardModel

import bittensor as bt

diversity_reward_model = ModelDiversityRewardModel()
blacklist_reward_model = BlacklistFilter()


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
    assert (rewards[0].item() == 1) and (rewards[1].item() == 0)


def check_incorrect_image_size():
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
    assert (rewards[0].item() == 1) and (rewards[1].item() == 0)
