import sys

sys.path.append("/home/ubuntu/ImageAlchemy/")

import pytest
import torch
from neurons.protocol import ImageGeneration
from neurons.validator.reward import ModelDiversityRewardModel
from neurons.validator.utils import get_promptdb_backup

reward_model = ModelDiversityRewardModel()
prompt_history_db = get_promptdb_backup(netuid = 25, limit = 10)

@pytest.mark.parametrize("prompt", prompt_history_db)
def test_synapse_custom(prompt):
    synapse_custom = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=-1,
        model_type="alchemy",
    )

    responses = [reward_model.generate_image(synapse_custom)]
    rewards = reward_model.get_rewards(responses, rewards = torch.zeros(len(responses)), synapse = synapse_custom)
    assert rewards[0].item() == 1

@pytest.mark.parametrize("prompt", prompt_history_db)
def test_synapse_wrong_seed(prompt):
    synapse_custom = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=-1,
        model_type="alchemy",
    )

    synapse_wrong_seed = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=3,
        model_type="alchemy",
    )
    
    responses = [reward_model.generate_image(synapse_wrong_seed)]
    rewards = reward_model.get_rewards(responses, rewards = torch.zeros(len(responses)), synapse = synapse_custom)
    
    assert rewards[0].item() != 1

@pytest.mark.parametrize("prompt", prompt_history_db)
def test_synapse_low_steps(prompt):
    synapse_low_steps = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        steps=10,
        model_type="alchemy",
    )

    synapse_custom = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=-1,
        model_type="alchemy",
    )

    responses = [reward_model.generate_image(synapse_low_steps)]
    rewards = reward_model.get_rewards(responses, rewards = torch.zeros(len(responses)), synapse = synapse_custom)
    
    assert rewards[0].item() != 1