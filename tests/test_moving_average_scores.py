import sys

sys.path.append("/home/ubuntu/ImageAlchemy/")

import os

import scipy.stats as ss
import torch
from dotenv import load_dotenv
from neurons.validator.forward import update_moving_averages
from neurons.validator.reward import get_human_rewards
from neurons.validator.validator import StableValidator

load_dotenv()

sys.argv += [
    "--netuid",
    "25",
    "--subtensor.network",
    "test",
    "--wallet.name",
    "validator",
    "--wallet.hotkey default",
    "--alchemy.disable_manual_validator",
]
self = StableValidator()
self.load_state()

def test_non_zero_moving_averages():
    rewards = torch.tensor([0.6522690057754517, 0.7715857625007629, 0.7447815537452698, 0.7694319486618042, 0.03637188673019409, 0.7205913066864014, 0.0890098512172699, 0.7766138315200806, 0.0, 0.0]).to(self.device)
    uids = torch.tensor([39,34,37,35,40,38,36,33,22,58]).to(self.device)

    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, rewards
    ).to(self.device)

    update_moving_averages(self, scattered_rewards)
    
    assert self.moving_averaged_scores.sum().item() != 0

def test_large_rewards():
    
    test_uid_index = 39   
    uids =  torch.tensor([test_uid_index]).to(self.device)
    rewards = torch.tensor([self.moving_averaged_scores[test_uid_index] * 2]).to(self.device)

    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, rewards
    ).to(self.device)

    previous_moving_average = self.moving_averaged_scores[test_uid_index]
    update_moving_averages(self, scattered_rewards)
    current_moving_average = self.moving_averaged_scores[test_uid_index]

    assert current_moving_average > previous_moving_average


def test_rewards_with_nans():
    rewards = torch.zeros(len(self.moving_averaged_scores)).to(self.device)
    rewards[0]= float('nan')
    update_moving_averages(self, rewards)
    torch.isnan(self.moving_averaged_scores).sum().item() == 0

def test_zero_rewards():
    rewards = torch.zeros(len(self.moving_averaged_scores)).to(self.device)
    previous_moving_averaged_scores_sum = self.moving_averaged_scores.sum()
    update_moving_averages(self, rewards)
    current_moving_averaged_scores_sum = self.moving_averaged_scores.sum()
    assert previous_moving_averaged_scores_sum >= current_moving_averaged_scores_sum

def test_ones_rewards():
    rewards = torch.ones(len(self.moving_averaged_scores)).to(self.device)
    previous_moving_averaged_scores_sum = self.moving_averaged_scores.sum()
    update_moving_averages(self, rewards)
    current_moving_averaged_scores_sum = self.moving_averaged_scores.sum()
    assert previous_moving_averaged_scores_sum < current_moving_averaged_scores_sum