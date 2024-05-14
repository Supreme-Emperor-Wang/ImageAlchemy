import sys

sys.path.append("/home/ubuntu/ImageAlchemy/")

import os

import scipy.stats as ss
import torch
from dotenv import load_dotenv
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


def test_human_rewards_score_increase():
    test_uid_index = (
        (
            self.moving_averaged_scores
            == torch.min(self.moving_averaged_scores[self.moving_averaged_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    previous_score = self.moving_averaged_scores[test_uid_index]
    scores = get_human_rewards(
        self,
        self.moving_averaged_scores,
        mock=True,
        mock_winner=self.hotkeys[test_uid_index],
    )
    current_score = scores[[test_uid_index]]
    assert current_score > previous_score

def test_human_rewards_ranking_increase():
    test_uid_index = (
        (
            self.moving_averaged_scores
            == torch.min(self.moving_averaged_scores[self.moving_averaged_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    previous_rank = ss.rankdata(self.moving_averaged_scores.tolist())[test_uid_index]
    scores = get_human_rewards(
        self,
        self.moving_averaged_scores,
        mock=True,
        mock_winner=self.hotkeys[test_uid_index],
    )
    current_rank = ss.rankdata(scores.tolist())[test_uid_index]
    assert current_rank > previous_rank
