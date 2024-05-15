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

mock_loser_index = (
    (
        self.moving_average_scores
        == torch.max(self.moving_average_scores[self.moving_average_scores > 0])
    )
    .nonzero()[0]
    .item()
)
mock_winner_index = (
    (
        self.moving_average_scores
        == torch.min(self.moving_average_scores[self.moving_average_scores > 0])
    )
    .nonzero()[0]
    .item()
)
previous_rank = ss.rankdata(self.moving_average_scores.tolist())[mock_loser_index]
scores = self.moving_average_scores
for i in range(0, 10000):
    scores = get_human_rewards(
        self,
        scores,
        mock=True,
        mock_winner=self.hotkeys[mock_winner_index],
        mock_loser=self.hotkeys[mock_loser_index],
    )
    weights = torch.nn.functional.normalize(scores, p=1, dim=0)
    print(weights[mock_loser_index].item())
current_rank = ss.rankdata(scores.tolist())[mock_loser_index]


def test_human_rewards_score_increase():
    test_uid_index = (
        (
            self.moving_average_scores
            == torch.min(self.moving_average_scores[self.moving_average_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    previous_score = self.moving_average_scores[test_uid_index]
    scores = get_human_rewards(
        self,
        self.moving_average_scores,
        mock=True,
        mock_winner=self.hotkeys[test_uid_index],
    )
    current_score = scores[[test_uid_index]]
    assert current_score > previous_score


def test_human_weights_increase():
    test_uid_index = (
        (
            self.moving_average_scores
            == torch.min(self.moving_average_scores[self.moving_average_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    weights = torch.nn.functional.normalize(self.moving_average_scores, p=1, dim=0)
    previous_weight = weights[test_uid_index]
    scores = get_human_rewards(
        self,
        self.moving_average_scores,
        mock=True,
        mock_winner=self.hotkeys[test_uid_index],
    )
    current_weight = torch.nn.functional.normalize(scores, p=1, dim=0)[test_uid_index]
    assert current_weight > previous_weight


def test_human_rewards_score_decrease():
    reference_uid_index = 0
    mock_loser_index = (
        (
            self.moving_average_scores
            == torch.max(self.moving_average_scores[self.moving_average_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    mock_winner_index = (
        (
            self.moving_average_scores
            == torch.min(self.moving_average_scores[self.moving_average_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    previous_score_delta = (
        self.moving_average_scores[mock_loser_index]
        - self.moving_average_scores[reference_uid_index]
    )
    for i in range(0, 100):
        scores = self.moving_average_scores
        scores = get_human_rewards(
            self,
            scores,
            mock=True,
            mock_winner=self.hotkeys[mock_winner_index],
            mock_loser=self.hotkeys[mock_loser_index],
        )
    current_score_delta = scores[mock_loser_index] - scores[reference_uid_index]
    assert current_score_delta < previous_score_delta


def test_human_weights_decrease():
    mock_loser_index = (
        (
            self.moving_average_scores
            == torch.max(self.moving_average_scores[self.moving_average_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    mock_winner_index = (
        (
            self.moving_average_scores
            == torch.min(self.moving_average_scores[self.moving_average_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    previous_weight = torch.nn.functional.normalize(
        self.moving_average_scores, p=1, dim=0
    )[mock_loser_index]
    scores = get_human_rewards(
        self,
        self.moving_average_scores,
        mock=True,
        mock_winner=self.hotkeys[mock_winner_index],
        mock_loser=self.hotkeys[mock_loser_index],
    )
    current_weight = torch.nn.functional.normalize(scores, p=1, dim=0)[mock_loser_index]

    assert current_weight < previous_weight


self.background_timer.cancel()
