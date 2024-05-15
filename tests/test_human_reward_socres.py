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


def test_human_rewards_ranking_increase():
    test_uid_index = (
        (
            self.moving_average_scores
            == torch.min(self.moving_average_scores[self.moving_average_scores > 0])
        )
        .nonzero()[0]
        .item()
    )
    previous_rank = ss.rankdata(self.moving_average_scores.tolist())[test_uid_index]
    scores = get_human_rewards(
        self,
        self.moving_average_scores,
        mock=True,
        mock_winner=self.hotkeys[test_uid_index],
    )
    current_rank = ss.rankdata(scores.tolist())[test_uid_index]
    assert current_rank > previous_rank


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


def test_human_rewards_ranking_decrease():
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
    for i in range(0, 10000):
        scores = self.moving_average_scores
        scores = get_human_rewards(
            self,
            scores,
            mock=True,
            mock_winner=self.hotkeys[mock_winner_index],
            mock_loser=self.hotkeys[mock_loser_index],
        )
    current_rank = ss.rankdata(scores.tolist())[mock_loser_index]
    assert current_rank < previous_rank
