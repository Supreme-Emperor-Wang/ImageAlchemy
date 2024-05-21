import sys

sys.path.append("/home/ubuntu/ImageAlchemy/")


import copy
import uuid
from io import BytesIO

import pytest
import torch
import torchvision.transforms as T
from neurons.constants import DEV_URL, PROD_URL
from neurons.utils import post_batch
from neurons.validator.config import add_args, check_config, config
from neurons.validator.forward import post_moving_averages
from neurons.validator.reward import HumanValidationRewardModel
from neurons.validator.weights import post_weights

import bittensor as bt


class Neuron:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def config(cls):
        return config(cls)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    def __init__(self):
        self.config = Neuron.config()
        self.config.wallet.name = "validator"
        self.config.wallet.hotkey = "default"
        self.check_config(self.config)

    def load_state(self, path, moving_average_scores, device, metagraph):
        r"""Load hotkeys and moving average scores from filesystem."""
        state_dict = torch.load(f"{path}/model.torch")
        neuron_weights = torch.tensor(state_dict["neuron_weights"])

        has_nans = torch.isnan(neuron_weights).any()
        has_infs = torch.isinf(neuron_weights).any()

        # Check to ensure that the size of the neruon weights matches the metagraph size.
        if neuron_weights.shape < (metagraph.n,):
            moving_average_scores[: len(neuron_weights)] = neuron_weights.to(device)

        # Check for nans in saved state dict
        elif not any([has_nans, has_infs]):
            moving_average_scores = neuron_weights.to(device)

        # Zero out any negative scores
        for i, average in enumerate(moving_average_scores):
            if average < 0:
                moving_average_scores[i] = 0
        return moving_average_scores


def get_netuid(network):
    if network == "test":
        return 25
    else:
        return 26


def get_url(network):
    api_url = DEV_URL if network == "test" else PROD_URL
    return api_url


def get_args(network, neuron):
    neuron.config.netuid = get_netuid(network)
    neuron.config.subtensor.network = network
    subtensor = bt.subtensor(config=neuron.config)
    metagraph = bt.metagraph(
        netuid=neuron.config.netuid, network=neuron.config.subtensor.network, sync=False
    )
    metagraph.sync(subtensor=subtensor)
    moving_averages = torch.zeros(metagraph.n).to(neuron.config.device)
    moving_averages = neuron.load_state(
        neuron.config.alchemy.full_path,
        moving_averages,
        neuron.config.device,
        metagraph,
    )
    api_url = get_url(network)
    hotkeys = copy.deepcopy(metagraph.hotkeys)

    return metagraph, moving_averages, api_url, hotkeys


def create_dummy_batches(metagraph):
    uids = [0, 1, 2, 3, 4, 5]

    batches = [
        {
            "batch_id": str(uuid.uuid4()),
            "validator_hotkey": "5Cv9sBYUsif5rgkUbZfaQAVzBbnb9rZAUTNsyx8Eitzk9MA9",
            "prompt": "test",
            "nsfw_scores": [1 for _ in uids],
            "blacklist_scores": [1 for _ in uids],
            "miner_hotkeys": [metagraph.hotkeys[uid] for uid in uids],
            "miner_coldkeys": [metagraph.coldkeys[uid] for uid in uids],
            "computes": [
                T.transforms.ToPILImage()(
                    torch.full([3, 1024, 1024], 255, dtype=torch.float)
                ).save(BytesIO(), format="PNG")
                for _ in uids
            ],
            "should_drop_entries": [0 for uid in uids],
        }
    ]
    return batches


neuron = Neuron()


@pytest.mark.parametrize("network", ["test", "finney"])
def test_post_moving_averages(network):
    _, moving_averages, api_url, hotkeys = get_args(network, neuron)
    response = post_moving_averages(api_url, hotkeys, moving_averages)
    assert response == True


@pytest.mark.parametrize("network", ["test", "finney"])
def test_post_weights(network):
    _, moving_averages, api_url, hotkeys = get_args(network, neuron)
    raw_weights = torch.nn.functional.normalize(moving_averages, p=1, dim=0)
    response = post_weights(api_url, hotkeys, raw_weights)
    assert response.status_code == 200


@pytest.mark.parametrize("network", ["test", "finney"])
def test_submit_batch(network):
    metagraph, _, api_url, _ = get_args(neuron, network)
    dummy_batch = create_dummy_batches(metagraph)
    response = post_batch(api_url, dummy_batch)
    assert response.status_code == 200


@pytest.mark.parametrize("network", ["test", "finney"])
def test_get_votes(network):
    metagraph, _, api_url, _ = get_args(network, neuron)
    hv_reward_model = HumanValidationRewardModel(metagraph, api_url)
    human_voting_scores = hv_reward_model.get_votes(api_url)
    assert human_voting_scores.status_code == 200
    assert human_voting_scores.json() != {}
