import argparse
import os

import torch
from loguru import logger
from reward import DefaultRewardFrameworkConfig

import bittensor as bt


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)
    # bt.wallet.check_config(config)
    # bt.subtensor.check_config(config)

    if config.mock:
        config.neuron.mock_reward_models = True
        config.neuron.mock_gating_model = True
        config.neuron.mock_dataset = True
        config.wallet._mock = True

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            config.neuron.full_path + "/" + "completions.log",
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def add_args(cls, parser):
    # Netuid Arg
    parser.add_argument(
        "--netuid", type=int, help="Prompting network netuid", default=1
    )

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name. ",
        default="core_prompting_validator",
    )
    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run the validator on.",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--neuron.disable_log_rewards",
        action="store_true",
        help="Disable all reward logging, suppresses reward functions and their values from being logged to wandb.",
        default=False,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.05,
    )
    
    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="How many miners to query for a single prompt.",
        default=12,
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="Override the default epoch length (how often we set weights).",
        default=5,
    )
    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="Follow up query timeout.",
        default=10,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )

    parser.add_argument(
        "--reward.diversity_model_weight",
        type=float,
        help="Weight for the diversity reward model",
        default=DefaultRewardFrameworkConfig.diversity_model_weight,
    )
    parser.add_argument(
        "--reward.image_model_weight",
        type=float,
        help="Weight for the image reward model",
        default=DefaultRewardFrameworkConfig.image_model_weight,
    )
    parser.add_argument(
        "--reward.human_model_weight",
        type=float,
        help="Weight for the human reward model",
        default=DefaultRewardFrameworkConfig.human_model_weight,
    )


def config(cls):
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
