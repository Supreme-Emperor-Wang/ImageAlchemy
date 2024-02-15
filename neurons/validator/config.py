import argparse
import os

from loguru import logger
from neurons.constants import EVENTS_RETENTION_SIZE

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
            config.alchemy.name,
        )
    )
    config.alchemy.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.alchemy.full_path):
        os.makedirs(config.alchemy.full_path, exist_ok=True)

    # Add custom event logger for the events.
    logger.level("EVENTS", no=38, icon="📝")
    logger.add(
        config.alchemy.full_path + "/" + "completions.log",
        rotation=EVENTS_RETENTION_SIZE,
        serialize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        level="EVENTS",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    )


def add_args(cls, parser):
    # Netuid Arg
    parser.add_argument("--netuid", type=int, help="Network netuid", default=1)

    parser.add_argument(
        "--alchemy.name",
        type=str,
        help="Trials for this validator go in validator.root / (wallet_cold - wallet_hot) / validator.name.",
        default="image_alchemy_validator",
    )
    parser.add_argument(
        "--alchemy.device",
        type=str,
        help="Device to run the validator on.",
        default="cuda:0",
    )
    parser.add_argument(
        "--alchemy.disable_manual_validator",
        action="store_true",
        help="If set, we run the manual validator",
        default=False,
    )
    parser.add_argument(
        "--alchemy.streamlit_port",
        type=int,
        help="Port number for streamlit app",
        default=None,
    )


def config(cls):
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
