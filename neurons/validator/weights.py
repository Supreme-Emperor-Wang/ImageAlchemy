# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation
from loguru import logger

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Utils for weights setting on chain.

import neurons.validator as validator
import pandas as pd
import requests
import torch
from neurons.validator.utils import ttl_get_block

import bittensor as bt


def set_weights(self):
    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(self.moving_averaged_scores, p=1, dim=0)

    try:
        response = requests.post(
            f"{self.api_url}/validator/weights",
            json={
                "weights": {
                    hotkey: moving_average.item()
                    for hotkey, moving_average in zip(
                        self.hotkeys, self.moving_averaged_scores
                    )
                }
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code != 200:
            logger.info("Error logging weights to the Weights API")
        else:
            logger.info("Successfully logged weights to the Weights API")
    except:
        logger.info("Error logging weights to the Weights API")

    # print("raw_weights", raw_weights)
    # print("top10 values", raw_weights.sort()[0])
    # print("top10 uids", raw_weights.sort()[1])
    # Process the raw weights to final_weights via subtensor limitations.
    (
        processed_weight_uids,
        processed_weights,
    ) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=self.metagraph.uids.to("cpu"),
        weights=raw_weights.to("cpu"),
        netuid=self.config.netuid,
        subtensor=self.subtensor,
        metagraph=self.metagraph,
    )
    logger.info("processed_weights", processed_weights)
    logger.info("processed_weight_uids", processed_weight_uids)
    # Set the weights on chain via our subtensor connection.
    self.subtensor.set_weights(
        wallet=self.wallet,
        netuid=self.config.netuid,
        uids=processed_weight_uids,
        weights=processed_weights,
        wait_for_finalization=False,
        version_key=validator.__spec_version__,
    )
