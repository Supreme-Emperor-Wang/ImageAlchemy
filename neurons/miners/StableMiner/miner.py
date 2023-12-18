from base import BaseMiner
import bittensor as bt
from utils import output_log, warm_up
import torch


class StableMiner(BaseMiner):
    def __init__(self):
        #### Load the model
        (
            self.t2i_model,
            self.i2i_model,
            self.safety_checker,
            self.processor,
        ) = self.load_models()

        #### Optimize model
        if self.config.miner.optimize:
            self.t2i_model.unet = torch.compile(
                self.t2i_model.unet, mode="reduce-overhead", fullgraph=True
            )

            #### Warm up model
            output_log("Warming up model with compile...")
            warm_up(self.t2i_model, self.t2i_args)

        ### Set up mapping for the different synapse types
        self.mapping = {
            "text_to_image": {"args": self.t2i_args, "model": self.t2i_model},
            "image_to_image": {"args": self.i2i_args, "model": self.i2i_model},
        }

        #### Serve the axon
        self.start_axon()

        #### Start the miner loop
        self.loop()
