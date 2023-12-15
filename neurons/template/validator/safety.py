from torch import nn
import torch
import torchvision.transforms as transforms
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    CLIPConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    PreTrainedModel,
)


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=False
        )

        self.concept_embeds = nn.Parameter(
            torch.ones(17, config.projection_dim), requires_grad=False
        )
        self.special_care_embeds = nn.Parameter(
            torch.ones(3, config.projection_dim), requires_grad=False
        )

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(
            torch.ones(3), requires_grad=False
        )
        self.transform = transforms.Compose([transforms.PILToTensor()])
