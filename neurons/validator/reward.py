import torch
from typing import List
from torch import nn
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import bittensor as bt
from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel, PreTrainedModel
import numpy as np
import torchvision.transforms as transforms
from utils import compare_to_set, calculate_mean_dissimilarity
import ImageReward as RM

# TODO Edit to suit ImageNet
class RewardModelType(Enum):
    diversity = "diversity_reward_model"
    image = "image_reward_model"
    blacklist = "blacklist_filter"
    nsfw = "nsfw_filter"

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)
        self.transform = transforms.Compose([
            transforms.PILToTensor()
        ])

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": [], "bad_score": 0.0}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 1.0 # multiplier


            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - (concept_threshold * adjustment), 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - (concept_threshold * adjustment), 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    print('bad concept', concept_idx, result_img["concept_scores"][concept_idx])
                    result_img["bad_concepts"].append(concept_idx)
                    result_img['bad_score'] += result_img["concept_scores"][concept_idx]

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 and res['bad_score'] > 0.01 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if torch.is_tensor(images) or torch.is_tensor(images[0]):
                    images[idx] = torch.zeros_like(images[idx])  # black image
                else:
                    # images[idx] is a PIL image, so we can't use .shape, convert using transform
                    try:
                        images[idx] = np.zeros(transform(images[idx]).shape)  # black image
                    except:
                        images[idx] = np.zeros((512, 512, 3))

        if any(has_nsfw_concepts):
            print(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts

class BaseRewardModel:
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    def get_rewards(
        self, responses: List,
    ) -> torch.FloatTensor:
        ...

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
        self.count_limit = 3000

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        """
        This method normalizes the given rewards by updating the moving mean and variance statistics. The rewards are first standardized, and then scaled to the 0-1 range using a cumulative distribution function (CDF) to ensure they're in a comparable range across different environments.

        Args:
        rewards (torch.FloatTensor): The reward values to be normalized.

        Returns:
        torch.FloatTensor: The normalized reward values.

        Note:
        - This function uses Welford's online algorithm to update the mean and variance.
        - It standardizes the reward values using the updated mean and variance.
        - It then scales the standardized values to the 0-1 range using the error function (erf) as a CDF.
        """
        # Get the number of rewards (successful responses).
        new_count = rewards.numel()

        # Update stats only if there are new rewards.
        if 0 < new_count and 0 < self.count + new_count:
            # Calculate the mean and standard deviation of the new rewards.
            new_mean = rewards.mean()
            new_var = rewards.var(dim=0)

            # Compute the weights for the new and old rewards.
            new_weight = new_count / (self.count + new_count)
            old_weight = self.count / (self.count + new_count)

            # Save the difference in means before updating the old mean.
            diff = new_mean - self.mean

            # Update the old mean with the new mean and weights.
            self.mean = new_weight * new_mean + old_weight * self.mean
            # Update the old variance with the new variance and weights, and adjusting for the difference in means.
            self.var = (
                (new_weight * new_var)
                + (old_weight * self.var)
                + (new_weight * old_weight) * diff * diff
            )
            # Update the old count with the new count, but don't exceed the limit.
            self.count = min(self.count_limit, self.count + new_count)

        # Standardize the rewards using the updated mean and variance.
        rewards = rewards - self.mean
        if self.var > 0:
            rewards /= torch.sqrt(self.var)
        # Scale the standardized rewards to the range [0, 1] using the error function as a cumulative distribution function (CDF).
        rewards = 0.5 * (
            1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device))
        )

        return rewards

    def apply(
        self, responses: List[bt.Synapse]
    ) -> torch.FloatTensor:
        """Applies the reward model across each call. Unsuccessful responses are zeroed."""
        # Get indices of correctly responding calls.

        successful_generations_indices: List[int] = [
            idx
            for idx, resp in enumerate(responses)
            if resp.dendrite.status_code == 200
        ]

        # Get all completions from responding calls.
        successful_generations: List[str] = [
            responses[idx] for idx in successful_generations_indices
        ]

        # Reward each completion.
        successful_rewards = self.get_rewards(successful_generations,)

        # Softmax rewards across samples.
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)

        # Init zero rewards for all calls.
        filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

        # Fill reward tensor.
        for idx, reward, reward_normalized in zip(
            successful_generations_indices,
            successful_rewards,
            successful_rewards_normalized,
        ):
            filled_rewards[idx] = reward
            filled_rewards_normalized[idx] = reward_normalized

        # Return the filled rewards.
        return filled_rewards, filled_rewards_normalized

class BlacklistFilter(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.blacklist.value

    def __init__(self):
        super().__init__()
        self.question_blacklist = []
        self.answer_blacklist = []

    def reward(self, response) -> float:
        
        # TODO maybe delete this if not needed
        # Check the number of returned images in the response
        if len(response.images) != response.num_images_per_prompt:
            return 0.0

        # If any images in the response fail the reward for that response is 0.0
        for image in response.images:
            # Check if the image can be serialized
            try:
                img = bt.Tensor.deserialize(image)
            except:
                return 0.0
            
            # Check if the image is black image
            if img.sum() == 0:
                return 0.0

            # Check if the image has the type bt.tensor
            if not isinstance(image, bt.Tensor):
                return 0.0

            # check image size
            if not ((image.shape[1] == response.height) and (image.shape[2] == response.width)):
                return 0.0
        
        return 1.0

    def get_rewards(
        self, responses
    ) -> torch.FloatTensor:
        return torch.tensor(
            [self.reward(response) for response in responses],
            dtype=torch.float32,
        )

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards
    
class NSFWRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.nsfw.value

    def __init__(self):
        super().__init__()
        self.device = "cuda"
        # torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.safetychecker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to(self.device)
        self.processor = CLIPImageProcessor()

    def reward(self, response) -> float:
        
        # delete all none images
        for j, image in enumerate(response.images):
            if image is None:
                return 0.0

        if len(response.images) == 0:
           return 0.0
        try:
            clip_input = self.processor([bt.Tensor.deserialize(image) for image in response.images], return_tensors="pt").to(self.device)
            images, has_nsfw_concept = self.safetychecker.forward(images=response.images, clip_input=clip_input.pixel_values.to(self.device))

            any_nsfw = any(has_nsfw_concept)
            if any_nsfw:
                return 0.0

            # # remove all nsfw images from the response
            # for j, has_nsfw in enumerate(has_nsfw_concept):
            #     if has_nsfw:
            #         del responses[i].images[j]

        except Exception as e:
            print(response.images)
            bt.logging.trace(f"Error in NSFW detection: {e}")
            return 1.0
        
        return 1.0

    def get_rewards(
        self, responses
    ) -> torch.FloatTensor:
        return torch.tensor(
            [self.reward(response) for response in responses],
            dtype=torch.float32,
        )

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards

class ImageRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.image.value

    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.scoring_model = RM.load("ImageReward-v1.0", device=self.device)

    def reward(self, response) -> float:
        
        # # if theres no images, skip this response.
        # if len(response.images) == 0:
        #     top_images.append(None)
        #     continue

        img_scores = torch.zeros( len(response.images), dtype = torch.float32 )
        try:
            with torch.no_grad():

                images = [transforms.ToPILImage()(bt.Tensor.deserialize(image)) for image in response.images]
                
                _, scores = self.scoring_model.inference_rank(response.prompt, images)
                
                image_scores = torch.tensor(scores) 
                mean_image_scores = torch.mean(image_scores)    
                
                return mean_image_scores
            
        except Exception as e:
            print(e)
            print("error in response:")
            print(response)
            return 0.0
     
    def get_rewards(
        self, responses
    ) -> torch.FloatTensor:
        return torch.tensor(
            [self.reward(response) for response in responses],
            dtype=torch.float32,
        )

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards

from transformers import AutoImageProcessor, AutoModel, AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T
import torchvision.transforms as transforms
from datasets import Dataset


class DiversityRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.diversity.value

    def __init__(self):
        super().__init__()
        self.model_ckpt = "nateraw/vit-base-beans"
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_ckpt)
        self.processor = AutoImageProcessor.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt)
        self.hidden_dim = self.model.config.hidden_size            
        self.transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * self.extractor.size["height"])),
                T.CenterCrop(self.extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std),
            ]
        )
        # TODO take device argument in
        self.device = "cuda"
        
    def extract_embeddings(self, model: torch.nn.Module):
        """Utility to compute embeddings."""
        device = model.device

        def pp(batch):
            images = batch["image"]
            # `transformation_chain` is a compostion of preprocessing
            # transformations we apply to the input images to prepare them
            # for the model. For more details, check out the accompanying Colab Notebook.
            image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in images]
            )
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            with torch.no_grad():
                embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
            return {"embeddings": embeddings}

        return pp

    def get_rewards(
        self, responses
    ) -> torch.FloatTensor:
        # return torch.tensor(
        #     [self.reward(response) for response in responses],
        #     dtype=torch.float32,
        # )
        extract_fn = self.extract_embeddings(self.model.to(self.device))
        images = [T.transforms.ToPILImage()(bt.Tensor.deserialize(response.images[0])) for response in responses]
        ds = Dataset.from_dict({"image":images})
        embeddings = ds.map(extract_fn, batched=True, batch_size=24)
        embeddings = embeddings['embeddings']
        simmilarity_matrix = cosine_similarity(embeddings)

        similarity_scores = torch.zeros(len(responses), dtype=torch.float32)
        for i in range(0, len(simmilarity_matrix)):
            for j in range(0, len(simmilarity_matrix)):
                if int(simmilarity_matrix[i][j]) == 1:
                    simmilarity_matrix[i][j] = 0
            similarity_scores[i] = 1 - max(simmilarity_matrix[i])
        
        return similarity_scores

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards

