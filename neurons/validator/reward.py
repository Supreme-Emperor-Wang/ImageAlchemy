import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

import ImageReward as RM
import numpy as np
import requests
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from datasets import Dataset
from neurons.safety import StableDiffusionSafetyChecker
from neurons.validator.utils import calculate_mean_dissimilarity, cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    CLIPConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    PreTrainedModel,
)

import bittensor as bt

transform = T.Compose([T.PILToTensor()])


class RewardModelType(Enum):
    diversity = "diversity_reward_model"
    image = "image_reward_model"
    human = "human_reward_model"
    blacklist = "blacklist_filter"
    nsfw = "nsfw_filter"


def get_automated_rewards(self, responses, uids, task_type):

    event = {"task_type": task_type}

    # Initialise rewards tensor
    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(
        self.device
    )

    for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
        reward_i, reward_i_normalized = reward_fn_i.apply(responses, rewards)
        rewards += weight_i * reward_i_normalized.to(self.device)
        event[reward_fn_i.name] = reward_i.tolist()
        event[reward_fn_i.name + "_normalized"] = reward_i_normalized.tolist()
        print(str(reward_fn_i.name), reward_i_normalized.tolist())
    for masking_fn_i in self.masking_functions:
        mask_i, mask_i_normalized = masking_fn_i.apply(responses, rewards)
        rewards *= mask_i_normalized.to(self.device)
        event[masking_fn_i.name] = mask_i.tolist()
        event[masking_fn_i.name + "_normalized"] = mask_i_normalized.tolist()
        print(str(masking_fn_i.name), mask_i_normalized.tolist())

    if not self.config.alchemy.disable_manual_validator:
        print(f"Waiting {self.manual_validator_timeout} seconds for manual vote...")
        start_time = time.perf_counter()

        received_vote = False

        while (time.perf_counter() - start_time) < self.manual_validator_timeout:
            time.sleep(1)
            # If manual vote received
            if os.path.exists("neurons/validator/images/vote.txt"):
                # loop until vote is successfully saved
                while open("neurons/validator/images/vote.txt", "r").read() == "":
                    time.sleep(0.05)
                    continue

                try:
                    reward_i = (
                        int(open("neurons/validator/images/vote.txt", "r").read()) - 1
                    )
                except Exception as e:
                    print(f"An unexpected error occurred parsing the vote: {e}")
                    break

                ### There is a small possibility that not every miner queried will respond.
                ### If 12 are queried, but only 10 respond, we need to handle the error if
                ### the user selects the 11th or 12th image (which don't exist)
                if reward_i >= len(rewards):
                    print(
                        f"Received invalid vote for Image {reward_i+1}: it doesn't exist."
                    )
                    break

                print(f"Received manual vote for Image {reward_i+1}")

                ### Set to true so we don't normalize the rewards later
                received_vote = True

                reward_i_normalized: torch.FloatTensor = torch.zeros(
                    len(rewards), dtype=torch.float32
                ).to(self.device)
                reward_i_normalized[reward_i] = 1.0
                rewards += self.reward_weights[-1] * reward_i_normalized.to(self.device)
                if not self.config.alchemy.disable_log_rewards:
                    event["human_reward_model"] = reward_i_normalized.tolist()
                    event[
                        "human_reward_model_normalized"
                    ] = reward_i_normalized.tolist()

                break

        if not received_vote:
            delta = 1 - self.reward_weights[-1]
            if delta != 0:
                rewards /= delta
            else:
                print("The reward weight difference was 0 which is unexpected.")
            print("No valid vote was received")

        # Delete contents of images folder except for black image
        if os.path.exists("neurons/validator/images"):
            for file in os.listdir("neurons/validator/images"):
                (
                    os.remove(f"neurons/validator/images/{file}")
                    if file != "black.png"
                    else "_"
                )

    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, rewards
    ).to(self.device)

    return scattered_rewards, event, rewards

def get_human_rewards(self, rewards):
    _, human_voting_scores_normalised = self.human_voting_reward_model.get_rewards(
        self.hotkeys
    )
    scattered_rewards_adjusted = rewards + (
        self.human_voting_weight * human_voting_scores_normalised
    )
    return scattered_rewards_adjusted


def filter_rewards(self, rewards):

    for uid, count in self.isalive_dict.items():
        if count >= self.isalive_threshold:
            rewards[uid] = 0
    
    return rewards

@dataclass(frozen=True)
class DefaultRewardFrameworkConfig:
    """Reward framework default configuration.
    Note: All the weights should add up to 1.0.
    """

    diversity_model_weight: float = 0.05
    image_model_weight: float = 0.95
    human_model_weight: float = 0

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = (
            cosine_distance(image_embeds, self.special_care_embeds)
            .cpu()
            .float()
            .numpy()
        )
        cos_dist = (
            cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()
        )

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {
                "special_scores": {},
                "special_care": [],
                "concept_scores": {},
                "bad_concepts": [],
                "bad_score": 0.0,
            }

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 1.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(
                    concept_cos - (concept_threshold * adjustment), 3
                )
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append(
                        {concept_idx, result_img["special_scores"][concept_idx]}
                    )

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(
                    concept_cos - (concept_threshold * adjustment), 3
                )
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)
                    result_img["bad_score"] += result_img["concept_scores"][concept_idx]

            result.append(result_img)

        has_nsfw_concepts = [
            len(res["bad_concepts"]) > 0 and res["bad_score"] > 0.01 for res in result
        ]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if torch.is_tensor(images) or torch.is_tensor(images[0]):
                    images[idx] = torch.zeros_like(images[idx])  # black image
                else:
                    try:
                        images[idx] = np.zeros(
                            transform(images[idx]).shape
                        )  # black image
                    except:
                        images[idx] = np.zeros((1024, 1024, 3))

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
    def get_rewards(self, responses: List, rewards) -> torch.FloatTensor:
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
        self,
        responses: List[bt.Synapse],
        rewards,
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
        successful_rewards = self.get_rewards(successful_generations, rewards)

        # Softmax rewards across samples.
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)

        # Init zero rewards for all calls.
        filled_rewards = torch.zeros(len(responses), dtype=torch.float32)
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
            if not (
                (image.shape[1] == response.height)
                and (image.shape[2] == response.width)
            ):
                return 0.0

        return 1.0

    def get_rewards(self, responses, rewards) -> torch.FloatTensor:
        return torch.tensor(
            [
                self.reward(response) if reward != 0.0 else 0.0
                for response, reward in zip(responses, rewards)
            ],
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
        self.safetychecker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.device)
        self.processor = CLIPImageProcessor()

    def reward(self, response) -> float:
        # delete all none images
        for j, image in enumerate(response.images):
            if image is None:
                return 0.0

        if len(response.images) == 0:
            return 0.0
        try:
            clip_input = self.processor(
                [bt.Tensor.deserialize(image) for image in response.images],
                return_tensors="pt",
            ).to(self.device)
            images, has_nsfw_concept = self.safetychecker.forward(
                images=response.images,
                clip_input=clip_input.pixel_values.to(self.device),
            )

            any_nsfw = any(has_nsfw_concept)
            if any_nsfw:
                return 0.0

        except Exception as e:
            print(response.images)
            print(f"Error in NSFW detection: {e}")
            return 1.0

        return 1.0

    def get_rewards(self, responses, rewards) -> torch.FloatTensor:
        return torch.tensor(
            [
                self.reward(response) if reward != 0.0 else 0.0
                for response, reward in zip(responses, rewards)
            ],
            dtype=torch.float32,
        )

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.human.value

    def __init__(self, metagraph, api_url):
        super().__init__()
        self.device = "cuda"
        self.human_voting_scores = torch.zeros((metagraph.n)).to(self.device)
        self.api_url = api_url

    def get_rewards(self, hotkeys) -> torch.FloatTensor:
        max_retries = 3
        backoff = 2

        bt.logging.info("Extracting human votes")

        human_voting_scores = None

        for attempt in range(0, max_retries):
            try:
                human_voting_scores = requests.get(f"{self.api_url}/votes", timeout=2)

                if (human_voting_scores.status_code != 200) and (
                    attempt == max_retries
                ):
                    bt.logging.info(
                        f"Failed to retrieve the human validation votes {attempt+1} times. Skipping until the next step."
                    )
                    human_voting_scores = None
                    break

                elif (human_voting_scores.status_code != 200) and (
                    attempt != max_retries
                ):
                    continue

                else:
                    human_voting_round_scores = human_voting_scores.json()

                    human_voting_scores = {}

                    for inner_dict in human_voting_round_scores.values():
                        for key, value in inner_dict.items():
                            if key in human_voting_scores:
                                human_voting_scores[key] += value
                            else:
                                human_voting_scores[key] = value

                    break

            except Exception as e:
                print(
                    f"Encountered the following error retrieving the manual validator scores: {e}. Retrying in {backoff} seconds."
                )
                time.sleep(backoff)
                human_voting_scores = None
                break

        if human_voting_scores is not None:
            for index, hotkey in enumerate(hotkeys):
                if hotkey in human_voting_scores.keys():
                    self.human_voting_scores[index] = human_voting_scores[hotkey]

        if self.human_voting_scores.sum() == 0:
            human_voting_scores_normalised = self.human_voting_scores
        else:
            human_voting_scores_normalised = (
                self.human_voting_scores / self.human_voting_scores.sum()
            )

        return self.human_voting_scores, human_voting_scores_normalised


class ImageRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.image.value

    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.scoring_model = RM.load("ImageReward-v1.0", device=self.device)

    def reward(self, response) -> float:
        img_scores = torch.zeros(len(response.images), dtype=torch.float32)
        try:
            with torch.no_grad():
                images = [
                    transforms.ToPILImage()(bt.Tensor.deserialize(image))
                    for image in response.images
                ]
                _, scores = self.scoring_model.inference_rank(response.prompt, images)

                image_scores = torch.tensor(scores)
                mean_image_scores = torch.mean(image_scores)

                return mean_image_scores

        except Exception as e:
            print("ImageReward score is 0. No image in response.")
            return 0.0

    def get_rewards(self, responses, rewards) -> torch.FloatTensor:
        return torch.tensor(
            [self.reward(response) for response in responses],
            dtype=torch.float32,
        )


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
                T.Normalize(
                    mean=self.extractor.image_mean, std=self.extractor.image_std
                ),
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

    def get_rewards(self, responses, rewards) -> torch.FloatTensor:
        extract_fn = self.extract_embeddings(self.model.to(self.device))

        images = [
            T.transforms.ToPILImage()(bt.Tensor.deserialize(response.images[0]))
            for response, reward in zip(responses, rewards)
            if reward != 0.0
        ]
        ignored_indices = [
            index for index, reward in enumerate(rewards) if reward == 0.0
        ]
        if len(images) > 1:
            ds = Dataset.from_dict({"image": images})
            embeddings = ds.map(extract_fn, batched=True, batch_size=24)
            embeddings = embeddings["embeddings"]
            simmilarity_matrix = cosine_similarity(embeddings)

            dissimilarity_scores = torch.zeros(len(responses), dtype=torch.float32)
            for i in range(0, len(simmilarity_matrix)):
                for j in range(0, len(simmilarity_matrix)):
                    if i == j:
                        simmilarity_matrix[i][j] = 0
                dissimilarity_scores[i] = 1 - max(simmilarity_matrix[i])
        else:
            dissimilarity_scores = torch.tensor([1.0])

        if ignored_indices and (len(images) > 1):
            i = 0
            while i < len(rewards):
                if i in ignored_indices:
                    dissimilarity_scores = torch.cat(
                        [
                            dissimilarity_scores[:i],
                            torch.tensor([0]),
                            dissimilarity_scores[i:],
                        ]
                    )
                i += 1
        return dissimilarity_scores

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards / rewards.sum()
