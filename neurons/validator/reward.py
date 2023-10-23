import torch
from typing import List
from torch import nn
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

# TODO Edit to suit ImageNet
class RewardModelType(Enum):
    dpo = "dpo_reward_model"
    rlhf = "rlhf_reward_model"
    reciprocate = "reciprocate_reward_model"
    dahoas = "dahoas_reward_model"
    diversity = "diversity_reward_model"
    prompt = "prompt_reward_model"
    blacklist = "blacklist_filter"
    nsfw = "nsfw_filter"
    relevance = "relevance_filter"
    task_validator = "task_validator_filter"


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
        self, prompt: str, completion: List[str], name: str
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
        self, prompt: str, responses: List[bt.Synapse], name: str
    ) -> torch.FloatTensor:
        """Applies the reward model across each call. Unsuccessful responses are zeroed."""
        # Get indices of correctly responding calls.

        successful_completions_indices: List[int] = [
            idx
            for idx, resp in enumerate(responses)
            if resp.dendrite.status_code == 200
        ]

        # Get all completions from responding calls.
        successful_completions: List[str] = [
            responses[idx].completion.strip() for idx in successful_completions_indices
        ]

        # Reward each completion.
        successful_rewards = self.get_rewards(prompt, successful_completions, name)

        # Softmax rewards across samples.
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)

        # Init zero rewards for all calls.
        filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)

        # Fill reward tensor.
        for idx, reward, reward_normalized in zip(
            successful_completions_indices,
            successful_rewards,
            successful_rewards_normalized,
        ):
            filled_rewards[idx] = reward
            filled_rewards_normalized[idx] = reward_normalized

        # Return the filled rewards.
        return filled_rewards, filled_rewards_normalized

class Blacklist(BaseRewardModel):
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
            if not ((image.shape[1] == response.height) and (image.shape[2] == response.height)):
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

    safetychecker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker').to( DEVICE )

    @property
    def name(self) -> str:
        return RewardModelType.nsfw.value

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            NSFWRewardModel.nsfw_filter_model_path
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NSFWRewardModel.nsfw_filter_model_path
        ).to(self.device)

    def reward(self, prompt: str, completion: str, name: str) -> float:
        boundary = -0.5
        with torch.no_grad():
            message = completion
            input_ids = self.tokenizer(message)["input_ids"]

            # Returns the nsfw hate score for the chunk.
            def hate_score(chunk) -> float:
                nothate, hate = (
                    self.model(torch.tensor([chunk]).to(self.device)).logits[0].tolist()
                )
                return nothate, hate

            # Returns the max hate score from each chunk of size chunk_size
            def sum_nsfw_scores(input_ids, chunk_size):
                max_score = -1000
                for i in range(0, len(input_ids), chunk_size):
                    chunk_i = input_ids[i : i + chunk_size]
                    not_hate_score_i, hate_score_i = hate_score(chunk_i)
                    max_score = max(max(-not_hate_score_i, hate_score_i), max_score)
                return max_score

            # 0 when needs to be filtered out, 1 when it is safe
            return 0.0 if sum_nsfw_scores(input_ids, chunk_size=512) > boundary else 1.0

    def get_rewards(
        self, prompt: str, completions: List[str], name: str
    ) -> torch.FloatTensor:
        return torch.tensor(
            [self.reward(prompt, completion, name) for completion in completions],
            dtype=torch.float32,
        ).to(self.device)

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards