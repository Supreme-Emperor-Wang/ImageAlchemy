import bittensor as bt
import torch.nn as nn
import torch

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity.item()

def compare_to_set(image_array, target_size=(224, 224)):
    # convert image array to index, image tuple pairs
    image_array = [(i, image) for i, image in enumerate(image_array)]

    # if there are no images, return an empty matrix
    if len(image_array) == 0:
        return []

    # only process images that are not None
    style_vectors = extract_style_vectors([image for _, image in image_array if image is not None], target_size)
    # add back in the None images as zero vectors
    for i, image in image_array:
        if image is None:
            # style_vectors = torch.cat((style_vectors[:i], torch.zeros(1, style_vectors.size(1)), style_vectors[i:]))
            # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_CUDA_cat)
            # Fixed version:
            style_vectors = torch.cat((style_vectors[:i], torch.zeros(1, style_vectors.size(1)).to(style_vectors.device), style_vectors[i:]))

    similarity_matrix = torch.zeros(len(image_array), len(image_array))
    for i in range(style_vectors.size(0)):
        for j in range(style_vectors.size(0)):
            if image_array[i] is not None and image_array[j] is not None:
                similarity = cosine_similarity(style_vectors[i], style_vectors[j])
                likeness = 1.0 - similarity  # Invert the likeness to get dissimilarity
                likeness = min(1,max(0, likeness))  # Clip the likeness to [0,1]
                if likeness < 0.01:
                    likeness = 0
                similarity_matrix[i][j] = likeness

    return similarity_matrix.tolist()

def calculate_mean_dissimilarity(dissimilarity_matrix):
    num_images = len(dissimilarity_matrix)
    mean_dissimilarities = []

    for i in range(num_images):
        dissimilarity_values = [dissimilarity_matrix[i][j] for j in range(num_images) if i != j]
        # error: list index out of range
        if len(dissimilarity_values) == 0 or sum(dissimilarity_values) == 0:
            mean_dissimilarities.append(0)
            continue
        # divide by amount of non zero values
        non_zero_values = [value for value in dissimilarity_values if value != 0]
        mean_dissimilarity = sum(dissimilarity_values) / len(non_zero_values)
        mean_dissimilarities.append(mean_dissimilarity)

     # Min-max normalization
    non_zero_values = [value for value in mean_dissimilarities if value != 0]

    if(len(non_zero_values) == 0):
        return [0.5] * num_images

    min_value = min(non_zero_values)
    max_value = max(mean_dissimilarities)
    range_value = max_value - min_value
    if range_value != 0:
        mean_dissimilarities = [(value - min_value) / range_value for value in mean_dissimilarities]
    else:
        # All elements are the same (no range), set all values to 0.5
        mean_dissimilarities = [0.5] * num_images
    # clamp to [0,1]
    mean_dissimilarities = [min(1,max(0, value)) for value in mean_dissimilarities]

    # Ensure sum of values is 1 (normalize)
    # sum_values = sum(mean_dissimilarities)
    # if sum_values != 0:
    #     mean_dissimilarities = [value / sum_values for value in mean_dissimilarities]

    return mean_dissimilarities

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True
