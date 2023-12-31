import os


IA_BUCKET_NAME = "image-alchemy"
IA_MINER_BLACKLIST = "blacklist_for_miners.json"
IA_MINER_WHITELIST = "whitelist_for_miners.json"
IA_VALIDATOR_BLACKLIST = "blacklist_for_validators.json"
IA_VALIDATOR_WHITELIST = "whitelist_for_validators.json"
IA_VALIDATOR_WEIGHT_FILES = "weights.json"

WANDB_MINER_PATH = os.path.expanduser("~/.ImageAlchemy/wandb/miner")
WANDB_VALIDATOR_PATH = os.path.expanduser("~/.ImageAlchemy/wandb/validator")
