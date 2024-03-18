import os

IA_BUCKET_NAME = "image-alchemy"
IA_TEST_BUCKET_NAME = "image-alchemy-test"
IA_MINER_BLACKLIST = "blacklist_for_miners.json"
IA_MINER_WHITELIST = "whitelist_for_miners.json"


WANDB_MINER_PATH = os.path.expanduser("~/.ImageAlchemy/wandb/miner")
WANDB_VALIDATOR_PATH = os.path.expanduser("~/.ImageAlchemy/wandb/validator")

### Validator only
N_NEURONS = 12
N_NEURONS_TO_QUERY = 18
VPERMIT_TAO = 1024
FOLLOWUP_TIMEOUT = 10
MOVING_AVERAGE_ALPHA = 0.05
EVENTS_RETENTION_SIZE = "2 GB"
VALIDATOR_DEFAULT_REQUEST_FREQUENCY = 60
VALIDATOR_DEFAULT_QUERY_TIMEOUT = 15
ENABLE_IMAGE2IMAGE = False

IA_VALIDATOR_BLACKLIST = "blacklist_for_validators.json"
IA_VALIDATOR_WHITELIST = "whitelist_for_validators.json"
IA_VALIDATOR_WEIGHT_FILES = "weights.json"
IA_VALIDATOR_SETTINGS_FILE = "validator_settings.json"
IA_MINER_WARNINGLIST = "warninglist_for_miners.json"
