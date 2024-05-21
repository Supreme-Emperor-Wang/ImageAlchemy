import asyncio
import pathlib
import sys

import sentry_sdk

from neurons.constants import VALIDATOR_SENTRY_DSN

sentry_sdk.init(
    dsn=VALIDATOR_SENTRY_DSN,
)

if __name__ == "__main__":
    ### Add the base repository to the path so the validator can access it
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    if not file_path in sys.path:
        sys.path.append(file_path)

    ### Import StableValidator after fixing paths
    from validator import StableValidator

    asyncio.run(StableValidator().run())
