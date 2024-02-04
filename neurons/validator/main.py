import argparse
import asyncio
import os
import pathlib
import sys
from pathlib import Path

from aiohttp import web
from aiohttp.web_response import Response

import bittensor as bt

EXPECTED_ACCESS_KEY = os.environ.get('EXPECTED_ACCESS_KEY', "hello")

def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on')
    parser.add_argument('--http_port', type=int, default=8000)
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    _args = parser.parse_args()
    full_path = Path(
        f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator"
    ).expanduser()
    config.full_path = str(full_path)
    full_path.mkdir(parents=True, exist_ok=True)
    return config

class ValidatorApplication(web.Application):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

async def process_image(request: web.Request):
    # Check access key
    access_key = request.headers.get("access-key")
    if access_key != EXPECTED_ACCESS_KEY:
        return Response(status=401, reason="Invalid access key")

    try:
        response = await request.json()
        prompt = response['messages'][0]['content']
        response = validator_app.run(prompt)
        return web.Response(text=str(response))
    except ValueError:
        return Response(status=400)


if __name__ == "__main__":
    ### Add the base repository to the path so the validator can access it
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    if not file_path in sys.path:
        sys.path.append(file_path)

    from validator import StableValidator
    
    validator_app = StableValidator()
    validator_app.add_routes([web.post('/t2i/', process_image)])
    # validator_app.loop = asyncio.get_event_loop()
    loop = asyncio.get_event_loop()


    try:
        web.run_app(validator_app, port=8000, loop=loop)
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")

