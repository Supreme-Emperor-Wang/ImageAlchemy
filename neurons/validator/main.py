import argparse
import asyncio
import io
import os
import pathlib
import sys
from pathlib import Path

import torchvision.transforms as T
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
        # import torch
        # {'task_type': 'text_to_image', 'image_reward_model': [1.7191520929336548, 1.5444213151931763, 1.1091123819351196, 0.11631488800048828, 1.7717468738555908, 0.06871559470891953, 0.663800835609436, 0.9276009798049927, 1.271536111831665, 1.0087143182754517], 'image_reward_model_normalized': [0.16852588951587677, 0.15139730274677277, 0.10872461646795273, 0.011402172967791557, 0.17368167638778687, 0.006736086215823889, 0.06507139652967453, 0.09093132615089417, 0.12464676797389984, 0.09888274222612381], 'blacklist_filter': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'blacklist_filter_normalized': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'nsfw_filter': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'nsfw_filter_normalized': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'block': 1282355, 'step_length': 3.393881320953369, 'prompt_t2i': 'A bird flying in paradise', 'prompt_i2i': None, 'uids': [40, 33, 12, 34, 39, 35, 37, 32, 18, 38], 'hotkeys': ['5CCsUXA9JVN5YQfaGHRX3idHHJtNvim1cKwu9PFswfymti3r', '5EyZPUr4LGvJRCKmLHP1b3XVc6G9FfKBXszeRKjmU3iGPQ8j', '5DbwPiEastb82ZKA992EiQ1NHu3gTW5xwsingiFHPRiTk9W8', '5Dcx6p1zGGs9cPqPduCBuXKJYBe9K6mPVvnP9xKSjLXahr4W', '5CFEng7aCgGHDgdRcVcctmiS4DxSEuaqVTBiAqZUb6Kd9UQp', '5GziNQqT64mPqzYo2K9g3uwm9hs4Q5rVBuNi7btLAxf9oYo6', '5GKRf2Ece3a2mij11Lpth8XC1iybTDWDo634Q27HZAFx3scr', '5HYcdVzzRLoXcukvZCKC85f4pZxtjkrH45cazScVhBU5AdE8', '5E4iVWnXnRCsCiDbZjn96Um5XtyscCDbc58hvRjxzEMkunJM', '5DoFMjAoyMAPfd3yKUUpsMFDvcRwzCqXvUo3p33czW5Bdj14'], 'images': [Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024]), Tensor(dtype='torch.uint8', shape=[3, 1024, 1024])], 'rewards': [0.16852588951587677, 0.15139730274677277, 0.10872461646795273, 0.011402172967791557, 0.17368167638778687, 0.006736086215823889, 0.06507139652967453, 0.09093132615089417, 0.12464676797389984, 0.09888274222612381], 'stake': tensor(0.8748), 'rank': tensor(0.), 'vtrust': tensor(0.2143), 'dividends': tensor(0.0001), 'emissions': tensor(0.)}
        response = await validator_app.weight_setter.forward(prompt)
        best_image_index = response['rewards'].index(max(response['rewards']))
        image = T.transforms.ToPILImage()(bt.Tensor.deserialize(response['images'][best_image_index]))
        # breakpoint()
        # stream = io.BytesIO()
        # image.save(stream, "JPEG")
        # return web.Response(body=stream.getvalue(), content_type='image/jpeg')
        image.save('test.jpg')  
        resp = web.FileResponse(f'test.jpg')
        return resp
    except ValueError:
        return Response(status=400)


class ValidatorApplication(web.Application):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # self.weight_setter: WeightSetter | None = None

if __name__ == "__main__":
    ### Add the base repository to the path so the validator can access it
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    if not file_path in sys.path:
        sys.path.append(file_path)

    from validator import StableValidator
    
    loop = asyncio.get_event_loop()

    validator_app = ValidatorApplication()
    validator_app.add_routes([web.post('/t2i/', process_image)])
    validator_app.weight_setter = StableValidator(loop)
    # validator_app.weight_setter.loop.create_task(validator_app.weight_setter.run())

    # validator_api_app = StableValidator()

    try:
        web.run_app(validator_app, port=8000, loop=loop)
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
