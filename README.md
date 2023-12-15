<div align="center">

# **Image Alchemy** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

## Installation

Download the repository, navigate to the folder and then install the necessary requirements with the following chained command.

```bash
git clone https://github.com/Supreme-Emperor-Wang/ImageAlchemy &&\
cd ImageAlchemy &&\
pip install -r requirements.txt &&\
pip install -e .
```

---

## Converting a CivitAI model to a HF compatible model

ImageAlchemy utilizes the highly versatile [Diffusers](https://github.com/huggingface/diffusers) library which can't load fine-tuned SD & SDXL models from websites such as [civita](https://civitai.com/models/) out of the box. To convert any model on this website to a HF compatible model run:

```bash
python scripts/civitai_conversion.py
    --civitai_link https://civitai.com/api/download/models/147497 # Download link for the civitai model
    --is_sdxl # Flag for SD XL models
```

By default the outputs of this script are saved in ```~/civitai/models```.

---

## Running A Miner

```bash
# To run the miner
cd neurons
python -m neurons/miner/StableMiner/main.py
    --netuid <your netuid>  # The subnet id you want to connect to
    --subtensor.network <your chain url>  # blockchain endpoint you want to connect
    --wallet.name <your miner wallet> # name of your wallet
    --wallet.hotkey <your miner hotkey> # hotkey name of your wallet
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --wandb.entity <your wanbd entity> 
    --wandb.project <your wadnb project>
    --wandb.api <your wandb_api_key> # Your wanbd api key
```

---

## Running A Validator

```bash
# To run the miner
cd ImageAlchemy
OPENAI_API_KEY=YOUR_OPENAI_API_KEY python -m neurons/validator/main.py
    --netuid <your netuid>  # The subnet id you want to connect to
    --subtensor.network <your chain url>  # blockchain endpoint you want to connect
    --wallet.name <your miner wallet> # name of your wallet
    --wallet.hotkey <your miner hotkey> # hotkey name of your wallet
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
