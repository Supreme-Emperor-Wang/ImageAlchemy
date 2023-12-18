import pathlib, time, sys

from base import StableMiner
from neurons.shared import add_repo_to_path

if __name__ == "__main__":
    ### Add the base repository to the path so the miner can access it
    add_repo_to_path()

    ### Start the miner
    with StableMiner():
        while True:
            time.sleep(1)
