import time

from base import StableMiner

if __name__ == "__main__":
    with StableMiner():
        while True:
            time.sleep(1)
