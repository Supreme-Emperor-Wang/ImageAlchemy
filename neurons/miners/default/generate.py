import time, random
from typing import Dict, List
from utils import output_log


def generate(model, args: Dict) -> List:
    start = time.perf_counter()
    images = model(**args).images
    output_log(f"Time: {time.perf_counter() - start:.2f}s")
    return images
