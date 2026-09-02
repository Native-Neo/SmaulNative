#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time

import torch

from cpu_backend import configure

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=16_000_000)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--threads", type=int, default=None)
a = p.parse_args()
threads = configure(a.threads)

print(f"torch={torch.__version__} cpu={torch.get_num_threads()} threads={threads}")
