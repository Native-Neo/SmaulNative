#!/usr/bin/env python3
"""Deprecated alias: trainer lives in train.py."""
import train as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
