import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import pytest
import syntheticdata


def test_generated_systems_are_nonsingular():
    for _ in range(100):
        text = syntheticdata.gen_system_linear_equations()["instruction"]
        nums = re.findall(r"(-?\d+)x \+ (-?\d+)y", text)
        assert len(nums) == 2
        a, b = map(int, nums[0])
        c, d = map(int, nums[1])
        assert a * d - b * c != 0


def test_count_rejects_negative():
    with pytest.raises(SystemExit):
        syntheticdata.main(["--count", "-1"])
