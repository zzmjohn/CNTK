# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device
import pytest
import platform
import subprocess

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
from run_cifar_convnet_distributed import run_cifar_convnet_distributed

TOLERANCE_ABSOLUTE = 2E-1

def test_cifar_convnet_error(device_id):
    set_default_device(cntk_device(device_id))

    test_error = run_cifar_convnet_distributed()
    expected_test_error = 0.617

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)

def test_cifar_convnet_distributed_mpiexec(device_id):
    subprocess.call("mpiexec -n 2 python run_cifar_convnet_distributed.py", stderr=subprocess.STDOUT, shell=True)
