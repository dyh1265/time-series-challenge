import os
import tensorflow as tf
import random
import numpy as np

"""
Sets the random seed for reproducibility across Python, NumPy, and TensorFlow.
This function ensures that experiments are reproducible by setting the seed for Python's built-in random module,
NumPy, and TensorFlow. It also sets environment variables to enforce deterministic operations in TensorFlow.
Args:
    seed (int): The seed value to use for all random number generators.
Note:
    Setting these environment variables and seeds helps achieve reproducible results, but complete determinism
    may not be guaranteed on all hardware or with all TensorFlow operations.
"""

def setSeed(seed):
    os.environ['PYTHONHASHSEED'] = '0'

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
