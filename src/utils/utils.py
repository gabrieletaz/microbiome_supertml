import os
from markdown import markdown
import numpy as np

import torch
import torch.nn as nn
import random
from monai.utils.misc import set_determinism

from .args import args



# ----- Random Seed Control -----

def fix_random_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True




if __name__ == "__main__":
    pass
