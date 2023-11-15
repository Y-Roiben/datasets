import random
import numpy as np
import torch

def fix_seed(seed):
    """Fix the seed of the random number generator for reproducibility."""
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(1)