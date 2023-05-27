import torch
import numpy as np

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    sample_inputs = torch.randn(1024, 1000)
    sample_labels = torch.randint(0, 11, (1024,))