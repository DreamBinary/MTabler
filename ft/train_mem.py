from train import train
import torch
if __name__ == "__main__":
    torch.cuda.empty_cache()
    train("sdpa")
