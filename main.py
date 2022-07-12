import argparse
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Meta info
    parser.add_argument("--task_name", type=str, default="baseline", help="Task name to save.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run.")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for SGD optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay regularization for model.")

    args = parser.parse_args()

