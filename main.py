import torch
import argparse
import torch.optim as optim
from argparse import ArgumentParser

from models.MultiClassificationModel import MultiClassificationModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Meta info
    parser.add_argument("--task_name", type=str, default="baseline", help="Task name to save.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run.")

    # Training
    parser.add_argument("--num_epoch", type=int, default=0, help="Current epoch number.")
    parser.add_argument("--max_epoch", type=int, default=600, help="Max epoch number to run.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint path to load.")
    parser.add_argument("--save_freq", type=int, default=1, help="Save model every how many epoch.")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="Adam", help="Optimizer type.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for SGD optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay regularization for model.")

    args = parser.parse_args()

    # Load model & optimizer
    model = MultiClassificationModel()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("You must specify a valid optimizer type!")

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path)

        model.load()

    if args.mode == "train":
        # I got hungry. I wanna eat some snacks :(
        pass

    elif args.mode == "test":
        pass

    else:
        raise NotImplementedError("You must specify either to train or to test!")
