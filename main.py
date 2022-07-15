import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from models.MultiClassificationModel import MultiClassificationModel
from utils.experiment import get_loader, save_model, load_model, train_one_epoch, evaluate_one_epoch, \
    initiate_environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Meta info
    parser.add_argument("--task_name", type=str, default="baseline", help="Task name to save.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run.")
    parser.add_argument("--device", type=str, default=0 if torch.cuda.is_available() else "cpu", help="Device number.")
    parser.add_argument("--num_workers", type=int, default=0, help="Spawn how many processes to load data.")
    parser.add_argument("--rng_seed", type=int, default=114514, help='manual seed')

    # Training
    parser.add_argument("--num_epoch", type=int, default=0, help="Current epoch number.")
    parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch number to run.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint path to load.")
    parser.add_argument("--save_path", type=str, default="./save/", help="Checkpoint path to save.")
    parser.add_argument("--save_freq", type=int, default=1, help="Save model every how many epochs.")
    parser.add_argument("--val_freq", type=int, default=1, help="Save model every how many epochs.")
    parser.add_argument("--print_freq", type=int, default=1, help="Save model every how many batches.")
    parser.add_argument("--batch_size", type=int, default=8, help="Entry numbers every batch.")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="Adam", help="Optimizer type.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for SGD optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay regularization for model.")

    args = parser.parse_args()

    initiate_environment(args)

    # Prepare dataloader
    loader, val_loader = get_loader(args)

    # Load model & optimizer
    model = MultiClassificationModel()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("You must specify a valid optimizer type!")

    if args.checkpoint_path:
        load_model(args, model, optimizer)
    model = model.to(args.device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Main Function
    if args.mode == "train":
        stat_dict = {"train/loss": []}
        for epoch in range(args.num_epoch, args.max_epoch):
            train_one_epoch(epoch, loader, args, model, criterion, optimizer, stat_dict)

            if epoch % args.val_freq == 0:
                evaluate_one_epoch(val_loader, args, model, criterion="acc")

            if epoch % args.save_freq == 0:
                save_model(args, model, optimizer, epoch)

        save_model(args, model, optimizer)
        print("[Main] Model training has been completed!")

    elif args.mode == "test":
        evaluate_one_epoch(loader, args, model, criterion=None, save_name="result.txt")

    else:
        raise NotImplementedError("You must specify either to train or to test!")
