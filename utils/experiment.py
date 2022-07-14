import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.metric import calc_accuracy, draw_loss_curve
from datasets.dataset_landscape import LandScapeDataset


def initiate_environment(args):
    """
    initiate randomness.
    :param args: Runtime arguments.
    :return:
    """
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)

def get_loader(args):
    num_workers = args.num_workers
    val_dataset = LandScapeDataset("val")
    if args.mode == "train":
        dataset = LandScapeDataset(args.mode)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=args.batch_size)
    elif args.mode == "test":
        dataset = LandScapeDataset(args.mode)
        dataloader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=args.batch_size)
    else:
        raise NotImplementedError("You must specify either to train or to test!")

    val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, batch_size=args.batch_size)
    return dataloader, val_dataloader


def save_model(args, model, optimizer, epoch="last"):
    os.makedirs(args.save_path, exist_ok=True)
    checkpoint = {
        'config': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(args.save_path, args.task_name, f'ckpt_epoch_{epoch}.pth'))


def load_model(args, model, optimizer):
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train_one_epoch(epoch, train_loader, args, model, criterion, optimizer, stat_dict):
    """
    :param epoch: Epoch number.
    :param train_loader: Train loader.
    :param args: Runtime arguments.
    :param model: Network.
    :param criterion: Loss function.
    :param optimizer: SGD or Adam.
    :param stat_dict: {"train/loss": []}
    """
    model.train()
    start_time = time.time()

    print(f"==> [Epoch {epoch}] Starting Training...")
    for train_idx, train_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        train_input, train_label = train_data["image"].to(args.device), train_data["label"].to(args.device)
        pred_label = model(train_input)

        optimizer.zero_grad()
        loss = criterion(pred_label.reshape(-1, 2), train_label.reshape(-1).long())
        loss.backward()
        optimizer.step()

        if train_idx % args.print_freq == 0:
            # Calc accuracy for display of current batch
            stat_dict["train/loss"].append(loss.detach().item())
            tqdm.write(f"[Epoch {epoch+1} / {args.max_epoch}] [Batch {train_idx+1} / {len(train_loader)}] " +
                       f"Loss {loss:.4f}")
            draw_loss_curve(args, stat_dict["train/loss"])
    print(f"==> [Epoch {epoch}] Finished in {((time.time() - start_time)/60):.2f} minutes.")


def evaluate_one_epoch(loader, args, model, criterion=None, save_name=None):
    """
    :param loader: val or test loader.
    :param args: Runtime arguments.
    :param model: Network.
    :param criterion: Loss function.
    :param save_name: if exists, save current predicted results to `{args.save_path} / {args.task_name} /
            {save_name}.txt` for further testing.
    :return: None
    """
    model.eval()
    results, ground_truths = [], []
    print(f"==> [Eval] Start evaluating model...")
    with torch.no_grad():
        for data_idx, data in tqdm(enumerate(loader), total=len(loader)):
            pred = model(data['image'].to(args.device))
            results.append(pred.cpu().numpy())
            if 'label' in data:
                ground_truths.append(data['label'].cpu().numpy())

        results = [result.argmax(axis=-1) for result in results]
        if criterion == "acc":
            acc = calc_accuracy(results, ground_truths)
            print(f"==> [Eval] Current accuracy: {(acc*100):.2f}%")

        if save_name is not None:
            out_str = ""
            for x in results:
                out_str += ''.join([f"{y[0]},{y[1]},{y[2]}\n" for y in x])
            with open(f"{args.save_path}/{args.task_name}/{save_name}", 'w+') as file:
                file.write(out_str)
