from tqdm import tqdm

from utils.metric import calc_accuracy


def train_one_epoch(model, train_loader, state_dict):
    model.train()
    optimizer = state_dict['optimizer']
    criterion = state_dict['criterion']
    args = state_dict['args']
    epoch = state_dict['num_epoch']
    total_epoch = state_dict['total_epoch']

    for train_idx, train_data in tqdm(train_loader, total=len(train_loader)):
        train_input, train_label = train_data  # Unpack a tuple
        pred_label = model(train_input)

        optimizer.zero_grad()
        loss = criterion(pred_label, train_label)
        loss.backward()
        optimizer.step()

        if train_idx % args.print_freq == 0:
            # Calc accuracy for display of current batch
            accuracy = calc_accuracy(pred_label, train_label)
            tqdm.write(f"[{epoch}/{total_epoch}] Loss {loss:.4f} Accuracy {(accuracy * 100):.2f}%")


def evaluate_one_epoch(model, val_loader, state_dict):
    pass


def test_one_epoch(model, test_loader, state_dict):
    pass
