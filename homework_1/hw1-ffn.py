#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import csv

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, t, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        super().__init__()

        n_classes = t

        if activation_type == "relu":
            activation = nn.ReLU()
        elif activation_type == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation_type: {activation_type}")

        modules = []
        in_dim = n_features

        # Hidden layers with activation and optional dropout
        for _ in range(layers):
            modules.append(nn.Linear(in_dim, hidden_size))
            modules.append(activation)
            if dropout and dropout > 0.0:
                modules.append(nn.Dropout(p=dropout))
            in_dim = hidden_size

        # Output layer
        modules.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*modules)

    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN """
        return self.net(x)

def train_batch(X, y, model, optimizer, criterion):

    model.train()
    optimizer.zero_grad()
    scores = model(X)
    loss = criterion(scores, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def _parse_csv_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_one_config(train_X, train_y, dev_X, dev_y, test_X, test_y, n_classes, n_feats,
                   hidden_size, layers, activation, dropout,
                   optimizer_name, lr, l2_decay,
                   batch_size, epochs, seed=42, return_curves=False):

    utils.configure_seed(seed=seed)

    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        hidden_size,
        layers,
        activation,
        dropout
    )

    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optim_cls = optims[optimizer_name]
    optimizer = optim_cls(
        model.parameters(),
        lr=lr,
        weight_decay=l2_decay
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    train_losses_per_epoch = []
    train_accs_per_epoch = []
    val_accs_per_epoch = []

    for epoch in range(epochs):
        epoch_losses = []
        for Xb, yb in train_loader:
            loss = train_batch(Xb, yb, model, optimizer, criterion)
            epoch_losses.append(loss)

        epoch_train_loss = torch.tensor(epoch_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        _, val_acc = evaluate(model, dev_X, dev_y, criterion)

        train_losses_per_epoch.append(epoch_train_loss)
        train_accs_per_epoch.append(train_acc)
        val_accs_per_epoch.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    final_train_acc = train_accs_per_epoch[-1]
    _, test_acc = evaluate(model, test_X, test_y, criterion)

    cfg = {
        "width": hidden_size,
        "lr": lr,
        "dropout": dropout,
        "l2_decay": l2_decay,
        "best_val_acc": best_val_acc,
        "final_train_acc": final_train_acc,
        "test_acc": test_acc,
        "optimizer": optimizer_name,
        "activation": activation,
        "layers": layers,
        "batch_size": batch_size,
        "epochs": epochs
    }

    if return_curves:
        cfg["train_losses"] = train_losses_per_epoch
        cfg["train_accs"] = train_accs_per_epoch
        cfg["val_accs"] = val_accs_per_epoch

    return best_val_acc, cfg

@torch.no_grad()
def predict(model, X):
    scores = model(X)
    preds = torch.argmax(scores, dim=1)
    return preds


@torch.no_grad()
def evaluate(model, X, y, criterion):
    model.eval()
    scores = model(X)
    loss = criterion(scores, y).item()
    preds = torch.argmax(scores, dim=1)
    acc = (preds == y).float().mean().item()
    return loss, acc


def plot(epochs, plottables, filename=None, ylim=None):

    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()

    # Grid search args for Question 2
    parser.add_argument('-grid_search', action='store_true',
                        help="Run the 5x4x2x2=80 configuration grid search (one hidden layer).")
    parser.add_argument('-grid_lrs', type=str, default='0.1,0.01,0.001,0.0001',
                        help="Comma-separated learning rates for grid search.")
    parser.add_argument('-grid_dropout', type=str, default='0.0,0.5',
                        help="Comma-separated dropouts for grid search (e.g. '0.0,0.5').")
    parser.add_argument('-grid_l2', type=str, default='0.0,0.0001',
                        help="Comma-separated L2/weight_decay values for grid search.")
    parser.add_argument('-grid_out', type=str, default='ffn_grid_results.csv',
                        help="CSV filename to write grid-search results to.")

    # Depth experiment args for Question 3
    parser.add_argument('-depth_experiment', action='store_true',
                        help="Run depth experiment (Question 3): fixed width=32, varying layers={1,3,5,7,9}.")
    parser.add_argument('-depth_out', type=str, default='ffn_depth_results.csv',
                        help="CSV filename to write depth experiment results to.")

    #  args for single run mode
    parser.add_argument('-epochs', default=30, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation', choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz')
    parser.add_argument('-model', type=str, default='ffn', help="Model name for file saving")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    # Convert to tensors ,for easse of use
    train_X = torch.from_numpy(train_X).float()
    train_y = torch.from_numpy(train_y).long()
    dev_X = torch.from_numpy(dev_X).float()
    dev_y = torch.from_numpy(dev_y).long()
    test_X = torch.from_numpy(test_X).float()
    test_y = torch.from_numpy(test_y).long()

    n_classes = torch.unique(train_y).numel()
    n_feats = train_X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    # --------------------------
    # GRID SEARCH MODE
    # --------------------------
    if opt.grid_search:
        widths = [16, 32, 64, 128, 256]
        lrs = _parse_csv_floats(opt.grid_lrs)
        dropouts = _parse_csv_floats(opt.grid_dropout)
        l2s = _parse_csv_floats(opt.grid_l2)

        layers = 1
        batch_size = 64
        epochs = 30

        results = []
        start = time.time()

        for w in widths:
            for lr in lrs:
                for d in dropouts:
                    for l2 in l2s:
                        best_val_acc, cfg = run_one_config(
                            train_X, train_y, dev_X, dev_y, test_X, test_y, n_classes, n_feats,
                            hidden_size=w,
                            layers=layers,
                            activation=opt.activation,
                            dropout=d,
                            optimizer_name=opt.optimizer,
                            lr=lr,
                            l2_decay=l2,
                            batch_size=batch_size,
                            epochs=epochs,
                            seed=42,
                            return_curves=False  
                        )
                        results.append(cfg)
                        print(f"width={w:3d} lr={lr:g} drop={d:g} l2={l2:g} -> best_val_acc={best_val_acc:.4f} | final_train_acc={cfg['final_train_acc']:.4f}")

        # Highlight best per width
        best_by_width = {}
        for r in results:
            w = r["width"]
            if (w not in best_by_width) or (r["best_val_acc"] > best_by_width[w]["best_val_acc"]):
                best_by_width[w] = r

        for r in results:
            r["best_for_width"] = (r is best_by_width[r["width"]])

        #  CSV
        fieldnames = ["width", "lr", "dropout", "l2_decay", "best_val_acc", "final_train_acc",
                      "test_acc", "best_for_width", "optimizer", "activation", "layers",
                      "batch_size", "epochs"]
        with open(opt.grid_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        elapsed = time.time() - start
        print(f"\nGrid search done. Wrote {len(results)} rows to {opt.grid_out}")
        print(f"Took {int(elapsed//60)} minutes and {int(elapsed%60)} seconds\n")

        # Summary per width
        print("\n" + "="*80)
        print("BEST CONFIGURATION PER WIDTH:")
        print("="*80)
        for w in widths:
            b = best_by_width[w]
            print(f"Width={w:3d}: val_acc={b['best_val_acc']:.4f} | train_acc={b['final_train_acc']:.4f} | "
                  f"test_acc={b['test_acc']:.4f} | lr={b['lr']:g} drop={b['dropout']:g} l2={b['l2_decay']:g}")

        #  Best config
        overall_best = max(results, key=lambda x: x["best_val_acc"])
        print("\n" + "="*80)
        print("OVERALL BEST CONFIGURATION:")
        print("="*80)
        print(f"Width={overall_best['width']}, LR={overall_best['lr']}, "
              f"Dropout={overall_best['dropout']}, L2={overall_best['l2_decay']}")
        print(f"Best Val Acc: {overall_best['best_val_acc']:.4f}")
        print(f"Final Train Acc: {overall_best['final_train_acc']:.4f}")
        print(f"Test Acc: {overall_best['test_acc']:.4f}")

        # Retrain the model with curves for Question 2b
        print("\n" + "="*80)
        print("RETRAINING BEST MODEL TO GET CURVES (for Question 2b)...")
        print("="*80)
        _, best_cfg_with_curves = run_one_config(
            train_X, train_y, dev_X, dev_y, test_X, test_y, n_classes, n_feats,
            hidden_size=overall_best['width'],
            layers=overall_best['layers'],
            activation=overall_best['activation'],
            dropout=overall_best['dropout'],
            optimizer_name=overall_best['optimizer'],
            lr=overall_best['lr'],
            l2_decay=overall_best['l2_decay'],
            batch_size=overall_best['batch_size'],
            epochs=overall_best['epochs'],
            seed=42,
            return_curves=True
        )

        epochs_range = list(range(1, epochs + 1))

        # Training loss plot
        plt.clf()
        plt.plot(epochs_range, best_cfg_with_curves['train_losses'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Training Loss - Best Model (width={overall_best['width']})")
        plt.legend()
        plt.grid(True)
        plt.savefig('q2b_training_loss.pdf', bbox_inches='tight')
        print("Saved: q2b_training_loss.pdf")

        # Validation accuracy plot
        plt.clf()
        plt.plot(epochs_range, best_cfg_with_curves['val_accs'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f"Validation Accuracy - Best Model (width={overall_best['width']})")
        plt.legend()
        plt.grid(True)
        plt.savefig('q2b_validation_accuracy.pdf', bbox_inches='tight')
        print("Saved: q2b_validation_accuracy.pdf")

        # Final training accuracy vs width plot
        plt.clf()
        widths_sorted = sorted(widths)
        final_train_accs = [best_by_width[w]['final_train_acc'] for w in widths_sorted]
        plt.plot(widths_sorted, final_train_accs, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Hidden Layer Width')
        plt.ylabel('Final Training Accuracy')
        plt.title('Final Training Accuracy vs Width (Best Config per Width)')
        plt.grid(True)
        plt.savefig('q2c_train_acc_vs_width.pdf', bbox_inches='tight')
        print("Saved: q2c_train_acc_vs_width.pdf")

        print("\n" + "="*80)
        print("SUMMARY FOR REPORT:")
        print("="*80)
        print(f"\nQuestion 2a: See {opt.grid_out} for all 80 configurations")
        print(f"\nQuestion 2b:")
        print(f"  - Best model: width={overall_best['width']}, lr={overall_best['lr']}, "
              f"dropout={overall_best['dropout']}, l2={overall_best['l2_decay']}")
        print(f"  - Best validation accuracy: {overall_best['best_val_acc']:.4f}")
        print(f"  - Test accuracy: {overall_best['test_acc']:.4f}")
        print(f"  - Plots saved: q2b_training_loss.pdf, q2b_validation_accuracy.pdf")
        print(f"\nQuestion 2c:")
        print(f"  - Final training accuracies by width: {dict(zip(widths_sorted, final_train_accs))}")
        print(f"  - Plot saved: q2c_train_acc_vs_width.pdf")

        return

    # --------------------------
    # DEPTH EXPERIMENT MODE (Question 3)
    # --------------------------
    if opt.depth_experiment:
        print("\n" + "="*80)
        print("RUNNING DEPTH EXPERIMENT (Question 3)")
        print("="*80)
        print("\nYou need to specify the best hyperparameters from Question 2a for width=32.")
        print("Please run with additional arguments:")
        print("  -learning_rate <best_lr>")
        print("  -dropout <best_dropout>")
        print("  -l2_decay <best_l2>")
        print("  -optimizer <sgd or adam>")
        print("  -activation <relu or tanh>")
        print("\nExample:")
        print("  python hw1-ffn.py -depth_experiment -learning_rate 0.01 -dropout 0.5 -l2_decay 0.0001 -optimizer sgd -activation relu")
        print("="*80)

        depths = [1, 3, 5, 7, 9]
        fixed_width = 32
        batch_size = 64
        epochs = 30

        results = []
        start = time.time()

        for depth in depths:
            print(f"\nTraining model with {depth} hidden layer(s)...")
            best_val_acc, cfg = run_one_config(
                train_X, train_y, dev_X, dev_y, test_X, test_y, n_classes, n_feats,
                hidden_size=fixed_width,
                layers=depth,
                activation=opt.activation,
                dropout=opt.dropout,
                optimizer_name=opt.optimizer,
                lr=opt.learning_rate,
                l2_decay=opt.l2_decay,
                batch_size=batch_size,
                epochs=epochs,
                seed=42,
                return_curves=False
            )
            results.append(cfg)
            print(f"Depth={depth}: best_val_acc={best_val_acc:.4f} | final_train_acc={cfg['final_train_acc']:.4f}")

        fieldnames = ["layers", "width", "lr", "dropout", "l2_decay", "best_val_acc",
                      "final_train_acc", "test_acc", "optimizer", "activation",
                      "batch_size", "epochs"]
        with open(opt.depth_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        elapsed = time.time() - start
        print(f"\nDepth experiment done. Wrote {len(results)} rows to {opt.depth_out}")
        print(f"Took {int(elapsed//60)} minutes and {int(elapsed%60)} seconds\n")

        # Best depth
        best_depth_cfg = max(results, key=lambda x: x["best_val_acc"])

        print("\n" + "="*80)
        print("DEPTH EXPERIMENT RESULTS:")
        print("="*80)
        print(f"{'Depth':<10} {'Val Acc':<12} {'Train Acc':<12} {'Test Acc':<12}")
        print("-" * 80)
        for r in results:
            marker = " <-- BEST" if r is best_depth_cfg else ""
            print(f"{r['layers']:<10} {r['best_val_acc']:<12.4f} {r['final_train_acc']:<12.4f} {r['test_acc']:<12.4f}{marker}")

        # Retrain the model with curves for Question 3b
        print("\n" + "="*80)
        print("RETRAINING BEST DEPTH MODEL TO GET CURVES (for Question 3b)...")
        print("="*80)
        _, best_depth_with_curves = run_one_config(
            train_X, train_y, dev_X, dev_y, test_X, test_y, n_classes, n_feats,
            hidden_size=best_depth_cfg['width'],
            layers=best_depth_cfg['layers'],
            activation=best_depth_cfg['activation'],
            dropout=best_depth_cfg['dropout'],
            optimizer_name=best_depth_cfg['optimizer'],
            lr=best_depth_cfg['lr'],
            l2_decay=best_depth_cfg['l2_decay'],
            batch_size=best_depth_cfg['batch_size'],
            epochs=best_depth_cfg['epochs'],
            seed=42,
            return_curves=True
        )

        epochs_range = list(range(1, epochs + 1))

        # Training loss plot
        plt.clf()
        plt.plot(epochs_range, best_depth_with_curves['train_losses'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Training Loss - Best Depth Model ({best_depth_cfg['layers']} layers)")
        plt.legend()
        plt.grid(True)
        plt.savefig('q3b_training_loss.pdf', bbox_inches='tight')
        print("Saved: q3b_training_loss.pdf")

        # Validation accuracy plot
        plt.clf()
        plt.plot(epochs_range, best_depth_with_curves['val_accs'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f"Validation Accuracy - Best Depth Model ({best_depth_cfg['layers']} layers)")
        plt.legend()
        plt.grid(True)
        plt.savefig('q3b_validation_accuracy.pdf', bbox_inches='tight')
        print("Saved: q3b_validation_accuracy.pdf")

        #  Training accuracy vs depth plot
        plt.clf()
        depths_sorted = sorted(depths)
        final_train_accs_depth = [next(r['final_train_acc'] for r in results if r['layers'] == d) for d in depths_sorted]
        plt.plot(depths_sorted, final_train_accs_depth, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Hidden Layers')
        plt.ylabel('Final Training Accuracy')
        plt.title('Final Training Accuracy vs Depth (Fixed Width=32)')
        plt.grid(True)
        plt.xticks(depths_sorted)
        plt.savefig('q3c_train_acc_vs_depth.pdf', bbox_inches='tight')
        print("Saved: q3c_train_acc_vs_depth.pdf")

        print("\n" + "="*80)
        print("SUMMARY FOR REPORT (Question 3):")
        print("="*80)
        print(f"\nQuestion 3a: See {opt.depth_out} for all depth configurations")
        print(f"\nQuestion 3b:")
        print(f"  - Best model: {best_depth_cfg['layers']} layers")
        print(f"  - Best validation accuracy: {best_depth_cfg['best_val_acc']:.4f}")
        print(f"  - Test accuracy: {best_depth_cfg['test_acc']:.4f}")
        print(f"  - Plots saved: q3b_training_loss.pdf, q3b_validation_accuracy.pdf")
        print(f"\nQuestion 3c:")
        print(f"  - Final training accuracies by depth: {dict(zip(depths_sorted, final_train_accs_depth))}")
        print(f"  - Plot saved: q3c_train_acc_vs_depth.pdf")

        return

    # --------------------------
    # SINGLE RUN MODE (original behavior)
    # --------------------------
    train_dataloader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=opt.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.activation,
        opt.dropout
    )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(0, opt.epochs + 1)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    start = time.time()

    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accs.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accs.append(initial_val_acc)
    print('initial val acc: {:.4f}'.format(initial_val_acc))

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        model.train()
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('train loss: {:.4f}| train_acc: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
            epoch_train_loss, train_acc, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final Test Accuracy: {:.4f}'.format(test_acc))

    # plot
    config = (
        f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
        f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
        f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
    )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }

    accs = {
        "Train Accuracy": train_accs,
        "Valid Accuracy": valid_accs
    }

    plot(epochs, losses, filename=f'losses-{config}.pdf')
    plot(epochs, accs, filename=f'accs-{config}.pdf')
    print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(valid_accs):.4f}")


if __name__ == '__main__':
    main()