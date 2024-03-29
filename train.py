import os
import argparse
import warnings
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn

from datasets import create_dataloaders
from models.lstm import LSTM
from models.tcn import TCN
from models.linear import Linear
from metrics import *

MISSING = -1


def set_seed(seed):
    """
    Set a seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def data_augmentations(data):
    """
    Perform data augmentations on the in and output data.
    """

    for patient_id, patient_data in data.items():

        # Augment input
        X = patient_data['X']
        X = np.where(np.isnan(X), -1, X)
        data[patient_id]['X'] = X

        # Augment output
        y = patient_data['y']
        # Cast CRT values > 6 to 6
        y[:, 1] = np.where(y[:, 1] > 6, 6, y[:, 1])
        y = y.round()
        y = y.astype(int)

        # Turn labels to one-hot encoding
        labels_avpu = np.zeros((y.shape[0], 4))
        labels_avpu[range(labels_avpu.shape[0]), y[:, 0]-1] = 1
        labels_crt = np.zeros((y.shape[0], 7))
        labels_crt[range(labels_crt.shape[0]), y[:, 1]] = 1

        data[patient_id]['y'] = [labels_avpu, labels_crt]

    return data


def calculate_loss_weights(args, data):
    """
    Calculate the frequency per output class. This weight will be used in the
    loss functions to account for an imbalanced dataset. If a class is not
    encountered in the dataset, replace weights with 1000.
    """

    weights_avpu, weights_crt = np.zeros(4), np.zeros(7)

    for d in data.values():
        avpu, crt = d['y'][0], d['y'][1]

        u, c = np.unique(np.argmax(avpu, axis=1), return_counts=True)
        weights_avpu[u] += c

        u, c = np.unique(np.argmax(crt, axis=1), return_counts=True)
        weights_crt[u] += c

    with np.errstate(divide='ignore'):
        max_avpu = max(weights_avpu)
        weights_avpu = max_avpu / weights_avpu
        weights_avpu[weights_avpu == np.inf] = np.mean([w for w in weights_avpu if w < 10000])

        max_crt = max(weights_crt)
        weights_crt = max_crt / weights_crt
        weights_crt[weights_crt == np.inf] = np.mean([w for w in weights_crt if w < 10000])

    return torch.Tensor(weights_avpu), torch.Tensor(weights_crt)


def load_data(args, data_split=[0.7, 0.1, 0.2]):
    """
    Load the dataset, perform data augmentations on it and create dataloaders.
    """

    with open(os.getcwd() + args.data_path, 'rb') as file:
        data = pickle.load(file)

    data = data_augmentations(data)

    input_shape = data[list(data.keys())[0]]['X'].shape
    n_features, n_timesteps = input_shape[2], input_shape[1]
    loss_weights = calculate_loss_weights(args, data)

    return create_dataloaders(data, data_split, args.batch_size), \
        n_features, n_timesteps, loss_weights


def load_model(args, n_features, n_timesteps):

    if args.model == 'lstm':
        model = LSTM(
            num_input=n_features,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_output=[4, 7],
            dropout=args.lstm_dropout
        )

    elif args.model == 'tcn':

        # As to avoid deprecation warning from pytorch_tcn TCN
        warnings.filterwarnings("ignore", category=UserWarning)

        model = TCN(
            num_input=n_features,
            num_timesteps=n_timesteps,
            num_channels=args.num_channels,
            kernel_size=args.kernel_size,
            dilations=args.dilations,
            dilation_reset=args.dilation_reset,
            dropout=args.tcn_dropout,
            causal=args.causal,
            use_norm=args.use_norm,
            activation=args.activation,
            kernel_initializer=args.kernel_initializer,
            num_output=[4, 7],
        )

        # Re-activate the warnings
        warnings.filterwarnings("default", category=UserWarning)

    elif args.model == 'linear':
        model = Linear(n_features,
                       n_timesteps,
                       args.hidden_dims,
                       act_fn=args.act_fn,
                       num_output=[4, 7],
                       linear_dropout=args.linear_dropout)
    else:
        raise ValueError('model unknown')

    if args.verbose:
        print(f'Model:\n{model}')

    return model


@torch.no_grad()
def evaluate(args, model, dataloader, device, loss_fn_avpu, loss_fn_crt,
             average='macro', visualize=False):
    """
    Evaluate the model on accuracy, precision, recall, F1 score and multi-class
    AUROC and AUPRC.
    """

    model.eval()

    results = {}
    y_pred_avpu, y_pred_crt = [], []
    y_true_avpu, y_true_crt = [], []

    epoch_loss = []

    for inputs, labels_avpu, labels_crt in dataloader:
        inputs = inputs.float()
        labels_avpu = labels_avpu.float()
        labels_crt = labels_crt.float()

        inputs = inputs.to(device)
        labels_avpu = labels_avpu.to(device)
        labels_crt = labels_crt.to(device)

        # Forward
        avpu_output, crt_output = model(inputs)
        loss_avpu = loss_fn_avpu(avpu_output, labels_avpu)
        loss_crt = loss_fn_crt(crt_output, labels_crt)

        batch_loss = (loss_avpu.item() + loss_crt.item()) / inputs.shape[0]
        epoch_loss.append(batch_loss)

        y_pred_avpu.append(avpu_output)
        y_pred_crt.append(crt_output)
        y_true_avpu.append(labels_avpu)
        y_true_crt.append(labels_crt)

    y_pred_avpu = torch.concat(y_pred_avpu).cpu()
    y_pred_crt = torch.concat(y_pred_crt).cpu()
    y_true_avpu = torch.concat(y_true_avpu).cpu()
    y_true_crt = torch.concat(y_true_crt).cpu()

    results['loss'] = np.mean(epoch_loss)
    results['accuracy'] = accuracy(y_pred_avpu, y_pred_crt, y_true_avpu, y_true_crt)
    results['avpu'] = calculate_metrics(y_pred_avpu, y_true_avpu, average=average)
    results['crt'] = calculate_metrics(y_pred_crt, y_true_crt, average=average)

    # AUROC
    auroc_avpu = calculate_auroc(y_pred_avpu, y_true_avpu, name='avpu', all_classes=False,
                                 average=average, visualize=visualize and args.verbose)
    auroc_crt = calculate_auroc(y_pred_crt, y_true_crt, name='crt', all_classes=False,
                                 average=average, visualize=visualize and args.verbose)

    # AUPRC
    auprc_avpu = calculate_auprc(y_pred_avpu, y_true_avpu, name='avpu', all_classes=False,
                                 average=average, visualize=visualize and args.verbose)
    auprc_crt = calculate_auprc(y_pred_crt, y_true_crt, name='crt', all_classes=False,
                                 average=average, visualize=visualize and args.verbose)

    results['avpu'].append(auroc_avpu)
    results['crt'].append(auroc_crt)
    results['avpu'].append(auprc_avpu)
    results['crt'].append(auprc_crt)

    return results


def print_results(results):
    """
    Print the results.
    """
    print("========================")
    print(f" Accuracy    : {results['accuracy'].round(4)}")
    print(f" AVPU:")
    print(f" - accuracy  : {results['avpu'][0].round(4)}")
    print(f" - precision : {results['avpu'][1].round(4)}")
    print(f" - recall    : {results['avpu'][2].round(4)}")
    print(f" - F1 score  : {results['avpu'][3].round(4)}")
    print(f" - AUROC     : {results['avpu'][4].round(4)}")
    print(f" - AUPRC     : {results['avpu'][5].round(4)}")
    print(f"\n CRT:")
    print(f" - accuracy  : {results['crt'][0].round(4)}")
    print(f" - precision : {results['crt'][1].round(4)}")
    print(f" - recall    : {results['crt'][2].round(4)}")
    print(f" - F1 score  : {results['crt'][3].round(4)}")
    print(f" - AUROC     : {results['crt'][4].round(4)}")
    print(f" - AUPRC     : {results['crt'][5].round(4)}")
    print("========================\n")


def train(args, model, dataloaders, device, loss_weights=None):
    """
    Main training function.
    """

    train_loader, val_loader, test_loader = dataloaders[0], dataloaders[1], dataloaders[2]

    model = model.to(device)
    loss_fn_avpu = nn.CrossEntropyLoss(weight=loss_weights[0]).to(device)
    loss_fn_crt = nn.CrossEntropyLoss(weight=loss_weights[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_model = None
    results = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'val_acc': [],
        'avpu': [],
        'crt': [],
        'test': [],
        'best_epoch': []
    }

    for e in range(args.epochs):
        print(f'\nStarting epoch {e}')
        epoch_loss = []
        model.train()

        for inputs, labels_avpu, labels_crt in tqdm(train_loader):
            inputs = inputs.float()
            labels_avpu = labels_avpu.float()
            labels_crt = labels_crt.float()

            inputs = inputs.to(device)
            labels_avpu = labels_avpu.to(device)
            labels_crt = labels_crt.to(device)

            # Forward
            optimizer.zero_grad()
            out_avpu, out_crt = model.forward(inputs)
            loss_avpu = loss_fn_avpu(out_avpu, labels_avpu)
            loss_crt = loss_fn_crt(out_crt, labels_crt)

            # Backpropagation
            loss_avpu.backward(retain_graph=True)
            loss_crt.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            batch_loss = (loss_avpu.item() + loss_crt.item()) / inputs.shape[0]
            epoch_loss.append(batch_loss)

        epoch_loss = np.mean(epoch_loss)

        # Validation
        val_results = evaluate(args, model, val_loader, device, loss_fn_avpu, loss_fn_crt)
        if args.verbose:
            print(f"Training loss: {round(epoch_loss, 3)}")
            print(f"Validation loss: {round(val_results['loss'], 3)}")
            print("\n== Validation results ==")
            print_results(val_results)

        results['train_loss'].append(epoch_loss)
        results['val_loss'].append(val_results['loss'])
        results['val_acc'].append(val_results['accuracy'])
        results['avpu'].append(val_results['avpu'])
        results['crt'].append(val_results['crt'])

        if not best_model or val_results['accuracy'] >= max(results['val_acc']):
            best_model = deepcopy(model)
            results['best_epoch'] = e

    # Test
    test_results = evaluate(args, best_model, test_loader, device, loss_fn_avpu,
                            loss_fn_crt, average='macro', visualize=True)

    results['test'] = test_results
    results['test_loss'] = test_results['loss']

    if args.verbose:
        print(f"Test loss: {round(test_results['loss'], 3)}")
        print("\n===== Test results =====")
        print_results(results['test'])

    return best_model, results


def plot_results(results):
    # Training loss
    L = len(results['val_acc'])
    plt.subplot(2, 2, 1)
    plt.plot(range(L), results['train_loss'])
    plt.plot(range(L), results['val_loss'])
    plt.scatter(results['best_epoch'], results['test']['loss'], color='red', marker='x')
    plt.ylabel('Average loss')
    plt.xlabel('Epochs')
    plt.legend(['Training loss', 'Validation loss', 'Test loss'])
    plt.title('Training and validation loss')

    # Validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(L), np.array(results['val_acc'])*100)
    plt.axhline(results['test']['accuracy']*100, color='red', linestyle='--')
    plt.scatter(results['best_epoch'], results['test']['accuracy']*100, color='red', marker='x')
    plt.ylim(-5, 105)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.legend(['Validation', 'Test at best epoch'])
    plt.title('Validation accuracy')

    # Metrics during validation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # AVPU
    plt.subplot(2, 2, 3)
    plt.plot(np.array(results['avpu'])[:, :4])
    for i, metric in enumerate(results['test']['avpu'][:4]):
        plt.scatter(results['best_epoch'], metric, color=colors[i], marker='x')
    plt.legend(['Accuracy', 'Precision', 'Recall', 'F1 score'], framealpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.ylabel('Score')
    plt.xlabel('Epochs')
    plt.title('Validation metrics for AVPU')

    # CRT
    plt.subplot(2, 2, 4)
    plt.plot(np.array(results['crt'])[:, :4])
    for i, metric in enumerate(results['test']['crt'][:4]):
        plt.scatter(results['best_epoch'], metric, color=colors[i], marker='x')
    plt.legend(['Accuracy', 'Precision', 'Recall', 'F1 score'], framealpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.ylabel('Score')
    plt.xlabel('Epochs')
    plt.title('Validation metrics for CRT')

    plt.tight_layout()
    plt.show()


def main(args):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    set_seed(args.seed)

    # Load dataset
    dataloaders, n_features, n_timesteps, loss_weights = load_data(args)

    # Load model
    model = load_model(args, n_features, n_timesteps)

    # Train model
    model, results = train(args, model, dataloaders, device, loss_weights)

    # Plot results
    plot_results(results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, required=True,
                        help="path to dataset file (from current directory)")
    parser.add_argument('--results_dir', action='store', type=str,
                        help="path to results directory (from current directory)")
    parser.add_argument('--verbose', action='store_true',
                        help="show model information and results")

    # Model configurations
    parser.add_argument('--model', action='store', type=str, required=True,
                        help='model that needs to be trained')

    # TCN configurations
    parser.add_argument('--num_channels', nargs='+', type=int, default=[200,200,100,100],
                        help="model configuration for the TCN")
    parser.add_argument('--kernel_size', action='store', default=16, type=int,
                        help="model configuration for the TCN")
    parser.add_argument('--dilations', nargs='+', type=int, default=None,
                        help="model configuration for the TCN")
    parser.add_argument('--dilation_reset', action='store', default=None, type=int,
                        help="model configuration for the TCN")
    parser.add_argument('--tcn_dropout', action='store', default=0, type=float,
                        help="model configuration for the TCN")
    parser.add_argument('--causal', action='store', default=True, type=bool,
                        help="model configuration for the TCN")
    parser.add_argument('--use_norm', action='store', default='layer_norm', type=str,
                        help="model configuration for the TCN")
    parser.add_argument('--activation', action='store', default='relu', type=str,
                        help="model configuration for the TCN")
    parser.add_argument('--kernel_initializer', action='store', default='xavier_uniform',
                        type=str, help="model configuration for the TCN")

    # LSTM configurations
    parser.add_argument('--num_layers', action='store', default=4, type=int,
                        help="model configuration for the LSTM")
    parser.add_argument('--hidden_size', action='store', default=512, type=int,
                        help="model configuration for the LSTM")
    parser.add_argument('--lstm_dropout', action='store', default=0, type=float,
                        help="model configuration for the LSTM")
    
    # Linear configuration
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[400,200,200,100],
                        help="model configuration for the linear model")
    parser.add_argument('--act_fn', action='store', default='relu', type=str,
                        help="model configuration for the linear model")
    parser.add_argument('--linear_dropout', action='store', default=0, type=float,
                        help="model configuration for the linear model")
    

    # Training hyperparameters
    parser.add_argument('-s', '--seed', action='store', default=42, type=int,
                        help="seed for reproducibility")
    parser.add_argument('--lr', action='store', default=1e-4, type=float,
                        help="learning rate for optimizer")
    parser.add_argument('--batch_size', action='store', default=16, type=int,
                        help="batch size for training")
    parser.add_argument('-e', '--epochs', action='store', default=30, type=int,
                        help="number of training epochs")

    args = parser.parse_args()

    main(args)
