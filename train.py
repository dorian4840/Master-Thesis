import os
import argparse
import warnings
import json
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

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

    TODO:
    - laat werken voor binary
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

    with open(os.getcwd() + args.config_file) as file:
        config = json.loads(file.read())

    if args.model == 'lstm':
        model = LSTM(name=config['name'],
                     num_input=n_features,
                     hidden_size=config['hidden_size'],
                     num_layers=config['num_layers'],
                     num_output=[4, 7],
                     dropout=config['dropout'],
                     final_activation='softmax')

    elif args.model == 'tcn':

        # As to avoid deprecation warning from pytorch_tcn TCN
        warnings.filterwarnings("ignore", category=UserWarning)

        model = TCN(
            name=config['name'],
            num_input=n_features,
            num_timesteps=n_timesteps,
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dilations=config['dilations'],
            dilation_reset=config['dilation_reset'],
            dropout=config['dropout'],
            causal=config['causal'],
            use_norm=config['use_norm'],
            activation=config['activation'],
            kernel_initializer=config['kernel_initializer'],
            use_skip_connections=config['use_skip_connections'],
            input_shape=config['input_shape'],
            hidden_dims=config['hidden_dims'],
            num_output=[4, 7],
            final_activation=config['final_activation'],
        )

        # Re-activate the warnings
        warnings.filterwarnings("default", category=UserWarning)

    elif args.model == 'lstm_binary':
        pass

    elif args.model == 'tcn_binary':
        pass

    elif args.model == 'linear':
        model = Linear(n_features, n_timesteps)

    else:
        raise ValueError('model unknown')

    if args.verbose:
        print(f'Configuration:\n{config}')
        print(f'Model:\n{model}')

    return model


@torch.no_grad()
def evaluate(args, model, dataloader, device, average='micro', visualize=False):
    """
    Evaluate the model on accuracy, precision, recall, F1 score and multi-class
    AUROC and AUPRC.
    """

    model.eval()

    results = {}
    y_pred_avpu, y_pred_crt = [], []
    y_true_avpu, y_true_crt = [], []

    for inputs, labels_avpu, labels_crt in dataloader:
        inputs = inputs.float()
        labels_avpu = labels_avpu.float()
        labels_crt = labels_crt.float()

        inputs = inputs.to(device)

        # Forward
        out_avpu, out_crt = model.forward(inputs)

        y_pred_avpu.append(out_avpu)
        y_pred_crt.append(out_crt)
        y_true_avpu.append(labels_avpu)
        y_true_crt.append(labels_crt)

    y_pred_avpu = torch.concat(y_pred_avpu).cpu()
    y_pred_crt = torch.concat(y_pred_crt).cpu()
    y_true_avpu = torch.concat(y_true_avpu)
    y_true_crt = torch.concat(y_true_crt)

    results['accuracy'] = accuracy(y_pred_avpu, y_pred_crt, y_true_avpu, y_true_crt)
    results['avpu'] = calculate_metrics(y_pred_avpu, y_true_avpu, average=average)
    results['crt'] = calculate_metrics(y_pred_crt, y_true_crt, average=average)

    # AUROC
    auroc_avpu = calculate_auroc(y_pred_avpu, y_true_avpu, name='avpu', visualize=visualize and args.verbose)
    auroc_crt = calculate_auroc(y_pred_crt, y_true_crt, name='crt', visualize=visualize and args.verbose)

    # # AUPRC
    auprc_avpu = calculate_auprc(y_pred_avpu, y_true_avpu, name='avpu', visualize=visualize and args.verbose)
    auprc_crt = calculate_auprc(y_pred_crt, y_true_crt, name='crt', visualize=visualize and args.verbose)

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
    print(f"\n CRT:")
    print(f" - accuracy  : {results['crt'][0].round(4)}")
    print(f" - precision : {results['crt'][1].round(4)}")
    print(f" - recall    : {results['crt'][2].round(4)}")
    print(f" - F1 score  : {results['crt'][3].round(4)}")
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
    # scheduler = StepLR(optimizer, step_size=15, gamma=0.5) # Best so far

    best_model = None
    results = {
        'train_loss': [],
        'val_acc': [],
        'avpu': [],
        'crt': [],
        'test': [],
        'best_epoch': []
    }

    for e in range(args.epochs):
        print(f'\nStarting epoch {e}')
        epoch_loss = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # print(loss_avpu.item(), loss_crt.item())
            epoch_loss += (loss_avpu.item() + loss_crt.item())

        # if e < 16:
        #     scheduler.step()

        # Validation
        val_results = evaluate(args, model, val_loader, device,
                               average='weighted', visualize=False)
        if args.verbose:
            print("\n== Validation results ==")
            print_results(val_results)

        results['train_loss'].append(epoch_loss)
        results['val_acc'].append(val_results['accuracy'])
        results['avpu'].append(val_results['avpu'])
        results['crt'].append(val_results['crt'])

        if not best_model or val_results['accuracy'] >= max(results['val_acc']):
            best_model = deepcopy(model)
            results['best_epoch'] = e

    # for name, param in model.state_dict().items():
    #     with open(f'{name}.npy', 'wb') as f:
    #         np.save(f, param.cpu().numpy())

    # Test
    results['test'] = evaluate(args, best_model, test_loader, device,
                               average='weighted', visualize=True)
    if args.verbose:
        print("\n===== Test results =====")
        print_results(results['test'])

    # TODO: Save model and results

    return best_model, results


def plot_results(results):
    # Training loss
    L = len(results['val_acc'])
    plt.subplot(2, 2, 1)
    plt.plot(range(L), results['train_loss'])
    plt.ylabel('Training loss')
    plt.xlabel('Epochs')
    plt.title('Training loss')

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
    plt.plot(np.array(results['avpu']))
    for i, metric in enumerate(results['test']['avpu']):
        plt.scatter(results['best_epoch'], metric, color=colors[i], marker='x')
    plt.legend(['Accuracy', 'Precision', 'Recall', 'F1 score'], framealpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.ylabel('Score')
    plt.xlabel('Epochs')
    plt.title('Validation metrics for AVPU')

    # CRT
    plt.subplot(2, 2, 4)
    plt.plot(np.array(results['crt']))
    for i, metric in enumerate(results['test']['crt']):
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

    # Set seed
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
    parser.add_argument('--save_model', action='store_true',
                        help="save the best model and results")
    parser.add_argument('--verbose', action='store_true',
                        help="show model information and results")

    # Data augmentation
    parser.add_argument('--binary', action='store_true',
                        help='turn the training labels to binary')

    # Model configuration
    parser.add_argument('--model', action='store', type=str, required=True,
                        help='model that needs to be trained')
    parser.add_argument('--config_file', action='store', type=str, required=True,
                        help="configuration file for the model's hyperparameters")

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

    # Check if paths are valid
    assert os.path.isfile(os.getcwd() + args.data_path), 'dataset file not found'
    assert (args.save_model and args.results_dir) or (not args.save_model and not args.results_dir), \
        'save_model and results_dir must be both active or inactive'

    if args.results_dir:
        assert os.path.isdir(f'{args.results_dir}/model_stats'), 'model stats dir not found'
        assert os.path.isdir(f'{args.results_dir}/saved_models'), 'saved models dir not found'

    main(args)
