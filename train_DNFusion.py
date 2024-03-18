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

from datasets_combined import create_dataloaders
from metrics import *
from models.DNFusion import DNFusion

MISSING = -1
AMPLIFIER = 100


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
        X = patient_data['clinical_X']
        X = np.where(np.isnan(X), -1, X)
        data[patient_id]['clinical_X'] = X

        X = patient_data['vital_X']
        X = np.where(np.isnan(X), -1, X)
        data[patient_id]['vital_X'] = X

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
    
    print(weights_avpu, weights_crt)

    return torch.Tensor(weights_avpu), torch.Tensor(weights_crt)


def load_data(args, data_split=[0.7, 0.1, 0.2]):
    """
    Load the dataset, perform data augmentations on it and create dataloaders.
    """

    with open(os.getcwd() + args.data_path, 'rb') as file:
        data = pickle.load(file)

    data = data_augmentations(data)

    clinical_input_shape = data[list(data.keys())[0]]['clinical_X'].shape
    vital_input_shape = data[list(data.keys())[0]]['vital_X'].shape
    n_features = [clinical_input_shape[2], vital_input_shape[2]]
    n_timesteps = [clinical_input_shape[1], vital_input_shape[1]]
    loss_weights = calculate_loss_weights(args, data)

    return create_dataloaders(data, data_split, args.batch_size), \
        n_features, n_timesteps, loss_weights


def load_model(args, n_features, n_timesteps):

    # As to avoid deprecation warning from pytorch_tcn TCN
    warnings.filterwarnings("ignore", category=UserWarning)

    dropout = 0

    model1_config = {
        'num_inputs': n_features[0],
        'num_timesteps': n_timesteps[0],
        'num_channels': [400, 200, 200, 100],
        'kernel_size': 4,
        'dilations': [1, 2, 3, 1],
        'dilation_reset': 3,
        'dropout': dropout,
        'causal': True,
        'use_norm': 'layer_norm',
        'activation': 'relu',
        'kernel_initializer':'xavier_uniform'
    }

    model2_config = {
        'num_inputs': n_features[1],
        'num_timesteps': n_timesteps[1],
        'num_channels': [200, 200, 100, 100],
        'kernel_size': 16,
        'dilations': None,
        'dilation_reset': None,
        'dropout': dropout,
        'causal': True,
        'use_norm': 'weight_norm',
        'activation': 'relu',
        'kernel_initializer':'xavier_uniform'
    }


    initial_mean = ((len(model1_config['num_channels']) - 1) / 2) / AMPLIFIER
    initial_std = 0.5 / AMPLIFIER
    # initial_std = 2*(np.sqrt(-1/(8*np.log(0.1)))) / mask_amplifier
    threshold = 0.1

    model = DNFusion(model1_config,
                     model2_config,
                     num_outputs=[4, 7],
                     mmtm_ratio=8,
                     mask_mean=initial_mean,
                     mask_std=initial_std,
                     threshold=threshold,
                     warmup=args.warmup,
                     fusion=True)

    # Re-activate the warnings
    warnings.filterwarnings("default", category=UserWarning)

    if args.verbose:
        print(f'Configuration:\n{model1_config}, {model2_config}')
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

    for clinical_input, vital_input, labels_avpu, labels_crt in dataloader:
        clinical_input = clinical_input.float()
        vital_input = vital_input.float()
        labels_avpu = labels_avpu.float()
        labels_crt = labels_crt.float()

        clinical_input = clinical_input.to(device)
        vital_input = vital_input.to(device)
        labels_avpu = labels_avpu.to(device)
        labels_crt = labels_crt.to(device)

        # Forward
        avpu_output, crt_output = model(clinical_input, vital_input)
        loss_avpu = loss_fn_avpu(avpu_output, labels_avpu)
        loss_crt = loss_fn_crt(crt_output, labels_crt)

        batch_loss = (loss_avpu.item() + loss_crt.item()) / clinical_input.shape[0]
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

    train_loader, val_loader, test_loader = dataloaders[0], dataloaders[1], dataloaders[2]

    model = model.to(device)

    loss_fn_avpu = nn.CrossEntropyLoss(weight=loss_weights[0]).to(device)
    loss_fn_crt = nn.CrossEntropyLoss(weight=loss_weights[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_model = None
    results = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'val_acc': [],
               'avpu': [], 'crt': [], 'test': [], 'best_epoch': []}

    # Load pre-trained MMTM module
    if f'mmtm_pretrained_e10_s{args.seed}' in os.listdir('./models/'):
        args.epochs -= args.warmup
        model.mmtm.load_state_dict(torch.load(f'./models/mmtm_pretrained_e10_s{args.seed}'))
        model.end_warmup()
    else:
        model.start_warmup()

    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    torch.autograd.set_detect_anomaly(True)

    for e in range(args.epochs):
        print(f'\nStarting epoch {e}')
        print(f"Mean: {round(model.mask.mean.item()*AMPLIFIER, 4)}\nStd : {round(model.mask.std.item()*AMPLIFIER, 4)}")
        print(f"{[round(model.mask(i).item(), 4) for i in range(4)]}")

        if model.warmup and e >= model.warmup:
            model.end_warmup()
            torch.save(model.mmtm.state_dict(), f'./models/mmtm_pretrained_e10_s{args.seed}')
        
        epoch_loss = []
        model.train()

        for clinical_input, vital_input, labels_avpu, labels_crt in tqdm(train_loader):
            clinical_input = clinical_input.float().to(device)
            vital_input = vital_input.float().to(device)
            labels_avpu = labels_avpu.float().to(device)
            labels_crt = labels_crt.float().to(device)

            # Forward
            optimizer.zero_grad()
            avpu_output, crt_output = model(clinical_input, vital_input)
            loss_avpu = loss_fn_avpu(avpu_output, labels_avpu)
            loss_crt = loss_fn_crt(crt_output, labels_crt)

            # Backpropagation
            loss_avpu.backward(retain_graph=True)
            loss_crt.backward(retain_graph=True)
            optimizer.step()

            # Average loss per element in batch
            epoch_loss.append((loss_avpu.item() + loss_crt.item()) / clinical_input.shape[0])


        if not model.warmup:
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
                            loss_fn_crt, average='macro', visualize=False)

    results['test'] = test_results
    results['test_loss'] = test_results['loss']

    if args.verbose:
        print(f"Test loss: {round(test_results['loss'], 3)}")
        print("\n===== Test results =====")
        print_results(results['test'])

    print(f"Mean: {round(model.mask.mean.item()*100, 4)}\nStd : {round(model.mask.std.item()*100, 4)}")
    print(f"Mask: {[round(model.mask(i).item(), 4) for i in range(4)]}")

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
        if metric != 'loss':
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
        if metric != 'loss':
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

    # Training hyperparameters
    parser.add_argument('-s', '--seed', action='store', default=42, type=int,
                        help="seed for reproducibility")
    parser.add_argument('--lr', action='store', default=1e-4, type=float,
                        help="learning rate for optimizer")
    parser.add_argument('--batch_size', action='store', default=64, type=int,
                        help="batch size for training")
    parser.add_argument('-e', '--epochs', action='store', default=25, type=int,
                        help="number of training epochs (this includes warmup epochs)")
    parser.add_argument('--warmup', action='store', default=10, type=int,
                        help="number of epochs the fusion module will warmup/train")

    args = parser.parse_args()

    main(args)

