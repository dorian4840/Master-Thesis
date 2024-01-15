import os
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            confusion_matrix, f1_score
from copy import deepcopy
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.tcn_bai_torch import TCN
from preprocessing.datasets import IMPALA_Dataset, create_dataloaders


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def load_model(args, num_input, num_timesteps, num_output):
    """
    Load the Temporal Convolutional Network based on Bai et al.
    https://arxiv.org/pdf/1803.01271.pdf

    Model is modified to with extra linear layers at the end to flatten the output
    of the TCN and get the output to a desired length.
    """

    model = TCN(
        num_input=num_input,
        num_timesteps=num_timesteps,
        num_channels=[250, 100, 50],
        kernel_size=3,
        dilations=[1, 2, 3],
        dilation_reset=3,
        dropout=0.2,
        causal=True,
        use_norm='weight_norm',
        activation='relu',
        kernel_initializer='xavier_uniform',
        use_skip_connections=False,
        input_shape='NCL',
        hidden_dims=[100, 25],
        num_output=num_output
    )
    return model


def print_results(results):
    """
    Print the results from the evaluation function.
    """
    print("========================")
    print(f" Accuracy    : {results['overall_acc'].round(4)}")
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


def evaluate(model, dataloader, device):
    """
    Calculate the accuracy, precision and recall of the model.
    """

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):

            # if i > 2:
            #     break

            X = X.float().to(device)
            y = y.float().to(device)
        
            predictions.append(model.forward(X))
            labels.append(y)


    predictions = torch.concat(predictions).cpu().round() # Turn probability to binary
    labels = torch.concat(labels).cpu()

    results = {'overall_acc' : accuracy_score(labels.flatten(), predictions.flatten()).round(4)}

    # Metrics
    for i, outcome in enumerate(['avpu', 'crt']):
        accuracy = accuracy_score(labels[:, i], predictions[:, i])
        precision = precision_score(labels[:, i], predictions[:, i], zero_division=0.0)
        recall = recall_score(labels[:, i], predictions[:, i], zero_division=0.0)
        f1 = f1_score(labels[:, i], predictions[:, i], zero_division=0.0)
        # matrix = confusion_matrix(labels[:, outcome], predictions[:, outcome])

        results[outcome] = [accuracy, precision, recall, f1]
    
    return results


def mask_missing_values(X, masking_value):
    """
    Replace the missing values (-999) with a masking value.
    """

    X = torch.where(X == -1, masking_value, X)
    return X


def labels_to_binary(y):
    """
    Turn the output labels to binary differentiating between normal and abnormal
    values (AVPU > 1, CRT > 3).
    """
    if isinstance(y, np.ndarray):
        y[:, 0] = np.where(y[:, 0] > 1, 1, 0) # 0 = normal; 1 = abnormal
        y[:, 1] = np.where(y[:, 1] > 3, 1, 0) # 0 = normal; 1 = abnormal
    else:
        y[:, 0] = torch.where(y[:, 0] > 1, 1, 0) # 0 = normal; 1 = abnormal
        y[:, 1] = torch.where(y[:, 1] > 3, 1, 0) # 0 = normal; 1 = abnormal
    return y


def train(args):
    """
    Training function for the model.

    TODO:
    - Add lr scheduler and early stop
    """

    # As to avoid deprecation warning from pytorch_tcn TCN
    warnings.filterwarnings("ignore", category=UserWarning)

    # Seed functions
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    set_seed(args.seed)

    # Initialize dataloaders
    file_path = f"{os.getcwd()}/{args.data_dir}"
    dataset = IMPALA_Dataset(file_path)
    dataset.y = labels_to_binary(dataset.y) # Convert dataset to binary

    # Balance the dataset
    avpu_1, crt_1 = [], []
    for i, d in enumerate(dataset):
        if d[1][0] == 1:
            avpu_1.append(i)
        if d[1][1] == 1:
            crt_1.append(i)

    avpu_0 = list(np.random.choice([i for i in range(len(dataset)) if i not in avpu_1], len(avpu_1), replace=False))
    crt_0 = list(np.random.choice([i for i in range(len(dataset)) if i not in crt_1], len(crt_1), replace=False))

    dataset.X = dataset.X[list(set(avpu_0 + avpu_1 + crt_1 + crt_0))]
    dataset.y = dataset.y[list(set(avpu_0 + avpu_1 + crt_1 + crt_0))]

    train_loader, val_loader, test_loader = \
        create_dataloaders(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize model, loss function, optimizer, etc.
    input_shape, output_shape = dataset[0][0].shape, dataset[0][1].shape
    model = load_model(args, input_shape[0], input_shape[1],
                       output_shape[0]).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Save statistics
    best_model = None
    best_epoch = 0
    model_results = {
        'val_acc' : [],
        'train_loss' : [],
        'avpu' : [],
        'crt' : []
    }

    ### Training ###
    for e in range(args.epochs):
        print(f"Starting epoch {e}")
        epoch_loss = 0
        model.train()

        for i, (X, y) in enumerate(tqdm(train_loader)):
            X = X.float().to(device)
            y = y.float().to(device)

            # Calculate output
            optimizer.zero_grad()
            output = model.forward(X)
            loss = loss_fn(output, y)

            # Compute gradient
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if i == 0:
            #     print(y, torch.round(output, decimals=2))


        model_results['train_loss'].append(epoch_loss/len(train_loader))

        # Evaluation
        results = evaluate(model, val_loader, device)
        if args.verbosity:
            print("\nEvaluating model...")
            print_results(results)
        model_results['val_acc'].append(results['overall_acc'])
        model_results['avpu'].append(results['avpu'])
        model_results['crt'].append(results['crt'])
        
        # Save best model
        if not best_model or results['overall_acc'] >= max(model_results['val_acc']):
            # print(f"Better model found at epoch {e}")
            best_model = deepcopy(model)
            best_epoch = e

    ### Testing ###
    print("\nTesting model...")
    results = evaluate(best_model, test_loader, device)
    print_results(results)
    model_results['test'] = results

    # Save best model and model results
    if args.save_model:
        model_path = f"{os.getcwd()}/{args.results_dir}"
        model_name = f"{args.model_name}_s{args.seed}_lr{args.lr}_bs{args.batch_size}_e{args.epochs}"
        torch.save(model.state_dict(), f"{model_path}saved_models/{model_name}.pt")
        with open(f"{model_path}stats_models/{model_name}.pkl", "wb") as file:
            pickle.dump(model_results, file)

    return best_model, best_epoch, model_results


def plot_results(results, best_epoch):
    """
    Create plots of the results from training.
    """

    # Training loss and validation accuracy
    L = len(results['val_acc'])
    plt.subplot(2, 2, 1)
    plt.plot(range(L), results['train_loss'])
    plt.ylabel("Training loss")
    plt.xlabel("Epochs")
    plt.title("Training loss")
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.plot(range(L), np.array(results['val_acc'])*100)
    plt.axhline(results['test']['overall_acc']*100, color='red')
    plt.scatter(best_epoch, results['test']['overall_acc']*100, color='red', marker='x')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epochs")
    plt.legend(['Validation', 'Test at best epoch'])
    plt.title("Validation accuracy")
    plt.tight_layout()

    # Precision, recall and F1 score for AVPU and CRT
    plt.subplot(2, 2, 3)
    plt.plot(np.array(results['avpu']))
    plt.legend(['Accuracy', 'Precision', 'Recall', 'F1 score'])
    plt.xlabel("Epochs")
    plt.ylabel("Metric score")
    plt.title('AVPU')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(np.array(results['crt']))
    plt.legend(['Accuracy', 'Precision', 'Recall', 'F1 score'])
    plt.xlabel("Epochs")
    plt.ylabel("Metric score")
    plt.title('CRT')
    plt.tight_layout()

    plt.show()


def main(args):
    """
    Main function. Takes care of training the model, loading saved models,
    saving trained models.
    TODO:
    - Create if-statement to load a saved model.
    - Run model on different seeds.
    """

    model, best_epoch, results = train(args)

    # model.load_state_dict(torch.load(checkpoint_name))

    if args.verbosity:
        plot_results(results, best_epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", action='store', default=42,
                        type=int, help="seed")
    parser.add_argument("--lr", action='store', default=1e-3,
                        type=float, help="learning rate of the optimizer")
    parser.add_argument("--batch_size", action='store', default=32,
                        type=int, help="Batch size for the dataloaders")
    parser.add_argument("-e", "--epochs", action='store', default=50,
                        type=int, help="Max number of epochs")
    parser.add_argument("--data_dir", action='store', required=True, type=str,
                        help="Directory where data is stored")
    parser.add_argument("--results_dir", action='store', required=True, type=str,
                        help="Directory to store best model")
    parser.add_argument("--model_name", action='store', required=True, type=str,
                        help="Name of the best model")
    parser.add_argument("--verbosity", action="store_true",
                        help="print extra information")
    parser.add_argument("--save_model", action="store_true",
                        help="save the model and its results")

    args = parser.parse_args()

    assert os.path.isfile(args.data_dir+"_input"), "Input file does not exist"
    assert os.path.isfile(args.data_dir+"_labels"), "Labels file does not exist"
    assert os.path.isdir(args.results_dir), "Results directory does not exist"

    main(args)
