from utils import pickle_in, get_root
import torch
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from operator import add
from collections import deque

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(*data, device=None):
    if device is None:
        device = get_default_device()
    """Move tensor(s) to chosen device"""
    if len(data) > 1:
        return [x.to(device, non_blocking=True) for x in data]
    return data[0].to(device, non_blocking=True)


class ShallowNN(nn.Module):
    def __init__(self, n_variables):
        super().__init__()
        self.lin1 = nn.Linear(n_variables, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 5)
        self.lin4 = nn.Linear(5, 1)
        self.bn1 = nn.BatchNorm1d(n_variables)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.bn4 = nn.BatchNorm1d(5)
        self.drops = nn.Dropout(0.3)

    def forward(self, x):
        x = self.bn1(x.float())
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = F.relu(self.lin3(x))
        x = self.drops(x)
        x = self.bn4(x)
        x = self.lin4(x)
        return x


def get_optimizer(model, lr=0.001, wd=0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc


class Trainer(object):
    def __init__(self, model, metrics: dict, verbose=True):
        self.model = model
        self.performance_metrics = metrics
        self.verbose = verbose
        #TODO: Proper patience!
        self.lowest_loss = {'acc': 0, 'loss': np.inf}
        self.step_counter = 0
        self.best_state = deepcopy(model.state_dict())
        self.color = 'bgrcmyk'[np.random.randint(0, 7)]

    def save_state(self):
        self.best_state = deepcopy(model.state_dict())

    def reinstate(self):
        self.model.load_state_dict(self.best_state)


    def train(self, X, y):
        if not self.model.training:
            model.train()
        y_pred = self.model(X)
        performance_list = []
        for name, (metric, train_perf, _) in self.performance_metrics.items():
            performance = metric(y_pred, y)
            if name == 'loss':
                performance.backward()
            train_perf.append(performance.item())
        # One metric should be called 'loss'
        optimizer.step()
        self.step_counter += 1
        if self.verbose:
            print(self.step_counter, 'TRAIN:', *[(name, train_perf[-1]) for name, (_, train_perf, _) in trainer.performance_metrics.items()])


    def eval(self, X, y):
        has_improved = False
        if self.model.training:
            model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
            for name, (metric, _, eval_perf) in self.performance_metrics.items():
                performance = metric(y_pred, y)
                eval_perf.append(performance.item())

            # TODO: Proper patience!
            cur_acc, cur_loss = self.performance_metrics['acc'][-1][-1], self.performance_metrics['loss'][-1][-1]
            if cur_acc >= self.lowest_loss['acc'] and cur_loss  < self.lowest_loss['loss']:
                self.lowest_loss['acc'], self.lowest_loss['loss'] = cur_acc, cur_loss
                has_improved = True

        if self.verbose:
            print(self.step_counter, 'EVAL:', *[(name, eval_perf[-1]) for name, (_, _, eval_perf) in trainer.performance_metrics.items()])
        return has_improved


    def show(self):
        num_subplots = len(self.performance_metrics)
        for i, (name, (m, tp, ep)) in enumerate(self.performance_metrics.items()):
            plt.subplot(2, num_subplots, 1 + i * 2)
            plt.plot(tp, c=self.color)
            plt.title('training ' + name)
            plt.subplot(2, num_subplots, 2 + i * 2)
            plt.plot(ep, c=self.color)
            plt.title('evaluation ' + name)
        plt.show()

    def deserialize(self):
        return {'lowest_loss':         self.lowest_loss,
                'model':               to_device(self.model,'cpu'),
                'performance_metrics': self.performance_metrics}


def cross_val_split(indices, test_size, mode='separate'):
    indices = list(indices)
    n_indices = len(indices)
    min_fold_size = int(np.floor(n_indices * test_size))

    if mode == 'comprehensive':
        items = deque(indices)
        result = []
        for i in range(n_indices):
            result.append(list(items)[:min_fold_size])
            items.rotate(1)
        return result
    elif mode == 'separate':
        """Split the list of indices into n folds of test data."""
        n_folds = int(np.floor(n_indices / min_fold_size))
        rem = int(n_indices % min_fold_size)
        fold_sizes = [int(el) for el in (map(add, [min_fold_size]*n_folds, [1]*rem + [0]*(n_folds-rem)))]
        cumsum_fold_sizes = list(np.cumsum(fold_sizes))
        result = []
        state = 0
        for size in cumsum_fold_sizes:
            result.append(indices[state:size])
            state = size
        return result
    else:
        raise ValueError('I did not quite catch the method to be used.')


def train_test_split2(X, y, test_idx):
    train_idx = [idx for idx in X.index if idx not in test_idx]
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test,  y_test =  X.loc[test_idx],  y.loc[test_idx]
    return X_train, X_test, y_train, y_test


# Load data
X = pickle_in(get_root('data', 'radiomics_dataframes', 'advanced_radiomic_dataframe_DTI_freesurfer-smol.pkl'))
y = X.is_deprived
X = X.drop('is_deprived', axis=1)



# Partition Cross Validation Fold
test_idxs = cross_val_split(X.index, test_size=0.25, mode='comprehensive')
batch_size = 32

fold_result = []
for test_idx in test_idxs:
    break
    partitions = train_test_split2(X, y, test_idx)
    partitions = [torch.tensor(prt.values) for prt in partitions]
    X_train, X_test, y_train, y_test = to_device(*partitions)
    y_train, y_test = [prt.unsqueeze(1).type(torch.float) for prt in (y_train, y_test)]

    # Setup model
    model = ShallowNN(len(X.columns))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model, criterion = to_device(model, criterion)
    metrics = {'loss': (criterion, [], []), 'acc': (binary_acc, [], [])}
    trainer = Trainer(model, metrics, verbose=False)
    patience = 0
    # Train
    for i in range(5000):
        r = torch.randperm(len(X_train))
        # random permutation of training data
        X_train, y_train = X_train[r], y_train[r]
        # select first half for batch
        trainer.train(X_train[:batch_size], y_train[:batch_size])
        has_improved = trainer.eval(X_test, y_test)
        if has_improved:
            trainer.save_state()
            print(i, 'We just improved!')
        else:
            patience += 1
            if patience >= 100:
                patience = 0
                trainer.reinstate()
                print(i, 'we have to retreat...')
        if not np.remainder(i, 500) and i:
            trainer.show()
    trainer.reinstate()
    trainer.verbose = True
    trainer.eval(X_test, y_test)
    trainer.show()
    fold_result.append(deepcopy(trainer))
print("Accuracy: {:.1%}".format(np.mean([f.lowest_loss['acc'] for f in fold_result])) + "±{:.1%}".format(np.std([f.lowest_loss['acc'] for f in fold_result])))


full_file = r"D:\repositories\SLEEEP\scripts\7prediction\vault\experiment1\fold_results_42_fold_DTI_freesurfer-smol.pkl"
fold_result = pickle_in(full_file)