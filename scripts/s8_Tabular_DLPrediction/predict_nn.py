from utils import pickle_in, get_root, pickle_out
import os
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
import random
from scipy.stats import friedmanchisquare
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import iqr
from glob import glob
import scikit_posthocs as sp
import pandas as pd
from scipy.stats import shapiro


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
            print(self.step_counter, 'TRAIN:',
                  *[(name, train_perf[-1]) for name, (_, train_perf, _) in trainer.performance_metrics.items()])

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
            if cur_acc >= self.lowest_loss['acc'] and cur_loss < self.lowest_loss['loss']:
                self.lowest_loss['acc'], self.lowest_loss['loss'] = cur_acc, cur_loss
                has_improved = True

        if self.verbose:
            print(self.step_counter, 'EVAL:',
                  *[(name, eval_perf[-1]) for name, (_, _, eval_perf) in trainer.performance_metrics.items()])
        return has_improved

    def show(self, suptitle=''):
        num_subplots = len(self.performance_metrics)
        for i, (name, (m, tp, ep)) in enumerate(self.performance_metrics.items()):
            plt.subplot(2, num_subplots, 1 + i * 2)
            plt.plot(tp, c=self.color)
            plt.title('training ' + name)
            plt.subplot(2, num_subplots, 2 + i * 2)
            plt.plot(ep, c=self.color)
            plt.title('evaluation ' + name)
            plt.suptitle(suptitle)
        plt.show()

    def deserialize(self):
        return {'lowest_loss': self.lowest_loss,
                'model': self.model.to('cpu'),
                'performance_metrics': self.performance_metrics}


def cross_val_split(indices, tst_sz=0.25, val_sz=0.25, mode='separate', **kwargs):
    # sample IDs
    indices = list(indices)
    # number of samples
    n_indices = len(indices)
    # number of samples in test set
    min_t_fold_size = int(np.floor(n_indices * tst_sz))
    min_v_fold_size = int(np.floor(n_indices * val_sz))

    test_folds, val_folds = [], []
    items = deque(indices)
    if mode == 'comprehensive':
        for i in range(n_indices):
            test_folds.append(list(items)[:min_t_fold_size])
            val_folds.append(list(items)[:min_v_fold_size])
            items.rotate(1)
        return test_folds, val_folds
    elif mode == 'separate':
        """Split the list of indices into n folds of test data."""
        n_folds = int(np.floor(n_indices / min_t_fold_size))
        rem = int(n_indices % min_t_fold_size)
        fold_sizes = [int(el) for el in (map(add, [min_t_fold_size] * n_folds, [1] * rem + [0] * (n_folds - rem)))]
        cumsum_fold_sizes = list(np.cumsum(fold_sizes))
        state = 0
        for size in cumsum_fold_sizes:
            test_folds.append(indices[state:size])
            val_folds.append(list(items)[:min_v_fold_size])
            state = size
        return test_folds, val_folds
    else:
        raise ValueError('I did not quite catch the method to be used.')


def train_val_test_split(X, y, val_idx, test_idx):
    train_idx = [idx for idx in X.index if idx not in test_idx and idx not in val_idx]
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    return [X_train, X_val, X_test], [y_train, y_val, y_test]


# Load data
do_overwrite = False
test_size = 0.20
val_size = 0.10
tools = ['samseg', 'vuno', 'freesurfer', 'fastsurfer', ]
source_f_name_regex = '{}_DTI_{}{}.pkl'

for do_ft_select, is_advanced, feature_set_name in zip([False, True], ['advanced_', ''], ['-smol', '']):
    for tool_n, tool in enumerate(tools):
        fname = get_root('scripts', 's8_Tabular_DLPrediction', 'results',
                         source_f_name_regex.format('20220615-nn-predict-results-for-{}'.format(tool),
                                                    feature_set_name.replace('-', ''), ''))
        if os.path.isfile(fname):
            continue

        print(f'Started on "{tool}", tool {tool_n} of {len(tools)}')
        source_f_name_regex = '{}_DTI_{}{}.pkl'
        source_f_name = source_f_name_regex.format(is_advanced + 'radiomic_dataframe', tool, f'{feature_set_name}')
        source_path = get_root('data', 'radiomics_dataframes', source_f_name)

        X = pickle_in(source_path)
        y = X.pop('is_deprived')

        if do_ft_select:
            X.dropna(axis=1, inplace=True)

        # Partition Cross Validation Fold
        test_idxs, val_idxs = cross_val_split(X.index, test_size=test_size, val_size=val_size, mode='comprehensive')
        batch_size = 16

        fold_result = []
        print(f'0/{len(test_idxs)}')
        for fold_n, (test_idx, val_idx) in enumerate(zip(test_idxs, val_idxs)):
            torch.manual_seed(fold_n)
            random.seed(fold_n)
            np.random.seed(fold_n)

            Xs, ys = train_val_test_split(X, y, val_idx, test_idx)

            if do_ft_select:
                ch2 = SelectKBest(chi2, k=500)
                lower_bound = Xs[0].min().min()
                X_train, X_val, X_test = [x - lower_bound for x in Xs]
                X_train = ch2.fit_transform(X_train, ys[0])
                X_val = ch2.transform(X_val - lower_bound)
                X_test = ch2.transform(X_test - lower_bound)
                Xs = (X_train, X_val, X_test)
                partitions = (*Xs, *ys)
            partitions = [prt.values if hasattr(prt, 'values') else prt for prt in [*Xs, *ys]]
            partitions = [torch.tensor(prt) for prt in partitions]
            # To GPU
            X_train, X_val, X_test, y_train, y_val, y_test = to_device(*partitions)
            y_train, y_val, y_test = [prt.unsqueeze(1).type(torch.float) for prt in (y_train, y_val, y_test)]

            # Setup model
            model = ShallowNN(X_train.shape[-1])

            # l1_crit = nn.L1Loss(reduction='sum')
            # reg_loss = 0
            # for param in model.parameters():
            #    reg_loss += l1_crit(param)
            #
            # factor = 0.001
            # loss += factor * reg_loss

            n_epochs = 500
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=10 ** -3)
            model, criterion = to_device(model, criterion)
            metrics = {'loss': (criterion, [], []), 'acc': (binary_acc, [], [])}
            trainer = Trainer(model, metrics, verbose=False)
            patience = 0
            # Train
            for i in range(n_epochs):
                # random permutation of samples of training data
                r = torch.randperm(len(X_train))
                X_train, y_train = X_train[r], y_train[r]
                # select first half for batch
                trainer.train(X_train[:batch_size], y_train[:batch_size])
                # see improvement on validation set
                has_improved = trainer.eval(X_val, y_val)
                if has_improved:
                    trainer.save_state()
                else:
                    patience += 1
                    if patience >= 100:
                        patience = 0
                        trainer.reinstate()
            trainer.reinstate()
            trainer.verbose = True
            trainer.eval(X_test, y_test)

            trainer.show(suptitle=f'{["manual", "chi2"][int(do_ft_select)]},'
                                  f' {tool}, fold n={fold_n}')
            fold_result.append(deepcopy(trainer))
            print(f'{fold_n + 1}/{len(test_idxs)}')

        if not os.path.isfile(fname) or do_overwrite:
            pickle_out([f.lowest_loss for f in fold_result], fname)
        print(
            f'Accuracy: {np.mean([f.lowest_loss["acc"] for f in fold_result]):.1%}±{np.std([f.lowest_loss["acc"] for f in fold_result]):.1%}')

for do_ft_select, is_advanced, feature_set_name in zip([False, True], ['advanced_', ''], ['-smol', '']):
    accs = []
    losss = []
    for tool in tools:
        fold_result = pickle_in(get_root('scripts', 's8_Tabular_DLPrediction', 'results',
                                         source_f_name_regex.format('20220615-nn-predict-results-for-{}'.format(tool), feature_set_name.replace('-',''), '')))
        res = [f['acc'] for f in fold_result]
        res2 = [f['loss'] for f in fold_result]
        accs.append(res)
        losss.append(res2)
        _, s_p = shapiro(res2)
        # print('Shapiro Statistics=%.3f, p=%.3f' % (stat, p))
        if s_p < 0.05:
            tag = 'not gaussian '
        else:
            tag = '    gaussian '
        aq3, aq1 = np.percentile(res, [75, 25])
        lq3, lq1 = np.percentile(res2, [75, 25])
        print(['On feature subset:', 'On all features:  '][int(do_ft_select)], tag,
              tool.ljust(10), f"loss: {np.mean(res2):.2f} ({lq1:.2f}-{lq3:.2f}), accuracy: {np.mean(res):.1%} ({aq1:.1%}-{aq3:.1%})")
    _, f_p = friedmanchisquare(*losss)
    if f_p < 0.05:
        print('Friedman says groups are different')
    else:
        print('Friedman says groups are NOT different')

    ddff = pd.DataFrame(np.array(losss).T, columns=tools)
    print(sp.posthoc_nemenyi_friedman(ddff))

    print('')


files = glob(get_root('scripts', 's8_Tabular_DLPrediction', 'results', '*.pkl'))
for selec in ['smol.pkl', '_.pkl']:
    for tool in ['samseg', 'vuno', 'freesurfer', 'fastsurfer']:
        for f in files:
            if '20220615' in f and tool in f and selec in f:
                fold_acc = [x['acc'] for x in pickle_in(f)]
                fold_loss = [x['loss'] for x in pickle_in(f)]
                print(f'{os.path.basename(f).ljust(60)}'
                      f'{np.mean(fold_loss):.5f} '
                      f'({np.mean(fold_loss) - iqr(fold_loss):.5f}-'
                      f'{np.mean(fold_loss) + iqr(fold_loss):.5f}) | '
                      f'{np.mean(fold_acc):.3%} '
                      f'({np.mean(fold_acc) - iqr(fold_acc):.3%}-'
                      f'{np.mean(fold_acc) + iqr(fold_acc):.3%})')

for metric in ['acc', 'loss']:
    for f in files:
        fold_results = [x[metric] for x in pickle_in(f)]
        print(f'{metric.ljust(5)}'
              f'{os.path.basename(f).ljust(50)}'
              f'{np.median(fold_results):.2f}'
              f'({np.median(fold_results) - iqr(fold_results):.2f}-'
              f'{np.median(fold_results) + iqr(fold_results):.2f})')