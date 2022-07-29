import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from utils import get_root, pickle_in
from matplotlib import pyplot as plt
from utils import flatten
import numpy as np
from utils import freeSurfer_lut

'''
saves confusion matrices as png file 
to ConfusionMatrics.png by default
'''

def show_confusion(axis, array, ticks=['s', 'a', 'f', 'v'], do_xtick=True, do_ytick=True):
    plt.imshow(array)

    # place ticks
    if do_xtick:
        plt.xticks(np.arange(len(ticks)), ticks, rotation=90)
    else:
        plt.xticks(np.arange(len(ticks)), [''] * len(ticks))
    if do_ytick:
        plt.yticks(np.arange(len(ticks)), ticks)
    else:
        plt.yticks(np.arange(len(ticks)), [''] * len(ticks))
    plt.ylim([3.5, -0.5])

    # place text in cell
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            if array[x, y] == 1:
                txt = '1.'
            else:
                txt = "{:.2f}".format(array[x, y])
                if txt[0] == '0':
                    txt = txt[1:]
                else:
                    txt = txt[:-1]
            if array[x, y] < 0.8:
                c = 'w'
            else:
                c = 'k'
            plt.text(x, y, txt, {'c': c,
                                 'ha': 'center',
                                 'va': 'center',
                                 'fontsize': 'small'})
    plt.clim([0.6, 1])
    

    ax.xaxis.tick_top()
    ax.set_aspect('equal', adjustable='box')
    #plt.set_cmap('RdYlGn')
    plt.set_cmap('viridis')


data = pickle_in(get_root('scripts', 's5_Segmentations_ScoringAndVisualization', 'label_dice_dict.pkl'))
unique_labels = list(set([int(x) for x in flatten([list(data[k].keys()) for k in data])]))
unique_labels.sort()

lut = {0: (1, 4),
       1: (2, 8),
       2: (3, 12),
       3: (6, 9),
       4: (7, 13),
       5: (11, 14)}



# exclude all labels that are not in all sets
all_labels = [[int(x) for x in data[k]] for k in data]
really_all = []
for label in set(flatten(all_labels)):
    broken = False
    for all_label in all_labels:
        if not label in all_label:
            broken = True
            break
    if broken:
        continue
    really_all.append(label)
really_all.sort()
unique_labels = really_all[1:]

num_plots = 36
siz = int(np.sqrt(num_plots))


plt.figure(figsize=(12, 7.5))
figure_ticks = ['SAMSEG', 'Med-DeepBrain', 'ASEG', 'FastSurfer']
for i, label in enumerate(unique_labels):
    anat_name = freeSurfer_lut(label, 'name').split('-')
    if anat_name[0] == 'Left':
        start_name = 1
        company_label = freeSurfer_lut('-'.join(['Right', *anat_name[1:]]), 'label')
    elif anat_name[0] == 'Right':
        continue
    else:
        company_label = None
        start_name = 0

    confusion = np.eye(4).flatten()
    for j, key in enumerate(data):
        if company_label:
            mean_dice = np.mean(data[key][label] + data[key][company_label])
        else:
            mean_dice = np.mean(data[key][label])

        a, b = lut[j]
        confusion[a], confusion[b] = mean_dice, mean_dice
    confusion = np.reshape(confusion, [4, 4])

    #figure_ticks = [list(data.keys())[0].split('-')[0]] + [x.split('-')[-1] for x in data.keys()][:3]
    ax = plt.subplot(3, siz, i + 1)
    anat_name_formatted = ' '.join(anat_name[start_name:])
    anat_name_formatted = anat_name_formatted.replace('White Matter', 'WM')
    plt.title(anat_name_formatted)
    u, v = np.remainder(i, siz), int(np.floor(i / siz))
    show_confusion(axis=ax,
                   array=confusion,
                   ticks=figure_ticks,
                   do_xtick=v == 0,
                   do_ytick=u == 0)
    if i + 1 == num_plots:
        break
plt.savefig('ConfusionMatrices.png', bbox_inches='tight')