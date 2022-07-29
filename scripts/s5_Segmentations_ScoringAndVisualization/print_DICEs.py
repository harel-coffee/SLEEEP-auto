import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from scipy.stats import normaltest
from utils import get_root, pickle_in
from matplotlib import pyplot as plt
import os
from utils import flatten, freeSurfer_lut
import numpy as np

data = pickle_in(get_root('scripts', '4dice_scoring', 'label_dice_dict.pkl'))

print('labl'.ljust(4),
      'comparisson'.ljust(25),
      'dice'.ljust(6),
      'std'.ljust(6),
      'pval'.ljust(6),
      'n_samples')

unique_labels = list(set([int(x) for x in flatten([list(data[k].keys()) for k in data])]))
unique_labels.sort()


num_plots = 16

for i, label in enumerate(unique_labels):
    if len(data.keys()) == 6:
        confusion = np.eye(4).flatten()
    elif len(data.keys()) == 3:
        confusion = np.eye(3).flatten()

    for j, key in enumerate(data):
        # perform normality test
        if label in data[key]:
            cur_data = data[key][label]
            try:
                p_val = normaltest(cur_data).pvalue
                if p_val < 0.05:
                    p_val = '< 0.05'
                else:
                    p_val = np.round(p_val, 3)
            except:
                p_val = ''

            # print dice score
            mean_dice = np.mean(cur_data)
            print(str(label).ljust(4),
                  freeSurfer_lut(label, 'name').ljust(40),
                  key.ljust(25),
                  str(np.round(mean_dice, 3)).ljust(6),
                  str(np.round(np.std(cur_data), 3)).ljust(6),
                  str(p_val).ljust(6),
                  len(cur_data))

#             if i >= num_plots:
#                 continue
#             elif j == 0:
#                 confusion[1] = mean_dice
#                 confusion[4] = mean_dice
#             elif i == 1:
#                 confusion[2] = mean_dice
#                 confusion[8] = mean_dice
#             elif i == 2:
#                 confusion[3] = mean_dice
#                 confusion[12] = mean_dice
#             elif i == 3:
#                 confusion[6] = mean_dice
#                 confusion[9] = mean_dice
#             elif i == 4:
#                 confusion[7] = mean_dice
#                 confusion[13] = mean_dice
#             elif i == 5:
#                 confusion[11] = mean_dice
#                 confusion[14] = mean_dice
#             else:
#                 print('oops!')


#         confusion = np.reshape(confusion, [4, 4])
#         ax = plt.subplot(int(np.sqrt(16)), int(np.sqrt(16)), i + 1)
#         figure_ticks = [list(data.keys())[0].split('-')[0]] + [x.split('-')[-1] for x in data.keys()][:3]
#         plt.imshow(confusion)
#         for x in range(confusion.shape[0]):
#             for y in range(confusion.shape[1]):
#                 val = np.round(confusion[x, y], 2)
#                 if val == 1:
#                     txt = '1.'
#                 else:
#                     txt = str(val)[1:]
#                 plt.text(x, y, txt, {'c': 'k',
#                                      'fontsize': 'large',
#                                      'ha': 'center',
#                                      'va': 'center'})
#         plt.title(label)
#         plt.clim([0.6, 1])
#         plt.xticks(np.arange(len(figure_ticks)), figure_ticks)
#         plt.yticks(np.arange(len(figure_ticks)), figure_ticks)
#         ax.xaxis.tick_top()
#         plt.set_cmap('RdYlGn')


# plt.show()

