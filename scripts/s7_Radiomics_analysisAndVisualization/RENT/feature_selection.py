'''
    Modeled after notebook:
    https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Classification_example.ipynb
'''
from sklearn.model_selection import train_test_split
from utils import pickle_in, get_root, pickle_out
from matplotlib import pyplot as plt
import pandas as pd
import sys
import warnings
import numpy as np

# Local imports
sys.path.append(get_root('scripts', '6selection', 'RENT', 'src'))
from RENT import RENT, stability

# Set preferences
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 2000)


# Define data and partition
X = pickle_in(get_root('data', 'radiomics_dataframes', 'advanced_radiomic_dataframe_DTI_freesurfer-smol.pkl'))
Y = X.is_deprived
X.drop('is_deprived', axis='columns', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Define regularization parameters and L1-rations for elastic net.
my_C_params = np.concatenate([np.linspace(0.01, 1, 20), np.linspace(1, 3, 20)])
my_l1_ratios = np.concatenate([np.linspace(0, 1, 60)])

my_C_params = [0.2705263157894737]
my_l1_ratios = [0.847457627118644]

model = RENT.RENT_Classification(data=X_train,
                                 target=y_train,
                                 feat_names=X_train.columns,
                                 C=my_C_params,
                                 l1_ratios=my_l1_ratios,
                                 autoEnetParSel=True,
                                 poly='OFF',
                                 testsize_range=(0.25, 0.25),
                                 scoring='mcc',
                                 classifier='logreg',
                                 K=100,
                                 random_state=0,
                                 verbose=1)

# get cross-validation matrices
cv_score, cv_zeros, cv_harmonic_mean = model.get_cv_matrices()
pickle_out(model.get_cv_matrices(), 'many_cv_matrices.pkl')
pickle_out(cv_score.values, 'cv_score_values.pkl')
print(cv_score)

bar = np.zeros(cv_score.shape)
for i, x in enumerate(cv_score.values):
    for j, y in enumerate(x):
        bar[i, j] = np.nan_to_num(y)
plt.imshow(bar, clim=[0.3, 0.5])
plt.yticks(np.linspace(0, len(my_l1_ratios)-1, len(np.linspace(0, 1, 11))), [np.round(x, 2) for x in np.linspace(0, 1, 11)])
plt.ylabel('l1 ratios)')
plt.xlabel('C params')
plt.colorbar()
plt.show()

# train the model
model.train()

# Set selection criteria cutoffs
# selected_features = model.select_features(tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975)
selected_features = model.select_features(tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975)

model.get_summary_criteria()

X_train.columns[selected_features]

####

#predict test data
test_labels = y_test
test_data = X_test
train_data = X_train
train_labels = y_train

# Import what is needed for prediction and evaluation of predictions from test set
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR

# Scale the data accordingly
sc = StandardScaler()
train_data_sc = sc.fit_transform(train_data.iloc[:, selected_features])
test_data_sc = sc.transform(test_data.iloc[:, selected_features])

# Train model with
prediction_model = LR(penalty='none', max_iter=8000, solver="saga", random_state=0).\
        fit(train_data_sc, train_labels)

# Print results
print("f1 1: ", f1_score(test_labels, prediction_model.predict(test_data_sc)))
print("f1 0: ", f1_score(1 - test_labels, 1 - prediction_model.predict(test_data_sc)))
print("Accuracy: ", accuracy_score(test_labels, prediction_model.predict(test_data_sc)))
print("Matthews correlation coefficient: ", matthews_corrcoef(test_labels, prediction_model.predict(test_data_sc)))

model.plot_validation_study(test_data, test_labels, num_drawings=100, num_permutations=100, alpha=0.05)
