# importing libraries
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
pd.options.display.max_columns = 4
from utils import pickle_in, get_root

# Load data
X = pickle_in(get_root('data', 'radiomics_dataframes', 'advanced_radiomic_dataframe_DTI_freesurfer-smol.pkl'))
Y = X.is_deprived
X = X.drop('is_deprived', axis=1)


# Scale features to mean and unit variance
sc = StandardScaler()
sc.fit(X)
X = pd.DataFrame(sc.fit_transform(X))

# Partitioning
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
d_train = lgb.Dataset(X_train, label=y_train)

predict.py
# Set LGB-parameters
params = {'learning_rate': 0.08,
          'boosting_type': 'goss',  # GB decision tree
          'objective': 'binary',
          'metric': 'binary_logloss',
          'max_depth': 10}

# train the model on 100 epocs
clf = lgb.train(params, d_train, 10)

#prediction on the test set
y_pred = clf.predict(X_test)

y_pred = y_pred.round(0)  # converting from float to integer
y_pred = y_pred.astype(int)  # roc_auc_score metric
auroc = roc_auc_score(y_pred, y_test)  # 0.9672167056074766
print(auroc)

