# %%
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import os
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sn
import csv
import pickle

from tqdm import tqdm
from matplotlib import pyplot as plt


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

warnings.filterwarnings('ignore')


# %%
# from google.colab import drive
# drive.mount('/content/gdrive')
# os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"
# %cd "/content/gdrive/MyDrive/Kaggle"

# %%
# base_path =  '/content/gdrive/MyDrive/Kaggle/'
base_path = '.'
# data_path =  'BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
data_path = 'BreaKHis_v1/histology_slides/breast'
magnifications = ['40X', '100X', '200X', '400X']
classes = ['benign', 'malignant']
sub_classes = {
    'benign': ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma'],
    'malignant': ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
}
n_folds = (1, 2, 3, 4, 5)


# %%
encoder = LabelEncoder()
encoder.fit(classes)

# %%
load_ds_from_pickle = True
# load_ds_from_pickle = False

# %%
dataset = pd.DataFrame()

if load_ds_from_pickle:
    dataset = pd.read_pickle('dataset.pkl')
else:
    for clazz in classes:
        for sub_clazz in sub_classes[clazz]:
            path = os.path.join(base_path, data_path, clazz, "SOB", sub_clazz)
            for id in os.listdir(path):
                for magnification in magnifications:
                    path_to_files = os.path.join(path, id, magnification)
                    for file_name in os.listdir(path_to_files):
                        dataset = dataset.append({
                            'id': id,
                            'file_name': file_name,
                            'path': os.path.join(path_to_files, file_name),
                            'magnification': magnification,
                            'type': sub_clazz,
                            'lesion': clazz
                        }, ignore_index=True)

# 5 fold K
    n_folds = (1, 2, 3, 4, 5)
    folds_df = pd.DataFrame()
    for nfold in n_folds:
        fold_file = f"dsfold{nfold}.txt"

        fd = pd.read_csv(fold_file, delimiter="|", names=[
                         "file_name", "magnification", "fold", "grp"])
        fd = fd[["file_name", "grp"]]
        fd.rename(columns={"grp": f"fold_{nfold}"}, inplace=True)
        if folds_df.empty:
            folds_df = folds_df.append(fd)
        else:
            folds_df = folds_df.merge(fd, how="inner", on="file_name")

    folds_df.head()

    dataset = dataset.merge(folds_df, how="inner", on="file_name")

    dataset.to_pickle('dataset.pkl')

dataset.head()


# %% [markdown]
# ### Classifier Functions

# %%
def knn_clf(x_train, y_train, x_test, y_test):
    k = 1
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(x_train, y_train)

    return classification_report(y_test, knn_clf.predict(x_test), output_dict=True)


def svm_clf(x_train, y_train, x_test, y_test):
    svm_clf = SVC()
    svm_clf = svm_clf.fit(x_train, y_train)

    return classification_report(y_test, svm_clf.predict(x_test), output_dict=True)


def dt_clf(x_train, y_train, x_test, y_test):
    dt_clf = DecisionTreeClassifier()
    dt_clf = dt_clf.fit(x_train, y_train)

    return classification_report(y_test, dt_clf.predict(x_test), output_dict=True)


# %%
image_size = (46, 70, 3)

# %%
results = {
    "knn": {
    },
    "svm": {
    },
    "dt": {
    },
}

# %%


def run_clfs(clf, mag, n_fold):
    print("Classifier:", clf)
    print("Magnification:", mag)
    print("Fold:", n_fold)

    df = dataset.copy()[dataset["magnification"] == mag]
    train_df = df.copy()[dataset[n_fold] == "train"]
    test_df = df.copy().drop(train_df.index).reset_index(drop=True)

    x_train, x_test, y_train, y_test = [], [], [], []

    for i, row in tqdm(train_df.iterrows()):
        image = load_img(row["path"], target_size=image_size)
        x_train.append(img_to_array(image) / 255.0)
        y_train.append(row["lesion"])

    for i, row in tqdm(test_df.iterrows()):
        image = load_img(row["path"], target_size=image_size)
        x_test.append(img_to_array(image) / 255.0)
        y_test.append(row["lesion"])

    x_train_ = np.asarray(x_train)
    x_test_ = np.asarray(x_test)

    x_train_ = np.reshape(x_train_, (len(x_train_), np.prod(image_size)))
    x_test_ = np.reshape(x_test_, (len(x_test_), np.prod(image_size)))

    y_train_ = encoder.transform(y_train)
    y_test_ = encoder.transform(y_test)

    results[clf][mag] = {}
    if clf == "knn":
        # 1NN
        results[clf][mag][n_fold] = knn_clf(
            x_train_, y_train_, x_test_, y_test_)
    elif clf == "svm":
        # SVM
        results[clf][mag][n_fold] = svm_clf(
            x_train_, y_train_, x_test_, y_test_)
    elif clf == "dt":
        # Decision Tree
        results[clf][mag][n_fold] = dt_clf(
            x_train_, y_train_, x_test_, y_test_)

    print("Processed:", clf, mag, n_fold)

    with open(f"results/{clf}_{mag}_{n_fold}.pkl", 'wb') as f:
        pickle.dump(results, f)
