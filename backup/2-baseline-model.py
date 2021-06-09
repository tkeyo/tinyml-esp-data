# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
def save_model(classifier, directory, model_type, hz):
    '''Saves model to defined folder.'''

    import os
    import m2cgen as m2c
    
    BASE_PATH = f'models/{directory}/{model_type}'
    FILE_NAME = f'{model_type}_{hz}hz.py'

    if not os.path.isdir('models'):
        os.mkdir(BASE_PATH)

    code = m2c.export_to_python(classifier)
    
    with open(os.join(BASE_PATH + FILE_NAME), 'w') as f:
        f.writelines(code)


# %%
def transform_data_for_inference(df):
    '''
        Transoforms dataset for inference.
        ms,acc,gyro -> acc_x_0, gyro_x_0, acc_x_10, gyro_x_10, .... acc_x_n, gyro_x_n
    '''

    df_list=[]

    for time in df.index:
        _df = pd.DataFrame(df.loc[time]).T
        df_list.append(_df.add_suffix(f'_{str(int(time))}').reset_index(drop=True))

    return pd.concat(df_list, axis=1)


def get_filter_string(start, step):
    '''
        Creates a string to filter dataset for defined timetimestamps.
        To be used with df.filter(regex='<string returned from this functions>')
        Example: 0|50|100
    '''

    keep = np.arange(start, start+1+1000, step=int(step))
    return '|'.join(map(str, keep.astype(int)))


def line_color(inf_result):
    '''Returns color associated with inference result.'''
    colors = {
        1:'blue',
        2:'red',
        3:'green'
    }
    return colors[inf_result]

def downsample_dataset(df, freq):

    def get_period(frequency):
        return int(1000 / frequency)

    period = get_period(freq)

    last_index_ms = df.index[-1]
    keep = np.arange(last_index_ms, step=period)

    return df.loc[keep]


def run_inference(df, model, start, step):
    '''Runs inference.'''
    regex_filter = get_filter_string(start=start, step=step)
    data = list(df.filter(regex=f'_({regex_filter})$').loc[0])
    # print(len(data))
    return model.score(data)

def calculate_error(res, move_type):
    '''
        Calculates inference error rate in validation data.
    '''

    error_setup = {
        'circle': {'err_1':1,'err_2':2},
        'x':{'err_1':2, 'err_2':3},
        'y':{'err_1':1,'err_2':3}
    }

    err_1 = error_setup[move_type]['err_1']
    err_2 = error_setup[move_type]['err_2']

    val_counts = res['result'].value_counts().drop(0) # dropping `no movement`
    val_counts_keys = val_counts.keys()

    total_wrong = 0

    if err_1 in val_counts_keys:
        total_wrong += val_counts[err_1]
    if err_2 in val_counts_keys:
        total_wrong += val_counts[err_2]
    
    return (total_wrong / val_counts.sum()) * 100


# def calculate_misclassified_percentage_x(r


# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def train_decision_tree(X_train, X_test, y_train, y_test, hz, direc):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    save_model(classifier=clf, direc=direc, model_type='decision_tree', hz=hz)

    return accuracy, f1, precision, recall, clf


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def train_random_forest(X_train, X_test, y_train, y_test, hz, direc):
    clf = RandomForestClassifier(random_state=42, n_estimators=4)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    save_model(classifier=clf, direc=direc, model_type='random_forest', hz=hz)

    return accuracy, f1, precision, recall, clf


# %%
from sklearn.svm import LinearSVC
from sklearn import metrics

def train_svc(X_train, X_test, y_train, y_test, hz, direc):
    clf = LinearSVC(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    save_model(classifier=clf, direc=direc, model_type='svc', hz=hz)

    return accuracy, f1, precision, recall, clf


# %%
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def train_logistic_regression(X_train, X_test, y_train, y_test, hz, direc):
    clf = LogisticRegression(random_state=42,  max_iter=10_000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    save_model(classifier=clf, direc=direc, model_type='logistic_regression', hz=hz)

    return accuracy, f1, precision, recall, clf


# %%
collect_metrics = {}
collect_metrics['decision_tree'] = {}
collect_metrics['random_forest'] = {}
collect_metrics['svc'] = {}
collect_metrics['logistic_regression'] = {}

MODELS = 'no_shift'

for hz in [10,20,25,50,100]:
# for hz in [100]:
    print(f'HZ: {hz}')
    df = pd.read_csv(f'../../data/7-data_combined_all/20210522_data_all_{hz}hz.csv').reset_index(drop=True)
    df_train = df[(df['shift'] == 0)]
    df_train = df_train.dropna(axis=0)

    # print(list(df_train.columns))

    print('DF Shape', df_train.shape)
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['label','shift'],axis=1), df_train['label'], test_size=0.3, random_state=42)

    print('\n')
    print(f'Decision Tree {hz}hz')
    accuracy, f1, precision, recall, dt = train_decision_tree(X_train, X_test, y_train, y_test, hz=hz, direc=MODELS)

    test_data = [X_train.iloc[0]]
    dt_time = get_ipython().run_line_magic('timeit', '-o dt.predict(test_data)')

    collect_metrics['decision_tree'][hz] = {
        'accuracy':accuracy,
        'f1':f1,
        'precision':precision,
        'recall':recall,
        'time':dt_time.timings
    }
    print('\n')

    print(f'Random Forest {hz}hz')
    accuracy, f1, precision, recall, rfc = train_random_forest(X_train, X_test, y_train, y_test, hz=hz, direc=MODELS)

    test_data = [X_train.iloc[0]]
    rfc_time = get_ipython().run_line_magic('timeit', '-o rfc.predict(test_data)')

    collect_metrics['random_forest'][hz] = {
        'accuracy':accuracy,
        'f1':f1,
        'precision':precision,
        'recall':recall,
        'time': rfc_time.timings
    }
    print('\n')

    print(f'SVC {hz}hz')
    accuracy, f1, precision, recall, svc = train_svc(X_train, X_test, y_train, y_test, hz=hz, direc=MODELS)

    test_data = [X_train.iloc[0]]
    svc_time = get_ipython().run_line_magic('timeit', '-o svc.predict(test_data)')

    collect_metrics['svc'][hz] = {
        'accuracy':accuracy,
        'f1':f1,
        'precision':precision,
        'recall':recall,
        'time':svc_time.timings
    }
    print('\n')
    
    print(f'Logistic regression {hz}hz')
    accuracy, f1, precision, recall, lr = train_logistic_regression(X_train, X_test, y_train, y_test, hz=hz, direc=MODELS)

    test_data = [X_train.iloc[0]]
    lr_time = get_ipython().run_line_magic('timeit', '-o lr.predict(test_data)')

    collect_metrics['logistic_regression'][hz] = {
        'accuracy':accuracy,
        'f1':f1,
        'precision':precision,
        'recall':recall,
        'time':lr_time.timings
    }
    print('-' * 50)


# %%



