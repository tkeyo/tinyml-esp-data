# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np

from validation_utils import transform_data_for_inference, line_color, downsample_df, run_inference, calculate_error, run_validation

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from models.baseline.base.decision_tree import decision_tree_100hz, decision_tree_50hz, decision_tree_25hz, decision_tree_20hz, decision_tree_10hz
from models.baseline.base.random_forest import random_forest_100hz, random_forest_50hz, random_forest_25hz, random_forest_20hz, random_forest_10hz

models = [
    (decision_tree_100hz, 100, (0,0), 'decision_tree'),
    (decision_tree_50hz, 50, (0,1), 'decision_tree'),
    (decision_tree_25hz, 25, (0,2), 'decision_tree'),
    (decision_tree_20hz, 20, (0,3), 'decision_tree'),
    (decision_tree_10hz, 10, (0,4), 'decision_tree'),
    (random_forest_100hz, 100, (1,0), 'random_forest'),
    (random_forest_50hz, 50, (1,1), 'random_forest'),
    (random_forest_25hz, 25, (1,2), 'random_forest'),
    (random_forest_20hz, 20, (1,3), 'random_forest'),
    (random_forest_10hz, 10, (1,4), 'random_forest')
]


# %%
results_base_circle = run_validation(models, 'data/validation/move_circle_20210522_1.csv', 'base', is_plot=False, is_save_results=True)
results_base_x = run_validation(models, 'data/validation/move_x_20210522_1.csv', 'base', is_plot=False, is_save_results=True)
results_base_y = run_validation(models, 'data/validation/move_y_20210522_1.csv', 'base', is_plot=False, is_save_results=True)


# %%
from models.baseline.centered.decision_tree import decision_tree_100hz, decision_tree_50hz, decision_tree_25hz, decision_tree_20hz, decision_tree_10hz
from models.baseline.centered.random_forest import random_forest_100hz, random_forest_50hz, random_forest_25hz, random_forest_20hz, random_forest_10hz

models = [
    (decision_tree_100hz, 100, (0,0), 'decision_tree'),
    (decision_tree_50hz, 50, (0,1), 'decision_tree'),
    (decision_tree_25hz, 25, (0,2), 'decision_tree'),
    (decision_tree_20hz, 20, (0,3), 'decision_tree'),
    (decision_tree_10hz, 10, (0,4), 'decision_tree'),
    (random_forest_100hz, 100, (1,0), 'random_forest'),
    (random_forest_50hz, 50, (1,1), 'random_forest'),
    (random_forest_25hz, 25, (1,2), 'random_forest'),
    (random_forest_20hz, 20, (1,3), 'random_forest'),
    (random_forest_10hz, 10, (1,4), 'random_forest')
]


# %%
results_centered_circle = run_validation(models, 'data/validation/move_circle_20210522_1.csv', 'centered', is_plot=False, is_save_results=True)
results_centered_x = run_validation(models, 'data/validation/move_x_20210522_1.csv', 'centered', is_plot=False, is_save_results=True)
results_centered_y = run_validation(models, 'data/validation/move_y_20210522_1.csv', 'centered', is_plot=False, is_save_results=True)


# %%
from models.baseline.centered_aug.decision_tree import decision_tree_100hz, decision_tree_50hz, decision_tree_25hz, decision_tree_20hz, decision_tree_10hz
from models.baseline.centered_aug.random_forest import random_forest_100hz, random_forest_50hz, random_forest_25hz, random_forest_20hz, random_forest_10hz

models = [
    (decision_tree_100hz, 100, (0,0), 'decision_tree'),
    (decision_tree_50hz, 50, (0,1), 'decision_tree'),
    (decision_tree_25hz, 25, (0,2), 'decision_tree'),
    (decision_tree_20hz, 20, (0,3), 'decision_tree'),
    (decision_tree_10hz, 10, (0,4), 'decision_tree'),
    (random_forest_100hz, 100, (1,0), 'random_forest'),
    (random_forest_50hz, 50, (1,1), 'random_forest'),
    (random_forest_25hz, 25, (1,2), 'random_forest'),
    (random_forest_20hz, 20, (1,3), 'random_forest'),
    (random_forest_10hz, 10, (1,4), 'random_forest')
    ]


# %%
results_centered_aug_circle = run_validation(models, 'data/validation/move_circle_20210522_1.csv', 'centered_aug', is_plot=False, is_save_results=True)
results_centered_aug_x = run_validation(models, 'data/validation/move_x_20210522_1.csv', 'centered_aug', is_plot=False, is_save_results=True)
results_centered_aug_y = run_validation(models, 'data/validation/move_y_20210522_1.csv', 'centered_aug', is_plot=False, is_save_results=True)


# %%
from models.baseline.centered_smote.decision_tree import decision_tree_100hz, decision_tree_50hz, decision_tree_25hz, decision_tree_20hz, decision_tree_10hz
from models.baseline.centered_smote.random_forest import random_forest_100hz, random_forest_50hz, random_forest_25hz, random_forest_20hz, random_forest_10hz

models = [
    (decision_tree_100hz, 100, (0,0), 'decision_tree'),
    (decision_tree_50hz, 50, (0,1), 'decision_tree'),
    (decision_tree_25hz, 25, (0,2), 'decision_tree'),
    (decision_tree_20hz, 20, (0,3), 'decision_tree'),
    (decision_tree_10hz, 10, (0,4), 'decision_tree'),
    (random_forest_100hz, 100, (1,0), 'random_forest'),
    (random_forest_50hz, 50, (1,1), 'random_forest'),
    (random_forest_25hz, 25, (1,2), 'random_forest'),
    (random_forest_20hz, 20, (1,3), 'random_forest'),
    (random_forest_10hz, 10, (1,4), 'random_forest')
    ]


# %%
results_centered_smote_circle = run_validation(models, 'data/validation/move_circle_20210522_1.csv',  'centered_smote', is_plot=False, is_save_results=True)
results_centered_smote_x = run_validation(models, 'data/validation/move_x_20210522_1.csv', 'centered_smote', is_plot=False, is_save_results=True)
results_centered_smote_y = run_validation(models, 'data/validation/move_y_20210522_1.csv', 'centered_smote', is_plot=False, is_save_results=True)


# %%
from models.baseline.end.decision_tree import decision_tree_100hz, decision_tree_50hz, decision_tree_25hz, decision_tree_20hz, decision_tree_10hz
from models.baseline.end.random_forest import random_forest_100hz, random_forest_50hz, random_forest_25hz, random_forest_20hz, random_forest_10hz

models = [
    (decision_tree_100hz, 100, (0,0), 'decision_tree'),
    (decision_tree_50hz, 50, (0,1), 'decision_tree'),
    (decision_tree_25hz, 25, (0,2), 'decision_tree'),
    (decision_tree_20hz, 20, (0,3), 'decision_tree'),
    (decision_tree_10hz, 10, (0,4), 'decision_tree'),
    (random_forest_100hz, 100, (1,0), 'random_forest'),
    (random_forest_50hz, 50, (1,1), 'random_forest'),
    (random_forest_25hz, 25, (1,2), 'random_forest'),
    (random_forest_20hz, 20, (1,3), 'random_forest'),
    (random_forest_10hz, 10, (1,4), 'random_forest')
]


# %%
results_end_circle = run_validation(models, 'data/validation/move_circle_20210522_1.csv', 'end', is_plot=False, is_save_results=True)
results_end_x = run_validation(models, 'data/validation/move_x_20210522_1.csv', 'end', is_plot=False, is_save_results=True)
results_end_y = run_validation(models, 'data/validation/move_y_20210522_1.csv', 'end', is_plot=False, is_save_results=True)


# %%
import matplotlib.ticker as tick

fig, ax = plt.subplots(ncols=5, nrows=3, sharey=True, figsize=(30,15))

sns.pointplot(data=pd.DataFrame(results_base_circle), x='hz', y='error_percentage', hue='model', ax=ax[0][0])
sns.pointplot(data=pd.DataFrame(results_base_x), x='hz', y='error_percentage', hue='model', ax=ax[1][0])
sns.pointplot(data=pd.DataFrame(results_base_y), x='hz', y='error_percentage', hue='model', ax=ax[2][0])

sns.pointplot(data=pd.DataFrame(results_centered_circle), x='hz', y='error_percentage', hue='model', ax=ax[0][1])
sns.pointplot(data=pd.DataFrame(results_centered_x), x='hz', y='error_percentage', hue='model', ax=ax[1][1])
sns.pointplot(data=pd.DataFrame(results_centered_y), x='hz', y='error_percentage', hue='model', ax=ax[2][1])

sns.pointplot(data=pd.DataFrame(results_centered_aug_circle), x='hz', y='error_percentage', hue='model', ax=ax[0][2])
sns.pointplot(data=pd.DataFrame(results_centered_aug_x), x='hz', y='error_percentage', hue='model', ax=ax[1][2])
sns.pointplot(data=pd.DataFrame(results_centered_aug_y), x='hz', y='error_percentage', hue='model', ax=ax[2][2])

sns.pointplot(data=pd.DataFrame(results_centered_smote_circle), x='hz', y='error_percentage', hue='model', ax=ax[0][3])
sns.pointplot(data=pd.DataFrame(results_centered_smote_x), x='hz', y='error_percentage', hue='model', ax=ax[1][3])
sns.pointplot(data=pd.DataFrame(results_centered_smote_y), x='hz', y='error_percentage', hue='model', ax=ax[2][3])

sns.pointplot(data=pd.DataFrame(results_end_circle), x='hz', y='error_percentage', hue='model', ax=ax[0][4])
sns.pointplot(data=pd.DataFrame(results_end_x), x='hz', y='error_percentage', hue='model', ax=ax[1][4])
sns.pointplot(data=pd.DataFrame(results_end_y), x='hz', y='error_percentage', hue='model', ax=ax[2][4])

ax[0][0].title.set_text('\nBaseline data\n\n')
ax[0][1].title.set_text('\nCentered `x` and `y` move signals\n\n')
ax[0][2].title.set_text('\nCentered `x` and `y` move signals\n+ Augmentation\n')
ax[0][3].title.set_text('\nCentered `x` and `y` move signal\n+ SMOTE\n')
ax[0][4].title.set_text('\n`x` and `y` signals at the end\n\n')

ax[0][0].set_ylabel('Circle')
ax[1][0].set_ylabel('X')
ax[2][0].set_ylabel('Y')

ax[0][1].set_ylabel('')
ax[1][1].set_ylabel('')
ax[2][1].set_ylabel('')

ax[0][2].set_ylabel('')
ax[1][2].set_ylabel('')
ax[2][2].set_ylabel('')

ax[0][3].set_ylabel('')
ax[1][3].set_ylabel('')
ax[2][3].set_ylabel('')

ax[0][4].set_ylabel('')
ax[1][4].set_ylabel('')
ax[2][4].set_ylabel('')

ax[0][0].set_xlabel('')
ax[0][1].set_xlabel('')
ax[0][2].set_xlabel('')
ax[0][3].set_xlabel('')
ax[0][4].set_xlabel('')

ax[1][0].set_xlabel('')
ax[1][1].set_xlabel('')
ax[1][2].set_xlabel('')
ax[1][3].set_xlabel('')
ax[1][4].set_xlabel('')


ax[2][0].set_xlabel('\nSampling frequency [Hz]')
ax[2][1].set_xlabel('\nSampling frequency [Hz]')
ax[2][2].set_xlabel('\nSampling frequency [Hz]')
ax[2][3].set_xlabel('\nSampling frequency [Hz]')
ax[2][4].set_xlabel('\nSampling frequency [Hz]')

plt.gca().yaxis.set_major_formatter(tick.FuncFormatter(lambda x, post: f'{int(x)} %'))
fig.tight_layout();


# %%



