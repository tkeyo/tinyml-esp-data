# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# ## Label distribution in collected data
# 
# I am using the dataset with 100Hz sampling rate. Label distribution is the same for the downsampled dataset.
# %% [markdown]
# Each row of the dataset represents one type of movement that was labelled. Where:
# - acc_x, acc_y, acc_z are accelerations
# - gyro_x, gyro_y, gyro_z are angular velocities
# - all rows have a label where:
#     - 0 is `no` movement
#     - 1 is `x` axis movement
#     - 2 is `y` axis movement
#     - 3 is `circle` movement
# - `shift` is the shift from the originally labelled start. `shift` values are represented by steps - i.e. shift of 2 is 2 x 10ms = 20ms as the original sampling period was 10ms
# - all signals have a numeral suffix denoting the point in time. As an example acc_x_500 is the value of acceleration in the direction of axis X, 500ms since start.

# %%
df = pd.read_csv('data/transformed/20210529_v2_data_all_100hz.csv').reset_index(drop=True)
df = df[df['shift'] == 0] # using the baseline dataset - as labeled
df = df.drop('shift', axis=1)
df.head(5)


# %%
g = sns.catplot(x='label', data=df, kind='count', height=5, aspect=0.8);
g.ax.set_title('\nLabel distribution in data\n')
g.ax.set_xlabel('\nLabels')
g.ax.set_ylabel('Count')
g.ax.set_xticklabels(['No move','X','Y','Circle'], fontsize=12);

# %% [markdown]
# ## Plotting relationships between signals
# 
# 

# %%
def filter_dataset_by_label(df: pd.DataFrame, label: int) -> pd.DataFrame:
    return df[df['label'] == label].drop('label', axis=1)

def get_acc_dfs(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_acc_x = pd.DataFrame(df.filter(regex='acc_x')).T.reset_index()
    df_acc_y = pd.DataFrame(df.filter(regex='acc_y')).T.reset_index()
    df_acc_z = pd.DataFrame(df.filter(regex='acc_z')).T.reset_index()

    return df_acc_x, df_acc_y, df_acc_z

def extract_time_ms(df: pd.DataFrame) -> pd.DataFrame:
    df['ms'] = df['index'].str.extract(r'(\d{1,4})')
    return df

def extract_signal(df: pd.DataFrame) -> pd.DataFrame:
    df['signal'] = df['index'].str.extract(r'(.*)_\d{1,4}')
    return df

def melt_df(df: pd.DataFrame) -> list:
    signals = ['acc_x', 'acc_y','acc_z','gyro_x','gyro_y','gyro_z']
    return [pd.melt(df[df['signal'] == s].drop(['signal','ms'], axis=1)).rename(columns={'value':s}).drop('variable',axis=1) for s in signals]


# %%
df_x = filter_dataset_by_label(df, 1).T.reset_index()
df_y = filter_dataset_by_label(df, 2).T.reset_index()
df_circle = filter_dataset_by_label(df, 3).T.reset_index()


# %%
df_x = extract_signal(df_x)
df_y = extract_signal(df_y)
df_circle = extract_signal(df_circle)


# %%
df_x = extract_time_ms(df_x)
df_y = extract_time_ms(df_y)
df_circle = extract_time_ms(df_circle)


# %%
df_x = df_x.drop('index', axis=1)
df_y = df_y.drop('index', axis=1)
df_circle = df_circle.drop('index', axis=1)


# %%
df_x_acc_x, df_x_acc_y, df_x_acc_z, df_x_gyro_x, df_x_gyro_y, df_x_gyro_z = melt_df(df_x)
df_y_acc_x, df_y_acc_y, df_y_acc_z, df_y_gyro_x, df_y_gyro_y, df_y_gyro_z = melt_df(df_y)
df_circle_acc_x, df_circle_acc_y, df_circle_acc_z, df_circle_gyro_x, df_circle_gyro_y, df_circle_gyro_z = melt_df(df_circle)


# %%
df_x_acc = pd.concat([df_x_acc_x, df_x_acc_y, df_x_acc_z], axis=1)
df_y_acc = pd.concat([df_y_acc_x, df_y_acc_y, df_y_acc_z], axis=1)
df_circle_acc = pd.concat([df_circle_acc_x, df_circle_acc_y, df_circle_acc_z], axis=1)

df_x_gyro = pd.concat([df_x_gyro_x, df_x_gyro_y, df_x_gyro_z], axis=1)
df_y_gyro = pd.concat([df_y_gyro_x, df_y_gyro_y, df_y_gyro_z], axis=1)
df_circle_gyro = pd.concat([df_circle_gyro_x, df_circle_gyro_y, df_circle_gyro_z], axis=1)


# %%
df_x_acc['label'] = 'x'
df_x_gyro['label'] = 'x'

df_y_acc['label'] = 'y'
df_y_gyro['label'] = 'y'

df_circle_acc['label'] = 'circle'
df_circle_gyro['label'] = 'circle'


# %%
df_acc = pd.concat([df_x_acc, df_y_acc, df_circle_acc], axis=0)
df_gyro = pd.concat([df_x_gyro, df_y_gyro, df_circle_gyro], axis=0)


# %%
g = sns.pairplot(
    df_acc,
    hue='label',
    height=5,
    markers='+'
    );


# %%
import plotly.express as px

fig = px.scatter_3d(
    df_acc,
    x='acc_x',
    y='acc_y',
    z='acc_z',
    color='label',
    width=800,
    height=800,
    color_discrete_sequence=['rgba(197,90,69)', 'rgba(79,134,174)', 'rgba(152,142,216)']
    )

fig.update_traces(marker=dict(size=2,symbol='cross'))

fig.show()


# %%
g = sns.pairplot(
    df_gyro,
    hue='label',
    height=5,
    markers='+'
    );


# %%
import plotly.express as px

fig = px.scatter_3d(
    df_gyro,
    x='gyro_x',
    y='gyro_y',
    z='gyro_z',
    color='label',
    width=800,
    height=800,
    color_discrete_sequence=['rgba(197,90,69)', 'rgba(79,134,174)', 'rgba(152,142,216)']
    )

fig.update_traces(marker=dict(size=2,symbol='cross'))

fig.show()


# %%
df_x_shapes = df_x.set_index('ms')
df_y_shapes = df_y.set_index('ms')
df_circle_shapes = df_circle.set_index('ms')


# %%
df_x_shapes['mean'] = df_x_shapes.drop(['signal'], axis=1).mean(axis=1)
df_y_shapes['mean'] = df_y_shapes.drop(['signal'], axis=1).mean(axis=1)
df_circle_shapes['mean'] = df_circle_shapes.drop(['signal'], axis=1).mean(axis=1)


# %%
def plot_signals(df_x, df_y, df_circle, signal, text, sensor):
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(30,8))

    ax[0].plot(df_x[df_x['signal'] == signal].drop(['mean','signal'], axis=1), alpha=0.3);
    ax[1].plot(df_y[df_y['signal'] == signal].drop(['mean','signal'], axis=1), alpha=0.3);
    ax[2].plot(df_circle[df_circle['signal'] == signal].drop(['mean','signal'], axis=1), alpha=0.3);


    ax[0].plot(df_x[df_x['signal'] == signal]['mean'], alpha=1, color='red', linewidth=3);
    ax[1].plot(df_y[df_y['signal'] == signal]['mean'], alpha=1, color='red', linewidth=3);
    ax[2].plot(df_circle[df_circle['signal'] == signal]['mean'], alpha=1, color='red', linewidth=3);

    ax[0].set_title(f'{text} {sensor} of all `X` movements\n+ their mean\n')
    ax[1].set_title(f'{text} {sensor} of all `Y` movements\n+ their mean\n')
    ax[2].set_title(f'{text} {sensor} of all `circle` movements\n+ their mean\n')
    
    ax[0].set_ylabel('Acceleration [m/s^2]')
    for x in range(3):
        ax[x].set_xlabel('Time [ms]')

        temp = ax[x].xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::5]))
        for label in temp:
            label.set_visible(False)

    fig, ax = plt.subplots(ncols=3, figsize=(30,6))
    sns.heatmap(df_x[df_x['signal'] == signal].drop(['signal'], axis=1).corr(), ax=ax[0])
    sns.heatmap(df_y[df_y['signal'] == signal].drop(['signal'], axis=1).corr(), ax=ax[1])
    sns.heatmap(df_circle[df_circle['signal'] == signal].drop(['signal'], axis=1).corr(), ax=ax[2])
    
    ax[0].set_title(f'Correlation matrix\n{text} {sensor} of `X` movements\n');
    ax[1].set_title(f'Correlation matrix\n{text} {sensor} of `Y` movements\n');
    ax[2].set_title(f'Correlation matrix\n{text} {sensor} of `circle` movements\n');


# %%
plot_signals(df_x_shapes, df_y_shapes, df_circle_shapes, 'acc_x', 'X axis', 'acceleration')


# %%
plot_signals(df_x_shapes, df_y_shapes, df_circle_shapes, 'acc_y', 'Y axis', 'acceleration')


# %%
plot_signals(df_x_shapes, df_y_shapes, df_circle_shapes, 'acc_z', 'Z axis', 'acceleration')


# %%
plot_signals(df_x_shapes, df_y_shapes, df_circle_shapes, 'gyro_x', 'X axis', 'angular velocity')


# %%
plot_signals(df_x_shapes, df_y_shapes, df_circle_shapes, 'gyro_y', 'Y axis', 'angular velocity')


# %%
plot_signals(df_x_shapes, df_y_shapes, df_circle_shapes, 'gyro_z', 'Z axis', 'angular velocity')


