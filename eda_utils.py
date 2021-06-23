import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def filter_dataset_by_label(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """
    Filters a dataframe for a given label.

    Args:
        df: Input dataframe.
        label: Label in dataframe to be filtered.
    Returns:
        Returns a dataframe with only one label category.
    """    
    return df[df['label'] == label].drop('label', axis=1)


def get_acc_dfs(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Outputs 3 dataframes for each of the acceleration signals.

    Args:
        df: Signals dataframe.
    Returns:
        Returns dataframes for each of the 3 acceleration signals.
    """    
    df_acc_x = pd.DataFrame(df.filter(regex='acc_x')).T.reset_index()
    df_acc_y = pd.DataFrame(df.filter(regex='acc_y')).T.reset_index()
    df_acc_z = pd.DataFrame(df.filter(regex='acc_z')).T.reset_index()

    return df_acc_x, df_acc_y, df_acc_z


def extract_time_ms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts milisecond values from the index values of a dataframe.

    Args:
        df: Signals dataframe.

    Returns:
        Returns a dataframe that includes a `ms` time column.
    """    
    df['ms'] = df['index'].str.extract(r'(\d{1,4})')
    return df


def extract_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts a signal (acceleration, gyroscope) from the index values of
    a dataframe.

    Args:
        df: Signals dataframe.

    Returns:
        Returns a dataframe that includes a `signal` column.
    """    
    df['signal'] = df['index'].str.extract(r'(.*)_\d{1,4}')
    return df


def melt_df(df: pd.DataFrame) -> list:
    """
    Melts a dataframe for signal plotting.

    Args:
        df: Signals dataframe.

    Returns:
        Returns a list of signals.
    """    
    signals = ['acc_x', 'acc_y','acc_z','gyro_x','gyro_y','gyro_z']
    return [pd.melt(df[df['signal'] == s].drop(['signal','ms'], axis=1)).rename(columns={'value':s}).drop('variable',axis=1) for s in signals]


def plot_signals(
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
        df_circle: pd.DataFrame,
        signal: str,
        title_text: str,
        quantity: str):
    """
    Plots all signals, their mean and associated correlation matrix.

    Args:
        df_x: Dataframe containing x movements.
        df_y: Dataframe containing y movements.
        df_circle: Dataframe containing circle movements.
        signal: Signal - acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        title_text: Text to be used for titles.
        quantity: Plotted quantity text - acceleration/angular velocity
    """    
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(30,8))

    # plotting all signals
    ax[0].plot(df_x[df_x['signal'] == signal].drop(['mean','signal'], axis=1), alpha=0.3);
    ax[1].plot(df_y[df_y['signal'] == signal].drop(['mean','signal'], axis=1), alpha=0.3);
    ax[2].plot(df_circle[df_circle['signal'] == signal].drop(['mean','signal'], axis=1), alpha=0.3);

    # plotting mean of all signals
    ax[0].plot(df_x[df_x['signal'] == signal]['mean'], alpha=1, color='red', linewidth=3);
    ax[1].plot(df_y[df_y['signal'] == signal]['mean'], alpha=1, color='red', linewidth=3);
    ax[2].plot(df_circle[df_circle['signal'] == signal]['mean'], alpha=1, color='red', linewidth=3);

    # set titles
    ax[0].set_title(f'{title_text} {quantity} of all `X` movements\n+ their mean\n')
    ax[1].set_title(f'{title_text} {quantity} of all `Y` movements\n+ their mean\n')
    ax[2].set_title(f'{title_text} {quantity} of all `circle` movements\n+ their mean\n')
    
    ax[0].set_ylabel('Acceleration [m/s^2] / Angular velocity [deg/s]')
    
    # sets ticklabels 
    for x in range(3):
        ax[x].set_xlabel('Time [ms]')

        temp = ax[x].xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::5]))
        for label in temp:
            label.set_visible(False)

    # adding heatmaps
    fig, ax = plt.subplots(ncols=3, figsize=(30,6))
    sns.heatmap(df_x[df_x['signal'] == signal].drop(['signal'], axis=1).corr(), ax=ax[0])
    sns.heatmap(df_y[df_y['signal'] == signal].drop(['signal'], axis=1).corr(), ax=ax[1])
    sns.heatmap(df_circle[df_circle['signal'] == signal].drop(['signal'], axis=1).corr(), ax=ax[2])
    
    # set titles
    ax[0].set_title(f'Correlation matrix\n{title_text} {quantity} of `X` movements\n');
    ax[1].set_title(f'Correlation matrix\n{title_text} {quantity} of `Y` movements\n');
    ax[2].set_title(f'Correlation matrix\n{title_text} {quantity} of `circle` movements\n');