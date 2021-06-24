import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def downsample_df(df, period):
    '''Downsamples dataset.'''

    last_index_ms = df.index[-1]
    keep = np.arange(last_index_ms, step=period)

    return df.loc[keep]


def filter_df_by_signals(df, signals):
    '''Filter dataset byt signal'''
    return df.filter(regex=f'({"|".join(signals)})')


def run_inference(df, model, start, step):
    '''Runs inference.'''
    regex_filter = get_filter_string(start=start, step=step)
    data = list(df.filter(regex=f'_({regex_filter})$').loc[0])
    return model.score(data)


def calculate_error(res, move_type):
    '''Calculates inference error rate in validation data.'''

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


def get_move_from_path(s):
    '''Gets movement type from string path.'''
    import re
    return re.findall(r'_(x|circle|y)_', s)[0]


def run_validation(model_setup, dataset_path, dataset, is_plot=False, is_save_results=True):
    '''Plots validation results.'''
    validation_results = []
    
    if is_plot:
        fig, ax = plt.subplots(ncols=2, nrows=5, sharey=True, sharex=True, figsize=(30,25))
        
        # Create Legend
        blue_patch = mpatches.Patch(color='blue', label='X Movement')
        red_patch = mpatches.Patch(color='red', label='Y Movement')
        green_patch = mpatches.Patch(color='green', label='Circle Movement')
        fig.legend(handles=[blue_patch, red_patch, green_patch])
        
        fig.tight_layout()

    for setup in model_setup:
        # parse settings
        MODEL = setup[0]
        FREQ = setup[1]
        STEP = (1000 / FREQ)
        COL = setup[2][0]
        ROW = setup[2][1]

        df_val = pd.read_csv(dataset_path).set_index('ms')

        # initialize empty dataset to collect results
        inf_results = pd.DataFrame([],columns=['start','end','result'])

        # treat dataset 
        df_downsampled = downsample_df(df_val, STEP) # downsample dataset
        df_inference = transform_data_for_inference(df_downsampled) # converts dataset to inference format

        # generate a list of steps    
        inference_step = list(np.arange(0, df_val.index[-1] + 1 - 1010, step=STEP))

        results_for_plot = []
        
        for st in inference_step:
            res = np.argmax(run_inference(df_inference, MODEL, st, STEP))
            inf_results = pd.concat([inf_results, pd.DataFrame([{'start':st,'end':st+1000,'result':res}])], axis=0)

            if res in [1,2,3]:
                color = line_color(res)
                results_for_plot.append((color, st))
                    
                    
        # Plot signals
        if is_plot:
            ax[ROW][COL].plot(df_downsampled)
            
            for r in results_for_plot:
                color_plot = r[0]
                st_plot = r[1]   
                
                ax[ROW][COL].axvline(x=st_plot+500, ymin=0, ymax=0.4, color=color_plot, alpha=0.4)     
    
        # get error rate
        move = get_move_from_path(dataset_path)
        
        error_percentage = calculate_error(inf_results, move)
        validation_results.append({'model':setup[-1], 'hz':setup[1], 'dataset':dataset,'error_percentage': error_percentage, 'value_counts': inf_results['result'].value_counts()})

    if is_save_results:
        pd.DataFrame(validation_results).to_csv(f'output/validation/val_res_{dataset}_{move}.csv', index=False)
    
    return validation_results