
import numpy as np
import csv
import os

metrics = {'fid50k','training-loss'}


experiments = {
    'MAE-XL': { 
        'fid50k': 'figure_plots/datas/MiT_XL_FID.csv',
        #'training-loss': 'training-loss.csv'
    },
    'MAE-L': { 
        'fid50k': 'figure_plots/datas/MiT_L_FID.csv',
        #'training-loss': 'training-loss.csv'
    },
    'MAE-B': { 
        'fid50k': 'figure_plots/datas/MiT_S_FID.csv',
        #'training-loss': 'training-loss.csv'
    },
    'MAE-H' : {
        'fid50k': 'figure_plots/datas/MiT_H_FID.csv',
        #'training-loss': 'training-loss.csv'
    },
    "DiT-XL": {
        'fid50k': 'figure_plots/datas/DiT_XL_FID.csv',
        #'training-loss': 'training-loss.csv'
    },
    "DiT-L": {
        'fid50k': 'figure_plots/datas/DiT_L_FID.csv',
        #'training-loss': 'training-loss.csv'
    },
    "DiT-B": {
        'fid50k': 'figure_plots/datas/DiT_B_FID.csv',
        #'training-loss': 'training-loss.csv'
    },
}
def load_metric_data(metric, only_dot = False):
    datas = {}
    for k in experiments.keys():
        datas[k] = []
        metric_path = experiments[k][metric]
        with open(metric_path, 'r') as f:
            reader = csv.reader(f)
            # remove the header
            next(reader)
            # for DiT, keeps first seven, for MAE, keeps first eight
            if 'DiT' in k:
                for i in range(1000):
                    dataslice = next(reader) # (x,y), keep only x
                    dataslice = dataslice[:2]
                    if int(dataslice[0]) > 4500:
                        break
                    dataslice[0] =(int(dataslice[0])// 500) * 50000
                    if dataslice[0] < 100000:
                        continue
                    dataslice[1] = float(dataslice[1])
                    if only_dot:
                        if dataslice[0] == 400000:
                            datas[k].append(dataslice)
                    else:
                        datas[k].append(dataslice)
            else:
                for i in range(9):
                    dataslice = next(reader) # (x,y), keep only x
                    dataslice = dataslice[:2]
                    dataslice[0] =int( 5e4 * i)
                    if dataslice[0] < 100000:
                        continue
                    dataslice[1] = float(dataslice[1])
                    if only_dot:
                        if dataslice[0] == 400000:
                            datas[k].append(dataslice)
                    else:
                        datas[k].append(dataslice)
    return datas

if __name__ == "__main__":
    metric = 'fid50k'
    datas = load_metric_data(metric)
    print(datas)