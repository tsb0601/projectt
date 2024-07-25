import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    np_data_mae = torch.load('plot_valid_mae.pt')
    print(type(np_data_mae))
    np_data_klvae = torch.load('plot_valid_klvae.pt')
    x_mae = np_data_mae["x"]
    y_mae = np_data_mae["y"]
    x_klvae = np_data_klvae["x"]
    y_klvae = np_data_klvae["y"]
    plt.plot(x_mae, y_mae, label='mae')
    plt.plot(x_klvae, y_klvae, label='klvae')
    plt.legend()
    plt.ylabel('mse loss')
    
    plt.savefig('plot_valid.png')