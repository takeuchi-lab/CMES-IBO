import os
import sys
import signal
import time
import pickle
import certifi

import numpy as np
import numpy.matlib
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection, svm, metrics
from sklearn.datasets import fetch_openml
signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(data_name = 'MNIST'):
    datas = list()
    for lr in [1e-3, 1e-2, 1e-1, 1.]:
        for batch_size in [32, 64, 128, 256]:
            for ch1 in [8, 16, 32, 64]:
                for ch2 in [8, 16, 32, 64]:
                    # for drop_rate in [1/16., 1/8., 1/4., 1/2.]:
                    for rho in np.arange(0, 2, 0.1):
                        try:
                            with open('./cnn_{}_data/X_{}_{}_{}_{}_{}.pickle'.format(data_name, lr, batch_size, ch1, ch2, round(rho, 1)), 'rb') as f:
                                data = pickle.load(f)
                            datas.append(data)
                        except FileNotFoundError as e:
                            print(e)


    datas = np.vstack(datas)
    datas = datas[datas[:,5] == 20]
    datas = np.delete(datas, [5, 17], axis=1)

    datas[:,0] = np.log10(datas[:,0])
    datas[:,1:4] = np.log2(datas[:,1:4])


    # for i in range(np.shape(datas)[0]):
    #     print(datas[i])
    print(np.shape(datas))
    print(datas[:10, :])

    # np.set_printoptions(threshold=10000)
    # print(datas[:,4])

    with open('./cnn_{}_data/cnn_{}_data.pickle'.format(data_name, data_name), 'wb') as f:
        pickle.dump(datas, f)


if __name__ == '__main__':
    args = sys.argv
    data_name = args[1]
    main(data_name)
