import os
import sys
import signal
from abc import ABCMeta, abstractmethod
import pickle

import numpy as np
import numpy.matlib
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipyminimize

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../myutils"))
import myutils

signal.signal(signal.SIGINT, signal.SIG_DFL)

'''
https://www.sfu.ca/~ssurjano/index.html
Test functions in above site.
Test function for ordinary optimization is modified to multi-fidelity setting, but it can also be used to ordinary optimization.
Almost function is applied a minus because I want to consider about maximizing problem.
Parameters:
d : input dimension
M : number of fidelity
'''


class test_func():
    __metaclass__ = ABCMeta

    # def mf_values(self, input_list):
    #     '''
    #     return each fidelity output list

    #     Parameters:
    #     -----------
    #         input_list : list of numpy array
    #         list size is the numver of fidelity M
    #         each numpy array size is (N_m \times d), N_m is the number of data of each fidelity.

    #     Returns:
    #     --------
    #         output_list : list of numpy array
    #         each numpy array size is (N_m, 1)
    #     '''
    #     func_values_list = []
    #     for m in range(len(input_list)):
    #         if np.size(input_list[m]) != 0:
    #             func_values_list.append(self.values(input_list[m], fidelity=m) + np.random.normal(loc=0, scale=np.sqrt(self.noise_var) , size=(np.shape(input_list[m])[0], 1)))
    #         else:
    #             func_values_list.append(np.array([]))
    #     return func_values_list

    def mf_values(self, input_list):
        '''
        return each fidelity output list

        Parameters:
        -----------
            input_list : list of numpy array
            list size is the numver of fidelity M
            each numpy array size is (N_m \times d), N_m is the number of data of each fidelity.

        Returns:
        --------
            output_list : list of numpy array
            each numpy array size is (N_m, 1)
        '''
        func_values_list = []
        for m in range(len(input_list)):
            if np.size(input_list[m]) != 0:
                func_values_list.append(self.values(input_list[m], fidelity=m))
            else:
                func_values_list.append(np.array([]))
        return func_values_list

    @abstractmethod
    def values(self, input, fidelity=None):
        pass


def standard_length_scale(bounds):
    return (bounds[1] - bounds[0]) / 2.


class for_plot(test_func):
    '''
    Function for 1 dimensional example : d = 1, M = 2
    '''
    def __init__(self):
        self.d = 1
        self.M = 2
        self.bounds = np.array([[-np.pi], [2*np.pi]])
        X = np.c_[np.linspace(self.bounds[0, 0], self.bounds[1, 0], 100)]
        self.X = [X for m in range(self.M)]


    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        x = np.atleast_2d(input)
        if fidelity == 1:
            return -1*(np.cos(3*x) + 0.75*np.sin(2*x) - 0.1*x)
        elif fidelity == 0:
            return -1*(np.cos(3*x) + 0.5*np.sin(2*x) - 0.2*x)


class SynFun(test_func):
    '''
    Synthetic Function: d = 3, M = 2
    '''
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.M = 2
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]

        # self.w_1 = np.sqrt(np.array([[0.9], [0.9]]))
        # self.w_2 = np.sqrt(np.array([[0.05], [0.05]]))
        # self.kappa_1 = np.array([0.025, 0.025])
        # self.kappa_2 = np.array([0.025, 0.025])
        # ell_for_MT = 3.122156766261385

        self.w_1 = np.sqrt(np.array([[0.8], [0.8]]))
        self.w_2 = np.sqrt(np.array([[0.1], [0.1]]))
        self.kappa_1 = np.array([0.05, 0.05])
        self.kappa_2 = np.array([0.05, 0.05])

        # np.sqrt(1 / (2 * ( - np.log(cov))))
        # ell_for_MT = 2.178442285330266
        self.C_1 = np.linalg.cholesky(self.w_1.dot(self.w_1.T) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.dot(self.w_2.T) + np.diag(self.kappa_2))
        self.ell = 0.2
        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 2000
        basis_dim = feature_size//(2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)

        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]

    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)


class const_SynFun(test_func):
    '''
    Synthetic Function: d, C can be set arbitrary
    '''
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 10
        # self.ell = np.array([0.1, 0.2, 0.2])
        # self.ell = np.r_[np.array([0.1]), 0.2*np.ones(self.C)]
        self.ell = 0.2


        # self.d = 5
        # self.ell = 0.25
        # self.C = 2

        self.g_thresholds = -0.75 * np.ones(self.C)
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 500
        basis_dim = feature_size
        # all \ell is defined as same
        self.rbf_features_list = list()
        for c in range(self.C+1):
            self.rbf_features_list.append(myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim))
        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size, self.C+1))]

    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = 0
        elif (fidelity >= self.C+1) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = self.rbf_features_list[fidelity].transform(X)
        return np.c_[X_features.dot(self.coefficients[:,fidelity])]


class const_SynFun_compress(test_func):
    '''
    Synthetic Function: d, C can be set arbitrary
    '''
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 1
        self.latent_C = 10
        # self.ell = np.array([0.1, 0.2, 0.2])
        # self.ell = np.r_[np.array([0.1]), 0.2*np.ones(self.C)]
        self.ell = 0.2


        # self.d = 5
        # self.ell = 0.25
        # self.C = 2

        self.g_thresholds = -0.75 * np.ones(self.C)
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 500
        basis_dim = feature_size
        # all \ell is defined as same
        self.rbf_features_list = list()
        for c in range(self.latent_C+1):
            self.rbf_features_list.append(myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim))
        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size, self.latent_C+1))]

    def values(self, input, fidelity=None):
        X = np.atleast_2d(input)
        if fidelity is None:
            fidelity = 0
        elif (fidelity >= self.C+1) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        if fidelity == 0:
            X_features = self.rbf_features_list[fidelity].transform(X)
            return np.c_[X_features.dot(self.coefficients[:,fidelity])]
        elif fidelity == 1:
            output_list = list()
            for c in range(self.latent_C):
                X_features = self.rbf_features_list[c+1].transform(X)
                output_list.append(X_features.dot(self.coefficients[:,c+1]))
            output = np.min(np.vstack(output_list), axis=0)
            return np.c_[output]



class const_SynFun_noisy(test_func):
    '''
    Synthetic Function: d, C can be set arbitrary
    '''
    def __init__(self):
        self.noise_var = 1e-2
        self.d = 2
        self.C = 10
        # self.ell = np.array([0.1, 0.2, 0.2])
        # self.ell = np.r_[np.array([0.1]), 0.2*np.ones(self.C)]
        self.ell = 0.2


        # self.d = 5
        # self.ell = 0.25
        # self.C = 2

        self.g_thresholds = -0.75 * np.ones(self.C)
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 500
        basis_dim = feature_size
        # all \ell is defined as same
        self.rbf_features_list = list()
        for c in range(self.C+1):
            self.rbf_features_list.append(myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim))
        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size, self.C+1))]

    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = 0
        elif (fidelity >= self.C+1) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = self.rbf_features_list[fidelity].transform(X)
        return np.c_[X_features.dot(self.coefficients[:,fidelity])]


class const_SynFun_test(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 10
        self.M = self.C + 1
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]
        self.g_thresholds = -0.75 * np.ones(self.C)

        self.w_1 = np.sqrt(np.atleast_2d(0.00*np.ones(self.M)))
        self.w_2 = np.sqrt(np.atleast_2d(0.00*np.ones(self.M)))

        self.kappa_1 = 1.*np.ones(self.M)
        self.kappa_2 = 1.*np.ones(self.M)

        self.C_1 = np.linalg.cholesky(self.w_1.T.dot(self.w_1) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.T.dot(self.w_2) + np.diag(self.kappa_2))
        self.ell = 0.2

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 500*self.M
        basis_dim = feature_size//(2*self.M)
        feature_size = basis_dim * (2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)

        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]


    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)


class const_SynFun_pool(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 10
        self.M = self.C + 1
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]
        self.g_thresholds = -0.75 * np.ones(self.C)

        self.w_1 = np.sqrt(np.atleast_2d(0.00*np.ones(self.M)))
        self.w_1[:,-self.C//2:] = - self.w_1[:,-self.C//2:]
        self.w_2 = np.sqrt(np.atleast_2d(0.00*np.ones(self.M)))
        self.w_2[:,-self.C//2:] = - self.w_2[:,-self.C//2:]

        self.kappa_1 = 1.*np.ones(self.M)
        self.kappa_2 = 1.*np.ones(self.M)

        self.C_1 = np.linalg.cholesky(self.w_1.T.dot(self.w_1) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.T.dot(self.w_2) + np.diag(self.kappa_2))
        self.ell = 0.2

        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 1, 100)
        X1, X2 = np.meshgrid(x1, x2)
        self.X = np.c_[np.c_[X1.ravel()], np.c_[X2.ravel()]]

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 500*self.M
        basis_dim = feature_size//(2*self.M)
        feature_size = basis_dim * (2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)


        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]


    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)



class const_SynFun_minus_corr(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 8
        self.M = self.C + 1
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]
        self.g_thresholds = -1.*np.ones(self.C)

        self.w_1 = np.sqrt(np.atleast_2d(0.8*np.ones(self.M)))
        self.w_1[:,-self.C//2:] = - self.w_1[:,-self.C//2:]
        self.w_2 = np.sqrt(np.atleast_2d(0.1*np.ones(self.M)))
        self.w_2[:,-self.C//2:] = - self.w_2[:,-self.C//2:]

        self.kappa_1 = 0.05*np.ones(self.M)
        self.kappa_2 = 0.05*np.ones(self.M)

        self.C_1 = np.linalg.cholesky(self.w_1.T.dot(self.w_1) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.T.dot(self.w_2) + np.diag(self.kappa_2))
        self.ell = 0.1

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 2000
        basis_dim = feature_size//(2*self.M)
        feature_size = basis_dim * (2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)

        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]

    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)



class const_SynFun_plus_corr(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 10
        self.M = self.C + 1
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]
        self.g_thresholds = -0.75 * np.ones(self.C)

        self.w_1 = np.sqrt(np.atleast_2d(0.4*np.ones(self.M)))
        self.w_2 = np.sqrt(np.atleast_2d(0.4*np.ones(self.M)))

        self.kappa_1 = 0.1*np.ones(self.M)
        self.kappa_2 = 0.1*np.ones(self.M)

        self.C_1 = np.linalg.cholesky(self.w_1.T.dot(self.w_1) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.T.dot(self.w_2) + np.diag(self.kappa_2))
        self.ell = 0.2

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 2000
        basis_dim = feature_size//(2*self.M)
        feature_size = basis_dim * (2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)

        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]

    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)

class const_SynFun_plus_corr_pool(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 10
        self.M = self.C + 1
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]
        self.g_thresholds = -0.75 * np.ones(self.C)

        self.w_1 = np.sqrt(np.atleast_2d(0.00*np.ones(self.M)))
        self.w_1[:,-self.C//2:] = - self.w_1[:,-self.C//2:]
        self.w_2 = np.sqrt(np.atleast_2d(0.00*np.ones(self.M)))
        self.w_2[:,-self.C//2:] = - self.w_2[:,-self.C//2:]

        self.kappa_1 = 1.*np.ones(self.M)
        self.kappa_2 = 1.*np.ones(self.M)

        self.C_1 = np.linalg.cholesky(self.w_1.T.dot(self.w_1) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.T.dot(self.w_2) + np.diag(self.kappa_2))
        self.ell = 0.2

        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 1, 100)
        X1, X2 = np.meshgrid(x1, x2)
        self.X = np.c_[np.c_[X1.ravel()], np.c_[X2.ravel()]]

        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 500*self.M
        basis_dim = feature_size//(2*self.M)
        feature_size = basis_dim * (2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)

        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]


    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)




class SynFun_for_diffcost(test_func):
    '''
    Synthetic Function: d = 3, M = 2
    '''
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.M = 2
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]

        # self.w_1 = np.sqrt(np.array([[0.9], [0.9]]))
        # self.w_2 = np.sqrt(np.array([[0.05], [0.05]]))
        # self.kappa_1 = np.array([0.025, 0.025])
        # self.kappa_2 = np.array([0.025, 0.025])
        # ell_for_MT = 3.122156766261385

        self.w_1 = np.sqrt(np.array([[0.8], [0.8]]))
        self.w_2 = np.sqrt(np.array([[0.1], [0.1]]))
        self.kappa_1 = np.array([0.05, 0.05])
        self.kappa_2 = np.array([0.05, 0.05])

        # np.sqrt(1 / (2 * ( - np.log(cov))))
        # ell_for_MT = 2.178442285330266
        self.C_1 = np.linalg.cholesky(self.w_1.dot(self.w_1.T) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.dot(self.w_2.T) + np.diag(self.kappa_2))
        self.ell = 0.1
        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 1000
        basis_dim = feature_size//(2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)


        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]

    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)

class SynFun_for_ta(test_func):
    '''
    Synthetic Function: d = 3, M = 2
    '''
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.M = 4
        self.bounds = np.r_[np.c_[np.zeros(self.d)].T, np.c_[np.ones(self.d)].T]

        # self.w_1 = np.sqrt(np.array([[0.9], [0.9]]))
        # self.w_2 = np.sqrt(np.array([[0.05], [0.05]]))
        # self.kappa_1 = np.array([0.025, 0.025])
        # self.kappa_2 = np.array([0.025, 0.025])
        # ell_for_MT = 3.122156766261385

        self.w_1 = np.sqrt(np.array([[0.56], [0.64], [0.72], [0.8]]))
        self.w_2 = np.sqrt(np.array([[0.07], [0.08], [0.09], [0.1]]))
        self.kappa_1 = np.array([0.185, 0.14, 0.095, 0.05])
        self.kappa_2 = np.array([0.185, 0.14, 0.095, 0.05])

        # np.sqrt(1 / (2 * ( - np.log(cov))))
        # ell_for_MT = 2.178442285330266
        self.C_1 = np.linalg.cholesky(self.w_1.dot(self.w_1.T) + np.diag(self.kappa_1))
        self.C_2 = np.linalg.cholesky(self.w_2.dot(self.w_2.T) + np.diag(self.kappa_2))
        self.ell = 0.1
        # random seed for RFM sampling
        self.seed = 8
        self.func_sampling()

    def func_sampling(self, ell=None, seed=None):
        if not(ell is None):
            self.ell = ell
        if not(seed is None):
            self.seed = seed
        np.random.seed(self.seed)
        feature_size = 1000
        basis_dim = feature_size//(2*self.M)
        # all \ell is defined as same
        self.rbf_features_1 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)
        self.rbf_features_2 = myutils.RFM_RBF(lengthscales=self.ell*np.ones(self.d), input_dim=self.d, basis_dim=basis_dim)


        self.coefficients = np.c_[np.random.normal(0, 1, size=(feature_size))]

    def values(self, input, fidelity=None):
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        X = np.atleast_2d(input)
        X_features = list()
        for i in range(np.shape(X)[0]):
            X_features_1 = self.rbf_features_1.transform(X[i,:])
            X_features_2 = self.rbf_features_2.transform(X[i,:])
            X_features.append(np.c_[np.kron(self.C_1[fidelity, :], X_features_1), np.kron(self.C_2[fidelity, :], X_features_2)])
        X_features = np.vstack(X_features)
        return X_features.dot(self.coefficients)




class Gramacy(test_func):
    '''
    Gramacy et al. 2014,  Function: d = 2, C = 2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0], [1, 1]])
        self.d = 2
        self.C = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 0.5998
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        if fidelity==0 or fidelity is None:
            return - np.c_[x1 + x2]
        elif fidelity==1:
            return np.c_[np.sin(2 * np.pi * (x1**2 - 2*x2)) / 2. + x1 + 2*x2 - 1.5]
        elif fidelity==2:
            return np.c_[- x1**2 - x2**2 + 1.5]
        else:
            print('Not implemented fidelity')
            exit(1)

class Gardner1(test_func):
    '''
    Gardner et al. 2014,  Function: d = 2, C = 1
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0], [6, 6]])
        self.d = 2
        self.C = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        # self.maximum = 0
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        if fidelity==0 or fidelity is None:
            return - np.c_[np.cos(2*x1)*np.cos(x2) + np.sin(x1)]
        elif fidelity==1:
            # original paper is mistaken
            return - np.c_[np.cos(x1)*np.cos(x2) - np.sin(x1)*np.sin(x2) + 0.5]
        else:
            print('Not implemented fidelity')
            exit(1)

class Gardner2(test_func):
    '''
    Gardner et al. 2014,  Function: d = 2, C = 1
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0], [6, 6]])
        self.d = 2
        self.C = 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        # self.maximum = 0
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        if fidelity==0 or fidelity is None:
            return - np.c_[np.sin(x1) + x2]
        elif fidelity==1:
            return - np.c_[np.sin(x1) * np.sin(x2) + 0.95]
        else:
            print('Not implemented fidelity')
            exit(1)

class DTLZ1(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.C = 2
        self.M = self.C + 1 # Number of objective anc constraint functions
        self.bounds = np.array([np.zeros(self.d), np.ones(self.d)])
        # self.standard_length_scale = standard_length_scale(self.bounds)
        # self.maximum = 0
        self.g_thresholds = - 0.1*np.ones(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]

        g_temp = (input[:,self.C - self.d:] - 0.5)**2 - np.cos(20*np.pi*(input[:,self.C - self.d:] - 0.5))
        g = 100 * ( (self.d - self.C) + np.sum(np.c_[g_temp], axis=1))
        if fidelity==0 or fidelity is None:
            return - np.c_[(1 + g) / 2. * np.prod(input[:,:self.C], axis=1)]
        elif fidelity==1:
            return - np.c_[(1 + g) / 2. * np.prod(input[:,:self.C-1], axis=1) * (1 - x2)]
        elif fidelity==2:
            return - np.c_[(1 + g) / 2.  * (1 - x1)]
        else:
            print('Not implemented fidelity')
            exit(1)

class DTLZ2(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.C = 2
        self.M = self.C + 1 # Number of objective anc constraint functions
        self.bounds = np.array([np.zeros(self.d), np.ones(self.d)])
        # self.standard_length_scale = standard_length_scale(self.bounds)
        # self.maximum = 0
        self.g_thresholds = - 0.25*np.ones(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]

        g_temp = (input[:,self.C - self.d:] - 0.5)**2
        g = np.sum(np.c_[g_temp], axis=1)
        if fidelity==0 or fidelity is None:
            return - np.c_[(1 + g) / 2. * np.prod( np.cos(input[:,:self.C]*np.pi / 2.), axis=1)]
        elif fidelity==1:
            return - np.c_[(1 + g) / 2. * np.prod(np.cos(input[:,:self.C-1]*np.pi / 2.), axis=1) * np.sin(x2 * np.pi / 2.)]
        elif fidelity==2:
            return - np.c_[(1 + g) / 2. * np.sin(x1 * np.pi / 2.)]
        else:
            print('Not implemented fidelity')
            exit(1)


class DTLZ3(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.C = 2
        self.M = self.C + 1 # Number of objective anc constraint functions
        self.bounds = np.array([np.zeros(self.d), np.ones(self.d)])
        # self.standard_length_scale = standard_length_scale(self.bounds)
        # self.maximum = 0
        self.g_thresholds = - 0.25*np.ones(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]

        g_temp = (input[:,self.C - self.d:] - 0.5)**2 - np.cos(20*np.pi*(input[:,self.C - self.d:] - 0.5))
        g = 100 * ( (self.d - self.C) + np.sum(np.c_[g_temp], axis=1))
        if fidelity==0 or fidelity is None:
            return - np.c_[(1 + g) / 2. * np.prod( np.cos(input[:,:self.C]*np.pi / 2.), axis=1)]
        elif fidelity==1:
            return - np.c_[(1 + g) / 2. * np.prod(np.cos(input[:,:self.C-1]*np.pi / 2.), axis=1) * np.sin(x2 * np.pi / 2.)]
        elif fidelity==2:
            return - np.c_[(1 + g) / 2. * np.sin(x1 * np.pi / 2.)]
        else:
            print('Not implemented fidelity')
            exit(1)


class DTLZ4(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.C = 2
        self.M = self.C + 1 # Number of objective anc constraint functions
        self.bounds = np.array([np.zeros(self.d), np.ones(self.d)])
        # self.standard_length_scale = standard_length_scale(self.bounds)
        # self.maximum = 0
        self.g_thresholds = - 0.25*np.ones(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input.copy())
        alpha = 100
        input[:,:self.C] = input[:,:self.C]**alpha
        x1 = input[:,0]
        x2 = input[:,1]

        g_temp = (input[:,self.C - self.d:] - 0.5)**2
        g = np.sum(np.c_[g_temp], axis=1)
        if fidelity==0 or fidelity is None:
            return - np.c_[(1 + g) / 2. * np.prod( np.cos(input[:,:self.C]*np.pi / 2.), axis=1)]
        elif fidelity==1:
            return - np.c_[(1 + g) / 2. * np.prod(np.cos(input[:,:self.C-1]*np.pi / 2.), axis=1) * np.sin(x2 * np.pi / 2.)]
        elif fidelity==2:
            return - np.c_[(1 + g) / 2. * np.sin(x1 * np.pi / 2.)]
        else:
            print('Not implemented fidelity')
            exit(1)


class DTLZ5(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.C = 2
        self.M = self.C + 1 # Number of objective anc constraint functions
        self.bounds = np.array([np.zeros(self.d), np.ones(self.d)])
        self.g_thresholds = - 0.4*np.ones(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input.copy())
        g_temp = (input[:,self.C - self.d:] - 0.5)**2
        g = np.sum(np.c_[g_temp], axis=1)
        input[:,1:self.C] = (1 + 2 * np.c_[g] * input[:,1:self.C]) / (2 * (1 + np.c_[g]))
        x1 = input[:,0]
        x2 = input[:,1]

        if fidelity==0 or fidelity is None:
            return - np.c_[(1 + g) / 2. * np.prod( np.cos(input[:,:self.C]*np.pi / 2.), axis=1)]
        elif fidelity==1:
            return - np.c_[(1 + g) / 2. * np.prod(np.cos(input[:,:self.C-1]*np.pi / 2.), axis=1) * np.sin(x2 * np.pi / 2.)]
        elif fidelity==2:
            return - np.c_[(1 + g) / 2. * np.sin(x1 * np.pi / 2.)]
        else:
            print('Not implemented fidelity')
            exit(1)


class DTLZ6(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 3
        self.C = 2
        self.M = self.C + 1 # Number of objective anc constraint functions
        self.bounds = np.array([np.zeros(self.d), np.ones(self.d)])
        self.g_thresholds = - 0.4*np.ones(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input.copy())
        g_temp = input[:,self.C - self.d:]**0.1
        g = np.sum(np.c_[g_temp], axis=1)
        input[:,1:self.C] = (1 + 2 * np.c_[g] * input[:,1:self.C]) / (2 * (1 + np.c_[g]))
        x1 = input[:,0]
        x2 = input[:,1]

        if fidelity==0 or fidelity is None:
            return - np.c_[(1 + g) / 2. * np.prod( np.cos(input[:,:self.C]*np.pi / 2.), axis=1)]
        elif fidelity==1:
            return - np.c_[(1 + g) / 2. * np.prod(np.cos(input[:,:self.C-1]*np.pi / 2.), axis=1) * np.sin(x2 * np.pi / 2.)]
        elif fidelity==2:
            return - np.c_[(1 + g) / 2. * np.sin(x1 * np.pi / 2.)]
        else:
            print('Not implemented fidelity')
            exit(1)



class Test_MESC(test_func):
    '''
    Example for Irrational behavior of CMES.
    '''
    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0], [1, 1]])
        self.d = 2
        self.C = 6
        self.standard_length_scale = standard_length_scale(self.bounds)
        # self.maximum = 0
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0] - 0.5
        x2 = input[:,1] - 0.5
        if fidelity==0 or fidelity is None:
            return np.c_[np.exp(- np.sum((input - np.atleast_2d([0.5, 0.5]))**2, axis=1))]
        elif fidelity==1:
            return np.exp(np.c_[ x1 - x2] + 0.1) - 1
        elif fidelity==2:
            return np.exp(np.c_[-x1 + x2] + 0.1) - 1
        elif fidelity==3:
            return np.exp(np.c_[ x1 + x2] + 0.1) - 1
        elif fidelity==4:
            return np.exp(np.c_[-x1 - x2] + 0.1) - 1
        elif fidelity==5:
            return np.exp(np.c_[x1 + 0.05]) - 1
        elif fidelity==6:
            return np.exp(np.c_[- x1 + 0.05]) - 1
        else:
            print('Not implemented fidelity')
            exit(1)


class G1(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 13
        self.C = 9
        self.bounds = np.array([np.zeros(self.d), np.r_[np.ones(9), 100*np.ones(3), [1]]])
        self.g_thresholds = np.zeros(self.C)
        # x_star = [1,1,1,1,1,1,1,1,1,3,3,3,1]
        self.f_star = 15
        self.f_min = -5

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        x8 = input[:,7]
        x9 = input[:,8]
        x10 = input[:,9]
        x11 = input[:,10]
        x12 = input[:,11]
        x13 = input[:,12]

        if fidelity==0 or fidelity is None:
            return - np.c_[ 5 * np.sum(input[:,:4], axis=1) - 5*np.sum(input[:,:4]**2, axis=1) - np.sum(input[:,4:], axis=1)]
        elif fidelity==1:
            return - np.c_[2*x1 + 2*x2 + x10 + x11 - 10]
        elif fidelity==2:
            return - np.c_[ 2*x1 + 2*x3 + x10 + x12 - 10]
        elif fidelity==3:
            return - np.c_[ 2*x2 + 2*x3 + x11 + x12 - 10]
        elif fidelity==4:
            return - np.c_[- 8*x1 + x10]
        elif fidelity==5:
            return - np.c_[- 8*x2 + x11]
        elif fidelity==6:
            return - np.c_[- 8*x3 + x12]
        elif fidelity==7:
            return - np.c_[- 2*x4 - x5 + x10]
        elif fidelity==8:
            return - np.c_[- 2*x6 - x7 + x11]
        elif fidelity==9:
            return - np.c_[- 2*x8 - x9 + x12]
        else:
            print('Not implemented fidelity')
            exit(1)


# class G2(test_func):
#     '''
#     too many dimension d = 20
#     '''
#     def __init__(self):


#     def values(self, input, fidelity=None):

# class G3(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):


#     def values(self, input, fidelity=None):


class G4(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 5
        self.C = 6
        self.bounds = np.array([[78, 33, 27, 27, 27], [102, 45, 45, 45, 45]])
        self.g_thresholds = np.zeros(self.C)
        # x_star = [78,33,29.9952560256815985,45,36.7758129057882073]
        self.f_star = 30665.538671783328
        self.f_min = 22302.761885500004

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        if fidelity==0 or fidelity is None:
            return - np.c_[ 5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141]
        elif fidelity==1:
            return - np.c_[ 85.334407 + 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5 - 92]
        elif fidelity==2:
            return - np.c_[ - 85.334407 - 0.0056858*x2*x5 - 0.0006262*x1*x4 + 0.0022053*x3*x5]
        elif fidelity==3:
            return - np.c_[ 80.51249 + 0.0071317*x2*x5 + 0.0029955*x1*x2 + 0.0021813*x3**2 - 110]
        elif fidelity==4:
            return - np.c_[ - 80.51249 - 0.0071317*x2*x5 - 0.0029955*x1*x2 - 0.0021813*x3**2 + 90]
        elif fidelity==5:
            return - np.c_[ 9.300961 + 0.0047026*x3*x5 + 0.0012547*x1*x3 + 0.0019085*x3*x4 - 25]
        elif fidelity==6:
            return - np.c_[ - 9.300961 - 0.0047026*x3*x5 - 0.0012547*x1*x3 - 0.0019085*x3*x4 + 20]
        else:
            print('Not implemented fidelity')
            exit(1)


# class G5(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):


#     def values(self, input, fidelity=None):


# class G6(test_func):
#     def __init__(self):

#         self.d = 2
#         self.C = 2
#         self.bounds = np.array([[13, 0], [100, 100]])
#         self.g_thresholds = np.zeros(self.C)
#         # x_star = [14.09500000000000064, 0.8429607892154795668]
#         self.f_star = 6961.81387558103
#         self.f_min = -1241000.0000000005

#     def values(self, input, fidelity=None):
#         input = np.atleast_2d(input)
#         x1 = input[:,0]
#         x2 = input[:,1]
#         if fidelity==0 or fidelity is None:
#             return - np.c_[ (x1 - 10)**3 + (x2 - 20)**3]
#         elif fidelity==1:
#             return - np.c_[ - (x1 - 5)**2 - (x2 - 5)**2 + 100]
#         elif fidelity==2:
#             return - np.c_[ (x1 - 6)**2 + (x2 - 5)**2 - 82.81]
#         else:
#             print('Not implemented fidelity')
#             exit(1)

class G7(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 10
        self.C = 8
        self.bounds = np.array([-10*np.ones(self.d), 10*np.ones(self.d)])
        self.g_thresholds = np.zeros(self.C)
        # x_star = [2.17199634142692, 2.3636830416034, 8.77392573913157, 5.09598443745173, 0.990654756560493, 1.43057392853463, 1.32164415364306, 9.82872576524495, 8.2800915887356, 8.3759266477347]
        self.f_star = -24.306209068179676
        self.f_min = -7032.0

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        x8 = input[:,7]
        x9 = input[:,8]
        x10 = input[:,9]
        if fidelity==0 or fidelity is None:
            return - np.c_[ x1**2 + x2**2 + x1*x2 - 14*x1 - 16*x2 + (x3 - 10)**2 + 4*(x4 - 5)**2 + (x5 - 3)**2 + 2*(x6 - 1)**2 + 5*x7**2 + 7*(x8 -11)**2 + 2*(x9 - 10)**2 + (x10 - 7)**2 + 45]
        elif fidelity==1:
            return - np.c_[ - 105 + 4*x1 + 5*x2 - 3*x7 + 9*x8]
        elif fidelity==2:
            return - np.c_[ 10*x1 - 8*x2 - 17*x7 + 2*x8]
        elif fidelity==3:
            return - np.c_[ - 8*x1 + 2*x2 + 5*x9 - 2*x10 - 12]
        elif fidelity==4:
            return - np.c_[ 3*(x1 - 2)**2 + 4*(x2 - 3)**2 + 2*x3**2 - 7*x4 - 120]
        elif fidelity==5:
            return - np.c_[ 5*x1**2 + 8*x2 + (x3 - 6)**2 - 2*x4 - 40]
        elif fidelity==6:
            return - np.c_[x1**2 + 2*(x2 - 2)**2 - 2*x1*x2 + 14*x5 - 6*x6]
        elif fidelity==7:
            return - np.c_[ 0.5*(x1 - 8)**2 + 2*(x2 - 4)**2 + 3*x5**2 - x6 - 30]
        elif fidelity==8:
            return - np.c_[ -3*x1 + 6*x2 + 12*(x9 - 8)**2 - 7*x10]
        else:
            print('Not implemented fidelity')
            exit(1)

# class G8(test_func):
#     def __init__(self):
#         self.d = 2
#         self.C =2
#         self.bounds = np.array([[1e-4,1e-4], [10,10]])
#         self.g_thresholds = np.zeros(self.C)
#         # x_star = [1.22797135260752599, 4.24537336612274885]
#         self.f_star =   0.09582504141791358
#         self.f_min = - 90.35168915426715

#     def values(self, input, fidelity=None):
#         input = np.atleast_2d(input)
#         x1 = input[:,0]
#         x2 = input[:,1]
#         if fidelity==0 or fidelity is None:
#             return - np.c_[- (np.sin(2*np.pi*x1)**3 * np.sin(2*np.pi*x2)) / (x1**3 * (x1 + x2))]
#         elif fidelity==1:
#             return - np.c_[x1**2 - x2 + 1]
#         elif fidelity==2:
#             return - np.c_[1 - x1 + (x2 - 4)**2]
#         else:
#             print('Not implemented fidelity')
#             exit(1)

class G9(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 7
        self.C = 4
        self.bounds = np.array([-10*np.ones(self.d), 10*np.ones(self.d)])
        self.g_thresholds = np.zeros(self.C)
        # x_star = [2.33049935147405174, 1.95137236847114592, −0.477541399510615805, 4.36572624923625874, −0.624486959100388983, 1.03813099410962173, 1.5942266780671519]
        self.f_star = - 680.6300573767139
        self.f_min = - 10025263.0

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        if fidelity==0 or fidelity is None:
            return - np.c_[(x1 - 10)**2 + 5*(x2-12)**2 + x3**4 + 3*(x4 - 11)**2 + 10*x5**6 + 7*x6**2 + x7**4 - 4*x6*x7 - 10*x6 - 8*x7]
        elif fidelity==1:
            return - np.c_[-127 + 2*x1**2 + 3*x2**4 + x3 + 4*x4**2 + 5*x5]
        elif fidelity==2:
            return - np.c_[-282 + 7*x1 + 3*x2 + 10*x3**2 + x4 - x5]
        elif fidelity==3:
            return - np.c_[-196 + 23*x1 + x2**2 + 6*x6**2 - 8*x7]
        elif fidelity==4:
            return - np.c_[4*x1**2 + x2**2 - 3*x1*x2 + 2*x3**2 + 5*x6 - 11*x7]
        else:
            print('Not implemented fidelity')
            exit(1)

class G10(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 8
        self.C = 6
        self.bounds = np.array([[100, 1000, 1000, 10, 10, 10 ,10, 10], [10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000]])
        self.g_thresholds = np.zeros(self.C)
        # x_star = [579.306685017979589,1359.97067807935605,5109.97065743133317,182.01769963061534, 295.601173702746792, 217.982300369384632, 286.41652592786852, 395.601173702746735]
        self.f_star = -7049.248020528817
        self.f_min = -30000.0

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        x8 = input[:,7]
        if fidelity==0 or fidelity is None:
            return -np.c_[x1 + x2 + x3]
        elif fidelity==1:
            return -np.c_[ -1 + 0.0025*(x4 + x6)]
        elif fidelity==2:
            return -np.c_[ -1 + 0.0025*(x5 + x7 - x4)]
        elif fidelity==3:
            return -np.c_[ -1 + 0.01*(x8 - x5)]
        elif fidelity==4:
            return -np.c_[ -x1*x6 + 833.33252*x4 + 100*x1 - 83333.333]
        elif fidelity==5:
            return -np.c_[ - x2*x7 + 1250*x5 + x2*x4 - 1250*x4]
        elif fidelity==6:
            return -np.c_[ -x3*x8 + 1250000 + x3*x5 - 2500*x5]
        else:
            print('Not implemented fidelity')
            exit(1)

# class G11(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):


# class G12(test_func):
#     '''
#     too many constraints exist (9**3)
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):


# class G13(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):

# class G14(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):

# class G15(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):

# class G16(test_func):
#     '''
#     too many constraints are needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):

# class G17(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):

class G18(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 9
        self.C = 13
        self.bounds = np.array([ np.r_[-10*np.ones(8), [0] ], np.r_[ 10*np.ones(8), [20] ]])
        self.g_thresholds = np.zeros(self.C)
        # x_star = [−0.657776192427943163, −0.153418773482438542, 0.323413871675240938, −0.946257611651304398, − 0.657776194376798906, −0.753213434632691414, 0.323413874123576972, −0.346462947962331735, 0.59979466285217542]
        self.f_star =  0.8660254037852283
        self.f_min = -400

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        x8 = input[:,7]
        x9 = input[:,8]
        if fidelity==0 or fidelity is None:
            return -np.c_[ -0.5*( x1*x4 - x2*x3 + x3*x9 - x5*x9 + x5*x8 - x6*x7)]
        elif fidelity==1:
            return -np.c_[ x3**2 + x4**2 - 1]
        elif fidelity==2:
            return -np.c_[ x9**2 - 1]
        elif fidelity==3:
            return -np.c_[ x5**2 + x6**2 - 1]
        elif fidelity==4:
            return -np.c_[ x1**2 + (x2 - x9)**2 - 1]
        elif fidelity==5:
            return -np.c_[ (x1 - x5)**2 + (x2 - x6)**2 - 1]
        elif fidelity==6:
            return -np.c_[ (x1 - x7)**2 + (x2 - x8)**2 - 1]
        elif fidelity==7:
            return -np.c_[ (x3 - x5)**2 + (x4 - x6)**2 - 1]
        elif fidelity==8:
            return -np.c_[ (x3 - x7)**2 + (x4 - x8)**2 - 1]
        elif fidelity==9:
            return -np.c_[x7**2 + (x8 - x9)**2 - 1]
        elif fidelity==10:
            return -np.c_[ x2*x3 - x1*x4]
        elif fidelity==11:
            return -np.c_[ - x3*x9]
        elif fidelity==12:
            return -np.c_[ x5*x9 ]
        elif fidelity==13:
            return -np.c_[x6*x7 - x5*x8]
        else:
            print('Not implemented fidelity')
            exit(1)

# class G19(test_func):
#     '''
#     dimension is very high d=15
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):


# class G20(test_func):
#     '''
#     equality condition is needed, too many dimension d
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):

# class G21(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):


# class G22(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):


# class G23(test_func):
#     '''
#     equality condition is needed
#     '''
#     def __init__(self):

#     def values(self, input, fidelity=None):

# class G24(test_func):
#     def __init__(self):
#         self.d = 2
#         self.C = 2
#         self.bounds = np.array([[0, 0], [3, 4]])
#         self.g_thresholds = np.zeros(self.C)
#         # x_star = [2.329520197477623.17849307411774]
#         self.f_star = 5.508013271595843
#         self.f_min = 0.0

#     def values(self, input, fidelity=None):
#         input = np.atleast_2d(input)
#         x1 = input[:,0]
#         x2 = input[:,1]
#         if fidelity==0 or fidelity is None:
#             return -np.c_[ - x1 - x2]
#         elif fidelity==1:
#             return -np.c_[ -2*x1**4 + 8*x1**3 - 8*x1**2 + x2 - 2]
#         elif fidelity==2:
#             return -np.c_[ -4*x1**4 + 32*x1**3 - 88*x1**2 + 96*x1 + x2 - 36]
#         else:
#             print('Not implemented fidelity')
#             exit(1)

class WeldedBeam(test_func):
    '''
    Coello Coello, C. A. and Montes, E. M. (2002),
    Hedar, A.-R. and Fukushima, 2006.
    Function: d = 4, C = 6
    '''
    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0.1, 0.1, 0.1, 0.1], [2., 10, 10, 2.]])
        self.d = 4
        self.C = 7
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 1.7250022
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        P = 6000.
        L = 14.
        E = 30. * 1e6
        G = 12. * 1e6
        tau_max = 13600.
        sigma_max = 30000.
        delta_max = 0.25
        if fidelity==0 or fidelity is None:
            return - np.c_[1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (L + x2)]
        elif fidelity==1:
            tau_prime = P/(np.sqrt(2)*x1*x2)
            M = P * (L + x2 / 2.)
            R = np.sqrt(x2**2 / 4. + ((x1 + x3) / 2.)**2)
            J = 2 * (0.707*x1*x2*(x2**2 / 12. + ((x1 + x3) / 2.)**2))
            tau_prime_2 = M * R / J
            return - np.c_[np.sqrt(tau_prime**2 + tau_prime*tau_prime_2*x2 / (R) + tau_prime_2**2) - tau_max]
        elif fidelity==2:
            return -np.c_[(6*P*L) / (x4 * x3**2) - sigma_max]
        elif fidelity==3:
            return -np.c_[x1 - x4]
        elif fidelity==4:
            return -np.c_[0.10471*x1**2 + 0.04811*x3*x4*(14. + x2) - 5.0]
        elif fidelity==5:
            return -np.c_[4*P*L**3 / (E*x3**3*x4) - delta_max]
        elif fidelity==6:
            return -np.c_[P - 4.013*np.sqrt( E * G * x3**2 * x4**6 / 36.) / L**2 * (1 - x3 / (2*L) * np.sqrt(E / (4*G)))]
        elif fidelity==7:
            return -np.c_[0.125 - x1]
        else:
            print('Not implemented fidelity')
            exit(1)

class PressureVessel(test_func):
    '''
    Coello Coello, C. A. and Montes, E. M. (2002),
    Hedar, A.-R. and Fukushima, 2006.
    Scalable constrained bayesian optimization (input domain)
    Function: d = 4, C = 4
    '''
    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[1., 1., 10., 10], [99, 99, 200, 200]])
        self.d = 4
        self.C = 4
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 8796.862246556188
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        # input 1 and 2 must be integer multiplied 0.0625
        input[:,0:2] = input[:,0:2] / 0.0625
        remainder = - input[:,0:2] % 1
        remainder[remainder < -0.5] = remainder[remainder < -0.5] + 1
        input[:,0:2] = (input[:,0:2] + remainder) * 0.0625
        if fidelity==0 or fidelity is None:
            return -np.c_[0.6224*x1*x3*x4 + 1.7781*x2*x3**2+3.1661*x1**2*x4+19.84*x1**2*x3]
        elif fidelity == 1:
            return - np.c_[- x1 + 0.0193*x3]
        elif fidelity == 2:
            return - np.c_[- x2 + 0.00954*x3]
        elif fidelity == 3:
            return - np.c_[-np.pi*x3**2*x4 -4.*np.pi/3. *x3**3 + 1296000]
        elif fidelity == 4:
            return - np.c_[x4 - 240.]

class TensionCompressionString(test_func):
    '''
    Coello Coello, C. A. and Montes, E. M. (2002),
    Hedar, A.-R. and Fukushima, 2006.
    Scalable constrained bayesian optimization (input domain)
    Function: d = 4, C = 4
    '''
    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0.05, 0.25, 2.], [2., 1.3, 15]])
        self.d = 3
        self.C = 4
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 0.012665285
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        if fidelity==0 or fidelity is None:
            return -np.c_[(x3 + 2.)*x2*x1**2]
        elif fidelity == 1:
            return -np.c_[ 1 - x2**3*x3 / (71785 * x1**4)]
        elif fidelity == 2:
            return - np.c_[(4*x2**2 - x1*x2) / (12566*(x2 * x1**3 - x1**4)) + 1./(5108 * x1**2) - 1.]
        elif fidelity == 3:
            return - np.c_[1 - 140.45*x1 / (x2**2 * x3)]
        elif fidelity == 4:
            return -np.c_[(x1 + x2) / 1.5 - 1.]



class const_HartMann6(test_func):
    '''
    HartMann 6-dimensional function: d = 6, C = 1
    Jalali et al. (2017).
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
        self.d = 6
        self.C = 1
        self.g_thresholds = np.zeros(self.C)
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.32237

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)

        if fidelity==0 or fidelity is None:
            return self._common_processing(input)
        elif fidelity == 1:
            return - np.c_[np.sqrt(np.sum(input**2, axis=1)) - 1]
        else:
            print('Not implemented fidelity')
            exit(1)

    def _common_processing(self, input):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])

        P = np.array([[1312, 1696, 5569, 124, 8283, 5886],
                      [2329, 4135, 8307, 3736, 1004, 9991],
                      [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091, 381]])*1e-4

        values = 0
        for i in range(4):
            inner = 0
            for j in range(6):
                inner -= A[i, j]*(input[:, j] - P[i, j])**2
            values += alpha[i]*np.power(np.e, inner)
        return np.c_[values]


class const_cnn_cifar10(test_func):
    '''
    CNN CIFAR10 real data Function: d = 5, M = 3
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-3., 5., 3., 3., 0.], [0., 8., 6., 6., 2.]])
        self.d = 5
        self.C = 10
        self.M = self.C + 1
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 0

        #データの読み込み
        with open('../test_functions/AutoML_data_CBO/cnn_CIFAR10_data/cnn_CIFAR10_data.pickle', 'rb') as f:
            data = pickle.load(f)

        self.g_thresholds = 0.5 * np.ones(self.C)

        feasible_index = np.where(np.all(data[:,5:15] >= self.g_thresholds, axis=1) == True)[0]
        print('number of feasible points:', np.size(feasible_index))

        # self.X = [data[:,0:5] for _ in range(self.M)]
        self.X = data[:,0:5]
        self.Y = [data[:,15]]
        self.Y.extend([data[:,5+i] for i in range(self.C)])

        # logit transformation
        self.Y = [ np.where(Y <= 0, 1e-5, Y) for Y in self.Y ]
        self.Y = [ np.where(Y >= 1, 1 - 1e-5, Y) for Y in self.Y ]
        self.Y = [ np.log(Y / (1 - Y)) for Y in self.Y ]
        self.g_thresholds = np.log(self.g_thresholds / (1 - self.g_thresholds))

        self.CONST_MAX = np.max(self.Y[0][feasible_index])
        self.GLOBAL_MIN = np.min(self.Y[0])

        print(self.CONST_MAX, np.max(self.Y[0]), self.GLOBAL_MIN)

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        elif (fidelity >= self.M) or (fidelity < 0):
            print('Not implemented fidelity')
            exit(1)

        match_index = list()
        for i in range(np.shape(input)[0]):
            tmp_match_index = np.where( np.all(self.X == input[i], axis=1) == True)[0]
            match_index.append(tmp_match_index)
            # match_index.append(np.where((self.X[0][:,0] == input[i, 0]) & (self.X[0][:,1] == input[i, 1]))[0])
        match_index = np.array(match_index).ravel()

        return np.c_[self.Y[fidelity][match_index]]



class HeatExchangerNetworkDesign1(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 9
        self.C = 16

        self.bounds = np.array([
            [0, 0, 0, 0, 1000, 0, 100, 100 + 1e-4, 600],
            [10, 200, 100, 200, 2000000, 600, 600 - 1e-4, 600, 900]])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 1.8931162966 * 1e2
        self.g_thresholds = np.zeros(self.C) - 1e-3

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        x8 = input[:,7]
        x9 = input[:,8]

        if fidelity==0 or fidelity is None:
            return - np.c_[35 * x1**0.6 + 35*x2**0.6]
        elif fidelity == 1:
            return - np.c_[ 200*x1*x4 - x3]
        elif fidelity == 2:
            return np.c_[ 200*x1*x4 - x3]
        elif fidelity == 3:
            return - np.c_[200*x2*x6 - x5]
        elif fidelity == 4:
            return np.c_[200*x2*x6 - x5]
        elif fidelity == 5:
            return - np.c_[x3 - 10000*(x7 - 100)]
        elif fidelity == 6:
            return np.c_[x3 - 10000*(x7 - 100)]
        elif fidelity == 7:
            return - np.c_[x5 - 10000*(300 - x7)]
        elif fidelity == 8:
            return np.c_[x5 - 10000*(300 - x7)]
        elif fidelity == 9:
            return - np.c_[x3 - 10000*(600 - x8)]
        elif fidelity == 10:
            return np.c_[x3 - 10000*(600 - x8)]
        elif fidelity == 11:
            return - np.c_[x5 - 10000*(900 - x9)]
        elif fidelity == 12:
            return np.c_[x5 - 10000*(900 - x9)]
        elif fidelity == 13:
            return - np.c_[x4 * np.log(x8 - 100) - x4*np.log(600 - x7) - x8 + x7 + 500]
        elif fidelity == 14:
            return np.c_[x4 * np.log(x8 - 100) - x4*np.log(600 - x7) - x8 + x7 + 500]
        elif fidelity == 15:
            return - np.c_[x6 * np.log(x9 - x7) - x6*np.log(600) - x9 + x7 + 600]
        elif fidelity == 16:
            return np.c_[x6 * np.log(x9 - x7) - x6*np.log(600) - x9 + x7 + 600]



class HeatExchangerNetworkDesign2(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 11
        self.C = 18

        self.bounds = np.array([
            [10**4, 10**4, 10**4, 0+1e-4, 0+1e-4, 0+1e-4, 100, 100, 100 + 1e-4, 200+1e-4, 300+1e-4],
            [81.9*10**4, 113.1*10**4, 205*10**4, 0.05074, 0.05074, 0.05074, 200, 300, 300, 300, 400]])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 7.0490369540 * 1e3
        self.g_thresholds = np.zeros(self.C) - 1e-3

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        x8 = input[:,7]
        x9 = input[:,8]
        x10 = input[:,9]
        x11 = input[:,10]

        if fidelity==0 or fidelity is None:
            return - np.c_[(x1 / (120*x4))**0.6 + (x2 / (80*x5))**0.6 + (x3 / (40*x6))**0.6]
        elif fidelity == 1:
            return - np.c_[ x1 - 10**4*(x7 - 100)]
        elif fidelity == 2:
            return np.c_[ x1 - 10**4*(x7 - 100)]
        elif fidelity == 3:
            return - np.c_[x2 - 10**4*(x8 - x7)]
        elif fidelity == 4:
            return np.c_[x2 - 10**4*(x8 - x7)]
        elif fidelity == 5:
            return - np.c_[x3 - 10**4*(500 - x8)]
        elif fidelity == 6:
            return np.c_[x3 - 10**4*(500 - x8)]
        elif fidelity == 7:
            return - np.c_[x1 - 10**4*(300 - x9)]
        elif fidelity == 8:
            return np.c_[x1 - 10**4*(300 - x9)]
        elif fidelity == 9:
            return - np.c_[x2 - 10**4*(400 - x10)]
        elif fidelity == 10:
            return np.c_[x2 - 10**4*(400 - x10)]
        elif fidelity == 11:
            return - np.c_[x3 - 10**4*(600 - x11)]
        elif fidelity == 12:
            return np.c_[x3 - 10**4*(600 - x11)]
        elif fidelity == 13:
            return - np.c_[x4 * np.log(x9 - 100) - x4*np.log(300 - x7) - x9 - x7 + 400]
        elif fidelity == 14:
            return np.c_[x4 * np.log(x9 - 100) - x4*np.log(300 - x7) - x9 - x7 + 400]
        elif fidelity == 15:
            return - np.c_[x5 * np.log(x10 - x7) - x5*np.log(400 - x8) - x10 + x7 - x8 + 400]
        elif fidelity == 16:
            return np.c_[x5 * np.log(x10 - x7) - x5*np.log(400 - x8) - x10 + x7 - x8 + 400]
        elif fidelity == 17:
            return - np.c_[x6 * np.log(x11 - x8) - x6*np.log(100) - x11 + x8 + 100]
        elif fidelity == 18:
            return np.c_[x6 * np.log(x11 - x8) - x6*np.log(100) - x11 + x8 + 100]


# class HaverlyPollingProblem(test_func):
#     """
#     https://reader.elsevier.com/reader/sd/pii/0098135494000972?token=81EACECCDD800D8B8466E259DCA6C0F55E2291AE27B770629E8C6756EC56985110366A7250CE4988A4957375798213FC&originRegion=us-east-1&originCreation=20210927073556
#     """
#     def __init__(self):
#         self.d = 10
#         self.C = 10

#         self.bounds = np.array([
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#             [300, 300, 100, 200, 100, 300, 100, 200, 200, 3]])
#         self.standard_length_scale = standard_length_scale(self.bounds)
#         self.maximum = -4.0000560000 * 1e2
#         self.g_thresholds = np.zeros(self.C) - 1e-3

#     def values(self, input, fidelity=None):
#         input = np.atleast_2d(input)
#         x1 = input[:,0]
#         x2 = input[:,1]
#         x3 = input[:,2]
#         x4 = input[:,3]
#         x5 = input[:,4]
#         x6 = input[:,5]
#         x7 = input[:,6]
#         x8 = input[:,7]
#         x9 = input[:,8]
#         x10 = input[:,9]

#         if fidelity==0 or fidelity is None:
#             return np.c_[ 9*x5 + 15*x9 - 6*x1 - 16*x2 - 10*x6]
#         elif fidelity == 1:
#             return - np.c_[ x1 + x2 - x3 - x4]
#         elif fidelity == 2:
#             return np.c_[  x1 + x2 - x3 - x4]
#         elif fidelity == 3:
#             return - np.c_[x3 - x5 + x7]
#         elif fidelity == 4:
#             return np.c_[x3 - x5 + x7]
#         elif fidelity == 5:
#             return - np.c_[x4 + x8 - x9]
#         elif fidelity == 6:
#             return np.c_[x4 + x8 - x9]
#         elif fidelity == 5:
#             return - np.c_[-x6 + x7 + x8]
#         elif fidelity == 6:
#             return np.c_[-x6 + x7 + x8]
#         elif fidelity == 7:
#             return - np.c_[-x10*(x3 + x4) + 3*x1 + x2]
#         elif fidelity == 8:
#             return np.c_[-x10*(x3 + x4) + 3*x1 + x2]
#         elif fidelity == 9:
#             return - np.c_[x10*x3 + 2*x7 - 2.5*x5]
#         elif fidelity == 10:
#             return - np.c_[x10*x4 + 2*x8 - 1.5*x9]



class OptimalOperationOfAlkylationUnit(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 7
        self.C = 14

        self.bounds = np.array([
            [1000, 0, 2000, 0, 0, 0, 0],
            [2000, 100, 4000, 100, 100, 20, 200]])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 4.5291197395*1e3
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]

        if fidelity==0 or fidelity is None:
            return np.c_[0.035*x1*x6 + 1.715*x1 + 10.0*x2 + 4.0565*x3 - 0.063*x3*x5]
        elif fidelity == 1:
            return - np.c_[ 0.0059553571*x6**2*x1 + 0.88392857*x3 - 0.1175625*x6*x1 - x1]
        elif fidelity == 2:
            return - np.c_[ 1.1088*x1 + 0.1303533*x1*x6 - 0.0066033*x1*x6**2 - x3]
        elif fidelity == 3:
            return - np.c_[6.66173269*x6**2 - 56.596669*x4 + 172.39878*x5 - 10000 - 191.20592*x6]
        elif fidelity == 4:
            return - np.c_[ 1.08702*x6 - 0.03762*x6**2 + 0.32175*x4 + 56.85075 - x5]
        elif fidelity == 5:
            return - np.c_[ 0.006198*x7*x4*x3 + 2462.3121*x2 - 25.125634*x2*x4 - x3*x4]
        elif fidelity == 6:
            return - np.c_[161.18996*x3*x4 + 5000.0*x2*x4 - 489510.0*x2 - x3*x4*x7]
        elif fidelity == 7:
            return - np.c_[0.33*x7 + 44.333333 - x5]
        elif fidelity == 8:
            return - np.c_[ 0.022556*x5 - 1.0 - 0.007595*x7]
        elif fidelity == 9:
            return - np.c_[ 0.00061*x3 - 1.0 - 0.0005*x1]
        elif fidelity == 10:
            return - np.c_[0.819672*x1 - x3 + 0.819672]
        elif fidelity == 11:
            return - np.c_[24500.0*x2 - 250.0*x2*x4 - x3*x4]
        elif fidelity == 12:
            return - np.c_[1020.4082*x4*x2 + 1.2244898*x3*x4 - 100000*x2]
        elif fidelity == 13:
            return - np.c_[6.25*x1*x6 + 6.25*x1 - 7.625*x3 - 100000]
        elif fidelity == 14:
            return - np.c_[1.22*x3 - x6*x1 - x1 +1.0]


class ReactorNetworkDesign(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 6
        self.C = 9

        self.bounds = np.array([
            [0, 0, 0, 0, 1e-5, 1e-5],
            [1, 1, 1, 1, 16, 16]])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = -3.8826043623*1e-1
        self.g_thresholds = np.zeros(self.C) - 1e-3

    def values(self, input, fidelity=None):
        k1 = 0.09755988
        k2 = 0.99*k1
        k3 = 0.0391908
        k4 = 0.9*k3

        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]

        if fidelity==0 or fidelity is None:
            return np.c_[x4]
        elif fidelity == 1:
            return - np.c_[ k1*x5*x2 +x1 - 1 ]
        elif fidelity == 2:
            return np.c_[ k1*x5*x2 +x1 - 1 ]
        elif fidelity == 3:
            return - np.c_[k3*x5*x3 +x3 +x1 - 1]
        elif fidelity == 4:
            return np.c_[k3*x5*x3 +x3 +x1 - 1]
        elif fidelity == 5:
            return - np.c_[k2*x6*x2 - x1 +x2]
        elif fidelity == 6:
            return np.c_[k2*x6*x2 - x1 +x2]
        elif fidelity == 7:
            return - np.c_[k4*x6*x4 +x2 - x1 +x4 - x3]
        elif fidelity == 8:
            return np.c_[k4*x6*x4 +x2 - x1 +x4 - x3]
        elif fidelity == 9:
            return - np.c_[ x5**0.5 + x6**0.5 - 4]



class WeightedMinimizationOfSpeedReducer(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 7
        self.C = 11

        self.bounds = np.array([
            [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5],
            [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 2.9944244658 * 1e3
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]

        if fidelity==0 or fidelity is None:
            return - np.c_[ 0.7854*x2**2*x1*(14.9334*x3 - 43.0934 + 3.3333*x3**2) + 0.7854*(x5*x7**2 + x4*x6**2) - 1.508*x1*(x7**2 + x6**2) + 7.477*(x7**3 + x6**3) ]
        elif fidelity == 1:
            return - np.c_[ -x1*x2**2*x3 + 27]
        elif fidelity == 2:
            return - np.c_[ -x1*x2**2*x3**2 + 397.5]
        elif fidelity == 3:
            return - np.c_[-x2*x6**4*x3*x4**-3 + 1.93]
        elif fidelity == 4:
            return - np.c_[ -x2*x7**4*x3*x5**-3 + 1.93]
        elif fidelity == 5:
            return - np.c_[10*x6**-3 * np.sqrt(16.91*10**6 + (745*x4*x2**-1*x3**-1)**2) - 1100]
        elif fidelity == 6:
            return - np.c_[10*x7**-3 * np.sqrt(157.5*10**6 + (745*x5*x2**-1*x3**-1)**2) - 850]
        elif fidelity == 7:
            return - np.c_[x2*x3 - 40]
        elif fidelity == 8:
            return - np.c_[-x1*x2**-1 + 5]
        elif fidelity == 9:
            return - np.c_[ x1*x2**-1 - 12]
        elif fidelity == 10:
            return - np.c_[ 1.5*x6 - x4 + 1.9]
        elif fidelity == 11:
            return - np.c_[ 1.1*x7 - x5 + 1.9]



class IndustrialRefrigerationSystem(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 14
        self.C = 15

        self.bounds = np.array([
            0.001*np.ones(self.d),
            5*np.ones(self.d)])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 3.2213000814 * 1e-2
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]
        x6 = input[:,5]
        x7 = input[:,6]
        x8 = input[:,7]
        x9 = input[:,8]
        x10 = input[:,9]
        x11 = input[:,10]
        x12 = input[:,11]
        x13 = input[:,12]
        x14 = input[:,13]

        if fidelity==0 or fidelity is None:
            return - np.c_[ 63098.88*x2*x4*x12 + 5441.5*x2**2*x12 + 115055.5*x2**1.664*x6 + 6172.27*x2**2 *x6 + 63098.88**x1*x3*x11 +5441.5*x2**2*x11 +115055.5*x1**1.664*x5 +6172.27*x1**2*x5 +140.53*x1*x11 + 281.29*x3*x11 + 70.26*x1**2 + 281.29*x1*x3 + 281.29*x3**2 +14437*x8**1.8812*x12**0.3424*x10*(x14**-1) * x1**2 * x7 * x9**-1 +20470.2*x7**2.893*x11**0.316*x1**2 ]
        elif fidelity == 1:
            return - np.c_[ 1.524*x7**-1 - 1]
        elif fidelity == 2:
            return - np.c_[ 1.524*x8**-1 - 1]
        elif fidelity == 3:
            return - np.c_[ 0.07789*x1*x1 - 2*x7**-1*x9 - 1]
        elif fidelity == 4:
            return - np.c_[ 7.05305*x9**-1*x1**2*x10*x8**-1*x2**-1*x14**-1 - 1]
        elif fidelity == 5:
            return - np.c_[ 0.0833*x13**-1*x14 - 1]
        elif fidelity == 6:
            return - np.c_[ 47.136*x2**0.333*x10**-1*x12 - 1.333*x8*x13**2.1195 + 62.06*x13**2.1195*x12**-1*x8**0.2*x10**-1 -1]
        elif fidelity == 7:
            return - np.c_[0.04771*x10*x8**1.8812*x12**0.3424 - 1]
        elif fidelity == 8:
            return - np.c_[0.0488*x9*x7**1.893*x11**0.316 - 1]
        elif fidelity == 9:
            return - np.c_[ 0.0099*x1*x3**-1 - 1]
        elif fidelity == 10:
            return - np.c_[ 0.0193*x2*x4**-1 - 1]
        elif fidelity == 11:
            return - np.c_[ 0.0298*x1*x5**-1 - 1]
        elif fidelity == 12:
            return - np.c_[ 0.056*x2*x6**-1 - 1]
        elif fidelity == 13:
            return - np.c_[ 2*x9**-1 - 1]
        elif fidelity == 14:
            return - np.c_[ 2*x10**-1 - 1]
        elif fidelity == 15:
            return - np.c_[ x12*x11**-1 - 1]



class Three_barTrussDesign(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 2
        self.C = 3

        self.bounds = np.array([
            1e-5*np.ones(self.d),
            1.*np.ones(self.d)])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 2.6389584338 * 1e2
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        l = 100
        P = 2
        sigma = 2
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]

        if fidelity==0 or fidelity is None:
            return - np.c_[ l*(x2 + 2*np.sqrt(2) * x1)]
        elif fidelity == 1:
            return - np.c_[ x2 / (2*x2*x1 + np.sqrt(2)*x1**2) * P - sigma]
        elif fidelity == 2:
            return - np.c_[ (x2 + np.sqrt(2)*x1) / (2*x2*x1 + np.sqrt(2)*x1**2) * P - sigma ]
        elif fidelity == 3:
            return - np.c_[ P / (x1 + np.sqrt(2)*x2) - sigma]


class MultipleDiskClutchBrakeDesign(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 5
        self.C = 8

        self.bounds = np.array([
            [60, 90, 1, 0, 2],
            [80, 110, 3, 1000, 9]])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = - 2.3524245790 * 1e-1
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]

        rho = 0.0000078
        delta_R = 20
        L_max = 30
        mu = 0.6
        V_sr_max = 10
        delta = 0.5
        s = 1.5
        T_max = 15
        n = 250
        I_z = 55
        M_s = 40
        M_f = 3
        p_max = 1

        M_h = 2 / 3 * mu*x4*x5*(x2**3 - x1**3) / (x2**2 - x1**2)
        omega = np.pi*n / 30
        A = np.pi * (x2**2 - x1**2)
        p_rz = x4 / A
        R_sr = 2/3 *(x2**3 - x1**3) / (x2**2*x1**2)
        V_sr = np.pi*R_sr*n / 30
        T = (I_z*omega) / (M_h + M_f)

        if fidelity==0 or fidelity is None:
            return - np.c_[ np.pi*(x2**2 - x1**2)*x3*(x5 + 1)*rho ]
        elif fidelity == 1:
            return - np.c_[ - p_max + p_rz]
        elif fidelity == 2:
            return - np.c_[ p_rz*V_sr - V_sr_max*p_max]
        elif fidelity == 3:
            return - np.c_[ delta_R + x1 - x2]
        elif fidelity == 4:
            return - np.c_[ -L_max + (x5+1)*(x3+delta)]
        elif fidelity == 5:
            return - np.c_[ s*M_s - M_h]
        elif fidelity == 6:
            return np.c_[ T]
        elif fidelity == 7:
            return - np.c_[-V_sr_max + V_sr]
        elif fidelity == 8:
            return - np.c_[T - T_max]

# class Hydro_staticThrustBearingDesign(test_func):
#     def __init__(self):
#         self.d = 4
#         self.C = 7

#         self.bounds = np.array([
#             [1, 1, 1e-6, 1],
#             [16, 16, 16*1e-6, 16]])
#         self.standard_length_scale = standard_length_scale(self.bounds)
#         self.maximum = - 1.6254428092 * 1e3
#         self.g_thresholds = np.zeros(self.C)

#     def values(self, input, fidelity=None):
#         input = np.atleast_2d(input)
#         R = input[:,0]
#         R0 = input[:,1]
#         mu = input[:,2]
#         Q = input[:,3]

#         # R[R == R0] += 1e-6

#         P = (np.log10(np.log10(8.122*10**6*mu + 0.8) ) +3.55) / 10.04
#         delta_T = 2*(10**P - 559.7)
#         E_f = 9336*Q*0.0307*0.5*delta_T
#         # temp = (R**4 / 4 - R0**4/4)
#         # temp[temp==0] = 1e-16
#         # h = (2*np.pi*750 / 60)**2 *(2*np.pi*mu) / E_f * temp
#         h = (2*np.pi*750 / 60)**2 *(2*np.pi*mu) / E_f *(R**4 / 4 - R0**4/4)
#         P0 = (6*mu*Q) / (np.pi*h**3) * np.log(R / R0)
#         W = (np.pi*P0) / 2 * (R**2 - R0**2) / np.log(R / R0)


#         if fidelity==0 or fidelity is None:
#             return - np.c_[ Q*P0 / 0.7 + E_f ]
#         elif fidelity == 1:
#             return - np.c_[ 1000 - P0]
#         elif fidelity == 2:
#             return - np.c_[ W - 101000]
#         elif fidelity == 3:
#             return - np.c_[ 5000 -  W / (np.pi* (R**2 - R0**2))]
#         elif fidelity == 4:
#             return - np.c_[ 50 - P0]
#         elif fidelity == 5:
#             return - np.c_[ 0.001 - 0.0307/(386.4*P0) * Q / (2*np.pi*R*h)]
#         elif fidelity == 6:
#             return - np.c_[ R - R0 ]
#         elif fidelity == 7:
#             return - np.c_[h - 0.001]


class Himmelblau_Function(test_func):
    def __init__(self):
        self.noise_var = 0
        self.d = 5
        self.C = 6

        self.bounds = np.array([
            [78, 33, 27, 27, 27],
            [102, 45, 45, 45, 45]])
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.0665538672 * 1e4
        self.g_thresholds = np.zeros(self.C)

    def values(self, input, fidelity=None):
        input = np.atleast_2d(input)
        x1 = input[:,0]
        x2 = input[:,1]
        x3 = input[:,2]
        x4 = input[:,3]
        x5 = input[:,4]

        G1 = 85.334407 + 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5
        G2 = 80.51249 + 0.00713172*x5 + 0.0029955*x1*x2 + 0.0021813*x3**2
        G3 = 9.300961 + 0.0047026*x3*x5 + 0.00125447*x1*x3 + 0.0019085*x3*x4

        if fidelity==0 or fidelity is None:
            return - np.c_[  5.3578547*x3**2 + 0.8356891*x1*x5 + 37.293239*x1 - 40792.141 ]
        elif fidelity == 1:
            return - np.c_[ - G1]
        elif fidelity == 2:
            return - np.c_[ G1 - 92]
        elif fidelity == 3:
            return - np.c_[ 90 - G2]
        elif fidelity == 4:
            return - np.c_[ G2 - 110]
        elif fidelity == 5:
            return - np.c_[ 20 - G3]
        elif fidelity == 6:
            return - np.c_[ G3 - 25]

















class Beale(test_func):
    '''
    Beale Function: d = 2, M = 2
    Three constants is changed to make low fidelity function.
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-4.5, -4.5], [4.5, 4.5]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 0

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, cons_list=[1.2, 2.5, 2.5])
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _common_processing(self, input, cons_list=None):
        if cons_list is None:
            cons_list = [1.5, 2.25, 2.625]
        first_term = (cons_list[0] - input[:, 0] + input[:, 0]*input[:, 1])**2
        second_term = (cons_list[1] - input[:, 0] +
                       input[:, 0]*input[:, 1]**2)**2
        third_term = (cons_list[2] - input[:, 0] +
                      input[:, 0]*input[:, 1]**3)**2
        return - np.c_[(first_term + second_term + third_term)]


class HartMann3(test_func):
    '''
    HartMann 3-dimensional function: d = 3, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0], [1, 1, 1]])
        self.d = 3
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.86278

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _common_processing(self, input, alpha_error=0):
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array([[3.0, 10, 30], [0.1, 10, 35],
                      [3.0, 10, 30], [0.1, 10, 35]])
        P = np.array([[3689, 1170, 2673], [4699, 4387, 7470], [
                     1091, 8732, 5547], [381, 5743, 8828]])*1e-4

        values = 0
        for i in range(4):
            inner = 0
            for j in range(3):
                inner -= A[i, j]*(input[:, j] - P[i, j])**2
            values += alpha[i]*np.power(np.e, inner)
        return np.c_[values]


class HartMann4(test_func):
    '''
    HartMann 4-dimensional function: d = 4, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.d = 4
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.135474

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _common_processing(self, input, alpha_error=0):
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = np.array([[1312, 1696, 5569, 124, 8283, 5886],
                      [2329, 4135, 8307, 3736, 1004, 9991],
                      [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091, 381]])*1e-4

        values = 0
        for i in range(4):
            inner = 0
            for j in range(4):
                inner -= A[i, j]*(input[:, j] - P[i, j])**2
            values += alpha[i]*np.power(np.e, inner)
        return np.c_[(values - 1.1)/0.839]


class HartMann6(test_func):
    '''
    HartMann 6-dimensional function: d = 6, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
        self.d = 6
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.32237

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _common_processing(self, input, alpha_error=0):
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])

        P = np.array([[1312, 1696, 5569, 124, 8283, 5886],
                      [2329, 4135, 8307, 3736, 1004, 9991],
                      [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091, 381]])*1e-4

        values = 0
        for i in range(4):
            inner = 0
            for j in range(6):
                inner -= A[i, j]*(input[:, j] - P[i, j])**2
            values += alpha[i]*np.power(np.e, inner)
        return np.c_[values]


class Borehole(test_func):
    '''
    Borehole function: d = 8, M = 2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0.05, 100, 63070, 990, 63.1, 700, 1120, 9855], [
                               0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]])
        self.d = 8
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = -7.8198 # On Efficient Global Optimization via Universal Kriging Surrogate Models

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        numerator = 2 * np.pi * input[:, 2] * (input[:, 3] - input[:, 5])
        log_ratio = np.log(input[:, 1] / input[:, 0])
        denominator = log_ratio * (1 + (2 * input[:, 6] * input[:, 2]) / (
            log_ratio * input[:, 0]**2 * input[:, 7]) + input[:, 2] / input[:, 4])
        values = numerator / denominator
        return - np.c_[values]

    def _low_fidelity_values(self, input):
        numerator = 5 * input[:, 2] * (input[:, 3] - input[:, 5])
        log_ratio = np.log(input[:, 1] / input[:, 0])
        denominator = log_ratio * (1.5 + (2 * input[:, 6] * input[:, 2]) / (
            log_ratio * input[:, 0]**2 * input[:, 7]) + input[:, 2] / input[:, 4])
        values = numerator / denominator
        return - np.c_[values]


class Branin(test_func):
    '''
    Branin function: d = 2, M = 2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-5, 0], [10, 15]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = -0.397887

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        first_term = a * (input[:, 1] - b*input[:, 0]
                          ** 2 + c*input[:, 0] - r)**2
        second_term = s*(1 - t)*np.cos(input[:, 0])
        return - np.c_[(first_term + second_term + s)]

    def _low_fidelity_values(self, input):
        a = 1.1
        b = 5. / (4 * np.pi**2)
        c = 4 / np.pi
        r = 5
        s = 8
        t = 1 / (10 * np.pi)
        first_term = a * (input[:, 1] - b*input[:, 0]
                          ** 2 + c*input[:, 0] - r)**2
        second_term = s*(1 - t)*np.cos(input[:, 0])
        return - np.c_[(first_term + second_term + s)]


class Colville(test_func):
    '''
    Colville function: d = 4, M = 2
    low_fidelity function is
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-10, -10, -10, -10], [10, 10, 10, 10]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 0

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, cons_list=[90, 0.9, 0.9, 100, 9, 20])
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _common_processing(self, input, cons_list=None):
        if cons_list is None:
            cons_list = [100, 1, 1, 90, 10.1, 19.8]
        term1 = cons_list[0] * (input[:, 0]**2 - input[:, 1])**2
        term2 = cons_list[1] * (input[:, 0] - 1)**2
        term3 = cons_list[2] * (input[:, 2] - 1)**2
        term4 = cons_list[3] * (input[:, 2]**2 - input[:, 3])**2
        term5 = cons_list[4] * ((input[:, 1] - 1)**2 + (input[:, 3] - 1)**2)
        term6 = cons_list[5] * (input[:, 1]-1)*(input[:, 3]-1)
        return - np.c_[(term1 + term2 + term3 + term4 + term5 + term6)]


class CurrinExp(test_func):
    '''
    Currin exponential function : d=2, M=2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0], [1, 1]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = None

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        FormerTerm = 1 - np.exp(-1/(2.*input[:, 1]))
        LatterTerm = (2300*input[:, 0]**3 + 1900*input[:, 0]**2 + 2092*input[:, 0] + 60) / (
            100*input[:, 0]**3 + 500*input[:, 0]**2 + 4*input[:, 0] + 20)
        values = FormerTerm * LatterTerm
        return np.c_[values]

    def _low_fidelity_values(self, input):
        input1 = np.copy(input) + 0.05

        input2 = np.copy(input)
        input2[:, 0] = input2[:, 0] + 0.05
        input2[:, 1] = input2[:, 1] - 0.05
        input2[:, 1][input2[:, 1] < 0] = 0

        input3 = np.copy(input)
        input3[:, 0] = input3[:, 0] - 0.05
        input3[:, 1] = input3[:, 1] + 0.05

        input4 = np.copy(input)
        input4[:, 0] = input4[:, 0] - 0.05
        input4[:, 1] = input4[:, 1] - 0.05
        input4[:, 1][input4[:, 1] < 0] = 0

        values = (self._high_fidelity_values(input1) + self._high_fidelity_values(input2) +
                  self._high_fidelity_values(input3) + self._high_fidelity_values(input4))/4.
        return np.c_[values]


class Forrester(test_func):
    '''
    Forrester function : d=1, M=2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0], [1]])
        self.d = 1
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        return - np.c_[((6*input - 2)**2 * np.sin(12*input - 4))]

    def _low_fidelity_values(self, input):
        A = 0.5
        B = 10
        C = -5
        values = self._high_fidelity_values(input)
        return np.c_[A*values - B*(input - 0.5) + C]


class Styblinski_tang(test_func):
    '''
    Styblinski-tang function : d=2, M=2
    I fix the dimension.
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-5, -5], [5, 5]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        d = 2  # np.size(input, 1)
        values = 0
        for i in range(d):
            values += input[:, i]**4 - 16*input[:, i]**2 + 5*input[:, i]
        return - np.c_[values/2]

    def _low_fidelity_values(self, input):
        d = 2  # np.size(input, 1)
        values = 0
        for i in range(d):
            values += 0.9*input[:, i]**4 - 15*input[:, i]**2 + 6*input[:, i]
        return - np.c_[values/2]


class Park1(test_func):
    '''
    Park function 1 : d=4, M=2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        FirstTerm = input[:, 0]/2. * \
            (np.sqrt(1+(input[:, 1] + input[:, 2]**2)
                     * input[:, 3]/input[:, 0]**2) - 1)
        SecondTerm = (input[:, 1] + 3*input[:, 3]) * \
            np.power(np.e, 1+np.sin(input[:, 2]))
        values = FirstTerm + SecondTerm
        return - np.c_[values]

    def _low_fidelity_values(self, input):
        values = (1 + np.sin(input[:, 0])/10.)*self._high_fidelity_values(
            input).ravel() - (-2*input[:, 0] + input[:, 1]**2 + input[:, 2]**2 + 0.5)
        return np.c_[values]


class Park2(test_func):
    '''
    Park function 2 : d=4, M=2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        values = 2./3.*np.power(np.e, input[:, 0]+input[:, 1]) - \
            input[:, 3]*np.sin(input[:, 2]) + input[:, 2]
        return - np.c_[values]

    def _low_fidelity_values(self, input):
        values = 1.2*self._high_fidelity_values(input) + 1
        return values


class Powell(test_func):
    '''
    Powell function : d=4, M=2
    I fix the dimension.
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-4, -4, -4, -4], [5, 5, 5, 5]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _high_fidelity_values(self, input):
        term1 = (input[:, 0] - 10*input[:, 1])**2
        term2 = 5*(input[:, 2] - input[:, 3])**2
        term3 = (input[:, 1] - 2*input[:, 2])**4
        term4 = 10 * (input[:, 0]**2 - input[:, 3])**4
        return - np.c_[term1 + term2 + term3 + term4]

    def _low_fidelity_values(self, input):
        term1 = 0.9*(input[:, 0] - 10*input[:, 1])**2
        term2 = 4*(input[:, 2] - input[:, 3])**2
        term3 = 0.9*(input[:, 1] - 2*input[:, 2])**4
        term4 = 9 * (input[:, 0]**2 - input[:, 3])**4
        return - np.c_[term1 + term2 + term3 + term4]


class Shekel(test_func):
    '''
    Shekel function : d=4, M=2
    '''

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [10, 10, 10, 10]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)

    def values(self, input, fidelity=None):
        '''
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        '''
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, m=5)
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print('Not implemented fidelity')
            exit(1)

    def _common_processing(self, input, m=10):
        beta = np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])/10.
        C = np.array([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                      [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],
                      [4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                      [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6], ])
        values = 0
        for i in range(m):
            inner = 0
            for j in range(4):
                inner += (input[:, j] - C[j, i])**2
            inner += beta[i]
            values += 1/inner
        return np.c_[values]


