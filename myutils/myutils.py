import numpy as np
import matplotlib
from math import factorial
import traceback

import matplotlib.pyplot as plt
import time
import copy
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.stats import mvn
from scipy.stats import norm
from scipydirect import minimize as directminimize
import GPy


OPTIMIZE_SEPARATOR = 5

class RFM_RBF():
    """
    rbf(gaussian) kernel of GPy k(x, y) = variance * exp(- 0.5 * ||x - y||_2^2 / lengthscale**2)
    """
    def __init__(self, lengthscales, input_dim, variance=1, basis_dim=1000):
        self.basis_dim = basis_dim
        self.std = np.sqrt(variance)
        self.random_weights = (1 / np.atleast_2d(lengthscales)) * \
            np.random.normal(size=(basis_dim, input_dim))
        self.random_offset = np.random.uniform(0, 2 * np.pi, size=basis_dim)

    def transform(self, X):
        X = np.atleast_2d(X)
        X_transform = X.dot(self.random_weights.T) + self.random_offset
        X_transform = self.std * np.sqrt(2 / self.basis_dim) * np.cos(X_transform)
        return X_transform

    def transform_grad(self, X):
        X = np.atleast_2d(X)
        X_transform_grad = X.dot(self.random_weights.T) + self.random_offset
        X_transform_grad = - self.std * np.sqrt(2 / self.basis_dim) * np.sin(X_transform_grad) * self.random_weights.T
        return X_transform_grad

class RFM_Linear_RBF():
    """
    rbf(gaussian) kernel of GPy k(x, y) = variance * exp(- 0.5 * ||x - y||_2^2 / lengthscale**2)
    """
    def __init__(self, lengthscales, input_dim, RBF_variance=1, Linear_variance=1, basis_dim_for_RBF=1000):
        self.basis_dim = basis_dim_for_RBF + input_dim + 1
        self.basis_dim_for_RBF = basis_dim_for_RBF
        self.RBF_std = np.sqrt(RBF_variance)
        self.Linear_std = np.sqrt(Linear_variance)
        self.random_weights = 1 / np.atleast_2d(lengthscales) * np.random.normal(size=(basis_dim_for_RBF, input_dim))
        self.random_offset = np.random.uniform(0, 2 * np.pi, size=basis_dim_for_RBF)

    def transform(self, X):
        X = np.atleast_2d(X)
        X = np.c_[X, np.c_[np.ones(np.shape(X)[0])]]
        X_transform = X[:,:-1].dot(self.random_weights.T) + self.random_offset
        X_transform = self.RBF_std * np.sqrt(2 / self.basis_dim_for_RBF) * np.cos(X_transform)
        X_transform = np.c_[X_transform, self.Linear_std*X]
        return X_transform

    def transform_grad(self, X):
        X = np.atleast_2d(X)
        X_grad = np.c_[np.eye(np.shape(X)[1]), np.c_[np.zeros(np.shape(X)[1])]]
        X_transform_grad = X.dot(self.random_weights.T) + self.random_offset
        X_transform_grad = - self.RBF_std * np.sqrt(2 / self.basis_dim_for_RBF) * np.sin(X_transform_grad) * self.random_weights.T
        X_transform_grad = np.c_[X_transform_grad, self.Linear_std*X_grad]
        return X_transform_grad



def minimize(obj_func, bounds_list, inputs_dim):
    res = directminimize(obj_func, bounds=bounds_list, algmethod=1, maxf=inputs_dim*2500)
    return res


def initial_design(fir_num, input_dim, bounds):
    if np.size(fir_num) == 1:
        initial_points = lhs(input_dim, samples=fir_num, criterion='maximin') * (bounds[1]- bounds[0]) + bounds[0]
    elif np.size(fir_num) == 2:
        low_initial_points = lhs(input_dim, samples=fir_num[0], criterion='maximin')  * (bounds[1]- bounds[0]) + bounds[0]
        high_initial_points = lhs_subset_greedy(low_initial_points, fir_num[1])
        initial_points = [low_initial_points, high_initial_points]
    elif np.size(fir_num) == 3:
        low_initial_points = lhs(input_dim, samples=fir_num[0], criterion='maximin')  * (bounds[1]- bounds[0]) + bounds[0]
        middle_initial_points = lhs_subset_greedy(low_initial_points, fir_num[1])
        high_initial_points = lhs_subset_greedy(middle_initial_points, fir_num[2])
        initial_points = [low_initial_points, middle_initial_points, high_initial_points]
    else:
        print('initial design for M >= 4 is not implemented')
        exit(1)
    return initial_points


def initial_design_pool_random(fir_num, input_dim, bounds, pool_X):
    if np.size(fir_num) == 1:
        initial_points, remain_X = random_maximin(fir_num, pool_X)
    elif np.size(fir_num) >= 2:
        initial_points, remain_X = random_maximin(fir_num[0], pool_X[0])
        initial_points = [initial_points for _ in range(np.size(fir_num))]
        remain_X = [remain_X for _ in range(np.size(fir_num))]
    return initial_points, remain_X

def random_maximin(fir_num, pool_X, num_sampling=10):
    maxdist = -np.inf

    for _ in range(num_sampling):
        if fir_num == np.shape(pool_X)[0]:
            sample_index = np.arange(fir_num)
            break
        random_index = np.random.choice(np.shape(pool_X)[0], fir_num, replace=False)
        d = _pdist(pool_X[random_index])

        if maxdist<np.min(d):
            maxdist = np.min(d)
            sample_index = random_index.copy()
    initial_points = pool_X[sample_index]
    remain_X = np.delete(pool_X, sample_index, axis=0)
    return initial_points, remain_X


def initial_design_pool(fir_num, input_dim, bounds, pool_X):
    if np.size(fir_num) == 1:
        initial_points = lhs(input_dim, samples=fir_num, criterion='maximin') * (bounds[1]- bounds[0]) + bounds[0]
        initial_points, pool_X = nearest_pool_input(initial_points, pool_X)
    elif np.size(fir_num) == 2:
        low_initial_points = lhs(input_dim, samples=fir_num[0], criterion='maximin')  * (bounds[1]- bounds[0]) + bounds[0]
        low_initial_points, pool_X[0] = nearest_pool_input(low_initial_points, pool_X[0])
        high_initial_points = lhs_subset_greedy(low_initial_points, fir_num[1])
        high_initial_points, pool_X[1] = nearest_pool_input(high_initial_points, pool_X[1])
        initial_points = [low_initial_points, high_initial_points]
    elif np.size(fir_num) == 3:

        low_initial_points = lhs(input_dim, samples=fir_num[0], criterion='maximin')  * (bounds[1]- bounds[0]) + bounds[0]
        low_initial_points, pool_X[0] = nearest_pool_input(low_initial_points, pool_X[0])

        middle_initial_points = lhs_subset_greedy(low_initial_points, fir_num[1])
        middle_initial_points, pool_X[1] = nearest_pool_input(middle_initial_points, pool_X[1])

        high_initial_points = lhs_subset_greedy(middle_initial_points, fir_num[2])
        high_initial_points, pool_X[2] = nearest_pool_input(high_initial_points, pool_X[2])
        initial_points = [low_initial_points, middle_initial_points, high_initial_points]
    else:
        print('initial design for M >= 4 is not implemented')
        exit(1)
    return initial_points, pool_X

def nearest_pool_input(selected_points, X):
    nearest_index = np.argmin(_two_inputs_dist(X, selected_points), axis=0)
    initial_points = X[nearest_index]
    remain_X = np.delete(X, nearest_index, axis=0)
    return initial_points, remain_X

def lhs_subset_greedy(lhs_points, num_samples, selected_points=None, criterion='maxmin'):
    if selected_points is None:
        random_index = np.random.randint(0, np.shape(lhs_points)[0], 1)
        selected_points = np.atleast_2d(lhs_points[random_index])
        remain_points = np.delete(lhs_points, random_index, axis=0)
    else:
        deplicate_index = [lhs_points.list().index(point) for point in selected_points.list()]
        remain_points = np.delete(lhs_points, deplicate_index, axis=0)

    for _ in range(num_samples-1):
        d = _two_inputs_dist(candidates=remain_points, selected_points=selected_points)
        append_index = np.argmax(np.min(d, axis=1))
        selected_points = np.r_[selected_points, np.atleast_2d(remain_points[append_index])]
        remain_points = np.delete(remain_points, append_index, axis=0)
    return selected_points

def _two_inputs_dist(candidates, selected_points):
    n, _ = candidates.shape
    m, _ = selected_points.shape
    norm_cand = np.dot(np.c_[np.sum(candidates ** 2, axis=1)], np.ones([1, m]))
    norm_sele = np.dot(np.ones([n, 1]), np.atleast_2d(np.sum(selected_points ** 2, axis=1)))
    norms = norm_cand + norm_sele - 2*candidates.dot(selected_points.T)
    return np.sqrt(norms)

def lhs(n, samples=None, criterion=None, iterations=None):
    H = None

    if samples is None:
        samples = n

    if criterion is not None:

        assert criterion.lower() in ('maximin', 'm'), 'Invalid value for "criterion": {}'.format(criterion)
    else:
        H = _lhsclassic(n, samples)

    if criterion is None:
        criterion = 'maximin'

    if iterations is None:
        iterations = 5

    if H is None:
        if criterion.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, samples, iterations, 'maximin')

    return H

def _lhsmaximin(n, samples, iterations, lhstype):
    maxdist = 0


    for _ in range(iterations):
        Hcandidate = _lhsclassic(n, samples)

        d = _pdist(Hcandidate)
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()

    return H

def _lhsclassic(n, samples):

    cut = np.linspace(0, 1, samples + 1)


    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a


    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    return H

def _pdist(x):
    x = np.atleast_2d(x)
    assert len(x.shape)==2, 'Input array must be 2d-dimensional'

    m, _ = x.shape
    if m<2:
        return []

    norm = np.matlib.repmat(np.c_[np.sum(x ** 2, axis=1)], 1, m)
    index = np.triu_indices(m, 1)
    d = norm[index] + norm.T[index] - 2*x.dot(x.T)[index]
    return d

def transform_copula(Y):
    Y_size = Y.shape[0]

    delta_size = 1. / Y_size
    Y_empirical_dist = np.zeros(Y_size)
    Y_empirical_dist[np.argsort(Y.ravel())] = np.arange(0, Y_size) / Y_size


    Y_empirical_dist[Y_empirical_dist < delta_size] = delta_size
    Y_empirical_dist[Y_empirical_dist > 1 - delta_size] = 1 - delta_size

    Y_copula = np.c_[norm.ppf(Y_empirical_dist)]
    return Y_copula

def transform_bilog(Y):
    Y_bilog = np.log(1 + np.abs(Y))
    Y_bilog *= np.sign(Y)
    return Y_bilog


class GPy_correlated_model(GPy.models.GPRegression):
    def __init__(self, X_list, Y_list, kernel_name='linear+rbf', noise_var=1e-6, normalizer=True):
        self.output_dim = len(X_list)
        self.input_dim = np.shape(X_list[0])[1]
        self.kernel_name = kernel_name
        self.my_normalizer = normalizer

        eval_num = [np.shape(X)[0] for X in X_list]
        output_indexes = np.hstack([i*np.ones(eval_num[i]) for i in range(len(eval_num))])
        X = np.c_[np.vstack([X for X in X_list if np.size(X) > 0]), output_indexes]
        Y = np.vstack([Y for Y in Y_list if np.size(Y) > 0])

        if self.kernel_name=='linear+rbf':

            X = np.c_[np.c_[X[:,:-1]], np.c_[np.ones(np.shape(X)[0])], np.c_[X[:,-1]]]
            K1 = GPy.kern.RBF(input_dim=self.input_dim, variance=1, ARD=True, active_dims=np.arange(self.input_dim).tolist()) + GPy.kern.Linear(input_dim=self.input_dim+1)
            K2 = GPy.kern.RBF(input_dim=self.input_dim, variance=1, ARD=True, active_dims=np.arange(self.input_dim).tolist()) + GPy.kern.Linear(input_dim=self.input_dim+1)
        elif self.kernel_name=='rbf':
            K1 = GPy.kern.RBF(input_dim=self.input_dim, variance=1, ARD=True)
            K2 = GPy.kern.RBF(input_dim=self.input_dim, variance=1, ARD=True)
        else:
            print('this kernel is not implemented', kernel_name)
            exit(1)

        B1 = GPy.kern.Coregionalize(input_dim=1, output_dim=self.output_dim, rank=1)
        B2 = GPy.kern.Coregionalize(input_dim=1, output_dim=self.output_dim, rank=1)
        kernel = K1**B1 + K2**B2

        self.std = [1.]
        self.mean = [0.]

        if normalizer:
            self.stds = np.array([np.std(Y) for Y in Y_list])
            self.means = np.array([np.mean(Y) for Y in Y_list])
        else:
            self.stds = np.ones(self.output_dim)
            self.means = np.zeros(self.output_dim)

        super().__init__(X=X, Y=(Y - np.c_[self.means[X[:,-1].astype(int)]]) / np.c_[self.stds[X[:,-1].astype(int)]], kernel=kernel, noise_var=noise_var, normalizer=False)

    def my_optimize(self, num_restarts=10):
        super().optimize()
        super().optimize_restarts(num_restarts=num_restarts)

    def add_XY(self, new_X_list, new_Y_list):
        original_scale_Y = self.Y * np.c_[self.stds[self.X[:,-1].astype(int)]] + np.c_[self.means[self.X[:,-1].astype(int)]]
        Y_list = [np.r_[original_scale_Y[self.X[:,-1]==i], new_Y_list[i]] if np.size(new_Y_list[i]) > 0 else original_scale_Y[self.X[:,-1]==i] for i in range(len(new_Y_list))]
        Y = np.vstack(Y_list)

        if self.kernel_name=='rbf':
            new_X_list = [np.c_[np.atleast_2d(new_X_list[i]), i*np.c_[np.ones(np.shape(new_X_list[i])[0])]] if np.size(new_X_list[i]) > 0 else [] for i in range(len(new_X_list))]
        elif self.kernel_name=='linear+rbf':

            new_X_list = [np.c_[np.atleast_2d(new_X_list[i]), np.c_[np.ones(np.shape(new_X_list[i])[0])], i*np.c_[np.ones(np.shape(new_X_list[i])[0])]] if np.size(new_X_list[i]) > 0 else [] for i in range(len(new_X_list))]
        X_list = [np.r_[self.X[self.X[:,-1]==i], new_X_list[i]] if np.size(new_X_list[i]) > 0 else self.X[self.X[:,-1]==i] for i in range(len(new_X_list))]
        X = np.vstack(X_list)

        if self.my_normalizer:
            self.stds = np.array([np.std(Y) for Y in Y_list])
            self.means = np.array([np.mean(Y) for Y in Y_list])

        self.set_XY(X, (Y  - np.c_[self.means[X[:,-1].astype(int)]]) / np.c_[self.stds[X[:,-1].astype(int)]])

    def set_hyperparameters_bounds(self, kernel_bounds, noise_var=1e-6):
        self['.*Gaussian_noise.variance'].constrain_fixed(noise_var)

        if self.kernel_name=='linear+rbf':
            self['.*mul.sum.rbf.variance'].constrain_bounded(0, 1.)
            self['.*mul.sum.linear.variances'].constrain_bounded(0, 1.)
            if self.kern.mul.sum.rbf.ARD:
                for i in range(self.input_dim - 2):
                    self['.*mul.sum.rbf.lengthscale'][[i]].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
            else:
                self['.*mul.sum.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
            self['.*mul.coregion.W'].constrain_bounded(-1, 1)
            self['.*mul.coregion.kappa'].constrain_bounded(1e-8, 1)

            self['.*mul_1.sum.rbf.variance'].constrain_bounded(0, 1.)
            self['.*mul_1.sum.linear.variances'].constrain_bounded(0, 1.)
            if self.kern.mul_1.sum.rbf.ARD:
                for i in range(self.input_dim - 2):
                    self['.*mul_1.sum.rbf.lengthscale'][[i]
                                                    ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
            else:
                self['.*mul_1.sum.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
            self['.*mul_1.coregion.W'].constrain_bounded(-1, 1)
            self['.*mul_1.coregion.kappa'].constrain_bounded(1e-8, 1)
        elif self.kernel_name=='rbf':
            self['.*mul.rbf.variance'].constrain_fixed(1.)
            if self.kern.mul.rbf.ARD:
                for i in range(self.input_dim - 1):
                    self['.*mul.rbf.lengthscale'][[i]].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
            else:
                self['.*mul.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
            self['.*mul.coregion.W'].constrain_bounded(-1, 1)
            self['.*mul.coregion.kappa'].constrain_bounded(1e-8, 1)

            self['.*mul_1.rbf.variance'].constrain_fixed(1.)
            if self.kern.mul.rbf.ARD:
                for i in range(self.input_dim - 1):
                    self['.*mul_1.rbf.lengthscale'][[i]].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
            else:
                self['.*mul_1.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
            self['.*mul_1.coregion.W'].constrain_bounded(-1, 1)
            self['.*mul_1.coregion.kappa'].constrain_bounded(1e-8, 1)


    def predict(self, x, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        x = np.atleast_2d(x)
        if self.kernel_name == 'linear+rbf':
            x = np.c_[np.c_[x[:,:-1]], np.c_[np.ones(np.shape(x)[0])], np.c_[x[:,-1]]]
        mean, var = super().predict(x, full_cov=full_cov, Y_metadata=Y_metadata, kern=kern, likelihood=likelihood, include_likelihood=include_likelihood)

        mean = mean * np.c_[self.stds[x[:,-1].astype(int)]] + np.c_[self.means[x[:,-1].astype(int)]]
        if full_cov == False:
            var = var * np.c_[self.stds[x[:,-1].astype(int)]]**2
        else:
            var = np.c_[self.stds[x[:,-1].astype(int)]] * np.c_[self.stds[x[:,-1].astype(int)]].T * var
        return mean, var


    def predictive_gradients(self, x):
        x = np.atleast_2d(x)
        if self.kernel_name == 'linear+rbf':
            x = np.c_[np.c_[x[:,:-1]], np.c_[np.ones(np.shape(x)[0])], np.c_[x[:,-1]]]
        mean_grad, var_grad = super().predictive_gradients(x)

        mean_grad = mean_grad * np.c_[self.stds[x[:,-1].astype(int)]]
        var_grad = var_grad * np.c_[self.stds[x[:,-1].astype(int)]]**2

        if self.kernel_name == 'linear+rbf':
            return mean_grad[:,:-2], var_grad[:,:-2]
        if self.kernel_name == 'rbf':
            return mean_grad[:,:-1], var_grad[:,:-1]

    def posterior_covariance_between_points(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if self.kernel_name == 'linear+rbf':
            X1 = np.c_[np.c_[X1[:,:-1]], np.c_[np.ones(np.shape(X1)[0])], np.c_[X1[:,-1]]]
            X2 = np.c_[np.c_[X2[:,:-1]], np.c_[np.ones(np.shape(X2)[0])], np.c_[X2[:,-1]]]
        return (np.c_[self.stds[X1[:,:-1].astype(int)]] * np.c_[self.stds[X2[:,:-1].astype(int)]].T) * super().posterior_covariance_between_points(X1, X2)


class GPy_independent_model(object):
    def __init__(self, X_list, Y_list, kernel_name='linear+rbf', noise_var=1e-6, normalizer=True, ARD=True):
        self.output_dim = len(X_list)
        self.input_dim = np.shape(X_list[0])[1]
        self.kernel_name = kernel_name
        self.my_normalizer = normalizer


        if kernel_name=='linear+rbf':
            X_list = [ np.c_[X, np.c_[np.ones(np.shape(X)[0])]] for X in X_list ]

        if normalizer:
            self.stds = [np.std(Y) for Y in Y_list]
            self.means = [np.mean(Y) for Y in Y_list]
        else:
            self.stds = np.ones(self.output_dim)
            self.means = np.zeros(self.output_dim)

        self.model_list = list()
        for i in range(self.output_dim):
            if self.kernel_name=='linear+rbf':
                kernel = GPy.kern.RBF(input_dim=self.input_dim, variance=1, ARD=ARD, active_dims=np.arange(self.input_dim).tolist()) + GPy.kern.Linear(input_dim=self.input_dim+1)
            elif self.kernel_name=='rbf':
                kernel = GPy.kern.RBF(input_dim=self.input_dim, variance=1, ARD=ARD)
            else:
                print('this kernel is not implemented', kernel)
                exit(1)

            self.model_list.append(GPy.models.GPRegression(X_list[i], (Y_list[i] - self.means[i]) / self.stds[i], kernel=kernel, noise_var=noise_var, normalizer=False))
            self.model_list[i].std = [1.]
            self.model_list[i].mean = [0.]


    def add_XY(self, X_list, Y_list):
        for i in range(self.output_dim):
            X = X_list[i]
            if np.size(X) > 0:
                if self.kernel_name=='linear+rbf':
                    X = np.c_[X, np.ones(np.shape(X)[0])]

                new_X = np.r_[self.model_list[i].X, X]
                new_Y = np.r_[self.model_list[i].Y * self.stds[i] + self.means[i], np.c_[Y_list[i]]]

                if self.my_normalizer:
                    self.stds[i] = np.std(new_Y)
                    self.means[i] = np.mean(new_Y)
                self.model_list[i].set_XY(new_X, (new_Y - self.means[i]) / self.stds[i])

    def my_optimize(self, num_restarts=10):
        for i in range(self.output_dim):
            self.model_list[i].optimize()
            self.model_list[i].optimize_restarts(num_restarts=num_restarts)

    def set_hyperparameters_bounds(self, kernel_bounds, noise_var=1e-6):
        if self.kernel_name == 'linear+rbf':
            for i in range(self.output_dim):
                self.model_list[i]['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
                self.model_list[i]['.*sum.rbf.variance'].constrain_bounded(0, 1.)
                self.model_list[i]['.*sum.linear.variances'].constrain_bounded(0, 1.)
                if self.model_list[i].kern.rbf.ARD:
                    for j in range(self.model_list[i].input_dim-1):
                        self.model_list[i]['.*sum.rbf.lengthscale'][[j]].constrain_bounded(kernel_bounds[0, j], kernel_bounds[1, j])
                else:
                    self.model_list[i]['.*sum.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
        if self.kernel_name == 'rbf':
            for i in range(self.output_dim):
                self.model_list[i]['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
                self.model_list[i]['.*rbf.variance'].constrain_fixed(1.)
                if self.model_list[i].kern.rbf.ARD:
                    for j in range(self.model_list[i].input_dim):
                        self.model_list[i]['.*rbf.lengthscale'][[j]].constrain_bounded(kernel_bounds[0, j], kernel_bounds[1, j])
                else:
                    self.model_list[i]['.*rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])



    def predict(self, x, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        x = np.atleast_2d(x)
        if self.kernel_name == 'linear+rbf':
            x = np.c_[np.c_[x[:,:-1]], np.c_[np.ones(np.shape(x)[0])], np.c_[x[:,-1]]]

        mean = list()
        if full_cov:
            cov = None
        else:
            var = list()

        if np.all( (x[:-1,-1] - x[1:, -1]) > 0 ) and np.shape(x)[0] > 1:

            print(x[:,-1])
            print('input x is not sorted w.r.t. output dim in predict')
            exit(1)
        else:
            for i in range(self.output_dim):
                x_temp = x[x[:,-1]==i, :-1]

                if np.size(x_temp) > 0:
                    mean_temp, var_temp = self.model_list[i].predict(x_temp, full_cov=full_cov)
                    mean.append(mean_temp.ravel() * self.stds[i] + self.means[i])
                    var_temp = var_temp * self.stds[i]**2
                    if full_cov:
                        if cov is None:
                            cov = var_temp
                        else:

                            cov = np.c_[np.r_[cov, np.zeros((np.shape(var_temp)[0],np.shape(cov)[1]))], np.r_[np.zeros((np.shape(cov)[0],np.shape(var_temp)[1])), var_temp]]
                    else:
                        var.append(var_temp.ravel())

        mean = np.c_[np.array(mean).ravel()]
        if full_cov:
            return mean, cov
        else:
            var = np.c_[np.array(var).ravel()]
            return mean, var

    def predict_noiseless(self, x, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        x = np.atleast_2d(x)
        if self.kernel_name == 'linear+rbf':
            x = np.c_[np.c_[x[:,:-1]], np.c_[np.ones(np.shape(x)[0])], np.c_[x[:,-1]]]

        mean = list()
        if full_cov:
            cov = None
        else:
            var = list()

        if np.all( (x[:-1,-1] - x[1:, -1]) > 0 ) and np.shape(x)[0] > 1:

            print('input x is not sorted w.r.t. output dim in predict')
            exit(1)
        else:
            for i in range(self.output_dim):
                x_temp = x[x[:,-1]==i, :-1]

                if np.size(x_temp) > 0:
                    mean_temp, var_temp = self.model_list[i].predict_noiseless(x_temp, full_cov=full_cov)
                    mean.append(mean_temp.ravel() * self.stds[i] + self.means[i])
                    var_temp = var_temp * self.stds[i]**2
                    if full_cov:
                        if cov is None:
                            cov = var_temp
                        else:

                            cov = np.c_[np.r_[cov, np.zeros((np.shape(var_temp)[0],np.shape(cov)[1]))], np.r_[np.zeros((np.shape(cov)[0],np.shape(var_temp)[1])), var_temp]]
                    else:
                        var.append(var_temp.ravel())

        mean = np.c_[np.array(mean).ravel()]
        if full_cov:
            return mean, cov
        else:
            var = np.c_[np.array(var).ravel()]
            return mean, var

    def predictive_gradients(self, x):
        x = np.atleast_2d(x)
        if self.kernel_name=='linear+rbf':
            x = np.c_[np.c_[x[:,:-1]], np.c_[np.ones(np.shape(x)[0])], np.c_[x[:,-1]]]

        mean_grad = list()
        var_grad = list()

        if np.all( (x[:-1,-1] - x[1:, -1]) > 0 ) and np.shape(x)[0] > 1:

            print('input x is not sorted w.r.t. output dim in gradients prediction')
            exit(1)
        else:
            for i in range(self.output_dim):
                x_temp = x[x[:,-1]==i, :-1]

                if np.size(x_temp) > 0:
                    mean_grad_temp, var_grad_temp = self.model_list[i].predictive_gradients(x_temp)
                    mean_grad.append(mean_grad_temp*self.stds[i])
                    var_grad.append(var_grad_temp*self.stds[i]**2)
            mean_grad = np.vstack(mean_grad)
            var_grad = np.vstack(var_grad)

        if self.kernel_name=='linear+rbf':
            return mean_grad[:,:-1], var_grad[:,:-1]
        else:
            return mean_grad, var_grad


    def posterior_covariance_between_points(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        if not(np.all(X1[0, -1] == X1[:,-1]) and np.all(X1[0, -1] == X2[:,-1])):
            print('only the covariance between same output dim can be computed in independent model.')
            print(X1)
            print(X2)
            exit(1)

        if self.kernel_name == 'linear+rbf':
            X1 = np.c_[np.c_[X1[:,:-1]], np.c_[np.ones(np.shape(X1)[0])], np.c_[X1[:,-1]]]
            X2 = np.c_[np.c_[X2[:,:-1]], np.c_[np.ones(np.shape(X2)[0])], np.c_[X2[:,-1]]]
        return self.stds[int(X1[0, -1])]**2 * self.model_list[int(X1[0, -1])].posterior_covariance_between_points(X1[:,:-1], X2[:,:-1])




class GPy_model_withIndicatorOne(GPy.models.GPRegression):
    def __init__(self, X, Y, kernel, noise_var=1e-6, normalizer=True):
        X = np.c_[np.atleast_2d(X), np.c_[np.ones(np.shape(X)[0])]]
        super().__init__(X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        self['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        if normalizer:
            self.std = self.normalizer.std.copy()
            self.mean = self.normalizer.mean.copy()
        else:
            self.std = [1.]
            self.mean = [0.]

    def predict(self, x, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        x = np.atleast_2d(x)
        x = np.c_[x, np.c_[np.ones(np.shape(x)[0])]]
        mean, var = super().predict(x, full_cov=full_cov, Y_metadata=Y_metadata, kern=kern, likelihood=likelihood, include_likelihood=include_likelihood)
        return mean, var

    def predictive_gradients(self, x):
        x = np.atleast_2d(x)
        x = np.c_[x, np.c_[np.ones(np.shape(x)[0])]]
        mean_grad, var_grad = super().predictive_gradients(x)
        return mean_grad[:,:-1], var_grad[:,:-1]


    def posterior_covariance_between_points(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        X1 = np.c_[X1, np.c_[np.ones(np.shape(X1)[0])]]
        X2 = np.c_[X2, np.c_[np.ones(np.shape(X2)[0])]]
        return (self.std**2) * super().posterior_covariance_between_points(X1, X2)


class GPy_model_withIntercept(GPy.models.GPRegression):
    def __init__(self, X, Y, kernel, noise_var=1e-6, normalizer=True):
        self.fit_mean_func(X, Y)
        Y_minus_prior = Y - self.intercept
        super().__init__(X=X, Y=Y_minus_prior, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        self['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        if normalizer:
            self.std = self.normalizer.std.copy()
            self.mean = self.normalizer.mean.copy()
        else:
            self.std = [1.]
            self.mean = [0.]

    def fit_mean_func(self, X, Y):
        A = np.c_[X, np.c_[np.ones(np.shape(X)[0])]]

        coefficients, _, _, _ = np.linalg.lstsq(A, Y.ravel(), rcond=None)
        self.intercept = coefficients[-1]

    def predict(self, x, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        x = np.atleast_2d(x)
        mean, var = super().predict(x, full_cov=full_cov, Y_metadata=Y_metadata, kern=kern, likelihood=likelihood, include_likelihood=include_likelihood)
        return mean+self.intercept, var

    def predictive_gradients(self, x):
        x = np.atleast_2d(x)
        mean_grad, var_grad = super().predictive_gradients(x)
        return mean_grad, var_grad


    def posterior_covariance_between_points(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return (self.std**2) * super().posterior_covariance_between_points(X1, X2)


class GPy_model(GPy.models.GPRegression):
    def __init__(self, X, Y, kernel, noise_var=1e-6, normalizer=True):
        super().__init__(X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        self['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        if normalizer:
            self.std = self.normalizer.std.copy()
            self.mean = self.normalizer.mean.copy()
        else:
            self.std = [1.]
            self.mean = [0.]

    def minus_predict(self, x):
        x = np.atleast_2d(x)
        return -1 * super().predict_noiseless(x)[0]

    def minus_predict_high(self, x):
        x = np.atleast_2d(np.r_[x, self.high_fidelity_feature])
        return -1 * super().predict_noiseless(x)[0]








    def maximum_inference(self, bounds_list, high_fidelity_feature=None):
        if high_fidelity_feature is None:
            res = minimize(self.minus_predict, bounds_list, self.input_dim)
        else:
            self.high_fidelity_feature = high_fidelity_feature
            res = minimize(self.minus_predict_high, bounds_list, (self.input_dim - np.size(high_fidelity_feature)))
        return np.atleast_2d(res['x'])


    def posterior_covariance_between_points(self, X1, X2):
        return (self.std**2) * super().posterior_covariance_between_points(X1, X2)


    def diag_covariance_between_points(self, X1, X2):
        assert np.shape(X1) == np.shape(X2), 'cannot compute diag (not square matrix)'
        Kx1 = self.kern.K(X1, self.X)
        Kx2 = self.kern.K(self.X, X2)
        K12 = self.kern.K(np.atleast_2d(X1[0,:]),np.atleast_2d(X2[0,:])) * np.c_[np.ones(np.shape(X1)[0])]

        diag_var = K12 - np.c_[np.einsum('ij,jk,ki->i', Kx1, self.posterior.woodbury_inv, Kx2)]
        return (self.std**2) * diag_var


def set_gpy_regressor(GPmodel, X, Y, kernel_bounds, noise_var=1e-6, optimize_num=10, optimize=True, normalizer=True):
    data_num, input_dim = np.shape(X)
    if GPmodel is None:

        kernel = GPy.kern.RBF(input_dim=input_dim, variance=1, ARD=True)
        gp_regressor = GPy_model(
            X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        gp_regressor['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        gp_regressor['.*rbf.variance'].constrain_fixed(1)
        if gp_regressor.kern.ARD:
            for i in range(input_dim):
                gp_regressor['.*rbf.lengthscale'][[i]
                                                ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
        else:
            gp_regressor['.*rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
        if optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    else:
        gp_regressor = GPy_model(
            X=X, Y=Y, kernel=GPmodel.kern, noise_var=GPmodel['.*Gaussian_noise.variance'].values, normalizer=GPmodel.normalizer)
        previous_data_num, _ = np.shape(GPmodel.X)
        temp_1 = data_num - data_num % OPTIMIZE_SEPARATOR
        temp_2 = previous_data_num - previous_data_num % OPTIMIZE_SEPARATOR
        if (temp_1 != temp_2) and optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    return gp_regressor


def set_gpy_regressor_addLinear(GPmodel, X, Y, kernel_bounds, noise_var=1e-6, optimize_num=10, optimize=True, normalizer=True):
    data_num, input_dim = np.shape(X)
    if GPmodel is None:

        kernel = GPy.kern.RBF(input_dim=input_dim, variance=1, ARD=True, active_dims=np.arange(input_dim).tolist()) + GPy.kern.Linear(input_dim=input_dim+1)
        gp_regressor = GPy_model_withIndicatorOne(
            X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        gp_regressor['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        gp_regressor['.*sum.rbf.variance'].constrain_bounded(0, 1.)
        gp_regressor['.*sum.linear.variances'].constrain_bounded(0, 1.)
        if gp_regressor.kern.rbf.ARD:
            for i in range(input_dim):
                gp_regressor['.*sum.rbf.lengthscale'][[i]
                                                ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
        else:
            gp_regressor['.*sum.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
        if optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    else:
        gp_regressor = GPy_model_withIndicatorOne(
            X=X, Y=Y, kernel=GPmodel.kern, noise_var=GPmodel['.*Gaussian_noise.variance'].values, normalizer=GPmodel.normalizer)
        previous_data_num, _ = np.shape(GPmodel.X)
        temp_1 = data_num - data_num % OPTIMIZE_SEPARATOR
        temp_2 = previous_data_num - previous_data_num % OPTIMIZE_SEPARATOR
        if (temp_1 != temp_2) and optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    return gp_regressor


def set_mogpy_regressor(GPmodel, X_list, Y_list, eval_num, kernel_bounds, M, noise_var=1e-6, optimize_num=10, optimize=True, normalizer=True, opt_separator = None):
    if opt_separator is not None:
        OPTIMIZE_SEPARATOR = opt_separator

    fidelity_indexes = np.hstack([i*np.ones(eval_num[i])
                                  for i in range(len(eval_num))])
    X = np.c_[np.vstack([X for X in X_list if np.size(X) > 0]), fidelity_indexes]
    Y = np.vstack([Y for Y in Y_list if np.size(Y) > 0])

    data_num = np.size(Y)
    input_dim = np.shape(X)[1] - 1
    if GPmodel is None:

        K = GPy.kern.RBF(input_dim=input_dim, ARD=True)
        B = GPy.kern.Coregionalize(input_dim=1, output_dim=M, rank=1)
        kernel = K**B
        for _ in range(M-1):
            K = GPy.kern.RBF(input_dim=input_dim, ARD=True)
            B = GPy.kern.Coregionalize(input_dim=1, output_dim=M, rank=1)
            kernel = kernel + K**B

        gp_regressor = GPy_model(X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        gp_regressor['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        gp_regressor['.*mul.rbf.variance'].constrain_fixed(1.)
        for m in range(M-1):
            gp_regressor['.*mul_'+str(m+1)+'.rbf.variance'].constrain_fixed(1.)

        if gp_regressor.kern.mul.rbf.ARD:
            for i in range(input_dim):
                gp_regressor['.*mul.rbf.lengthscale'][[i]].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
                for m in range(M-1):
                    gp_regressor['.*mul_'+str(m+1)+'.rbf.lengthscale'][[i]].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
        else:
            gp_regressor['.*mul.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])
            for m in range(M-1):
                gp_regressor['.*mul_'+str(m+1)+'.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])

        if optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    else:
        gp_regressor = GPy_model(X=X, Y=Y, kernel=GPmodel.kern,
                                 noise_var=GPmodel['.*Gaussian_noise.variance'].values, normalizer=GPmodel.normalizer)
        previous_data_num, _ = np.shape(GPmodel.X)
        temp_1 = data_num - data_num % OPTIMIZE_SEPARATOR
        temp_2 = previous_data_num - previous_data_num % OPTIMIZE_SEPARATOR
        if (temp_1 != temp_2 or data_num == previous_data_num)  and optimize:

            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    return gp_regressor


def set_mfgpy_regressor(GPmodel, X_list, Y_list, eval_num, kernel_bounds, M, noise_var=1e-6, optimize_num=10, optimize=True, normalizer=True, opt_separator = None):
    if OPTIMIZE_SEPARATOR is not None:
        OPTIMIZE_SEPARATOR = opt_separator
    fidelity_indexes = np.hstack([i*np.ones(eval_num[i])
                                  for i in range(len(eval_num))])
    X = np.c_[np.vstack([X for X in X_list if np.size(X) > 0]), fidelity_indexes]
    Y = np.vstack([Y for Y in Y_list if np.size(Y) > 0])

    data_num = np.size(Y)
    input_dim = np.shape(X)[1] - 1
    if GPmodel is None:

        K1 = GPy.kern.RBF(input_dim=input_dim, ARD=True)
        K2 = GPy.kern.RBF(input_dim=input_dim, ARD=True)
        B1 = GPy.kern.Coregionalize(input_dim=1, output_dim=M, rank=1)
        B2 = GPy.kern.Coregionalize(input_dim=1, output_dim=M, rank=1)
        kernel = K1**B1 + K2**B2

        gp_regressor = GPy_model(X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        gp_regressor = mf_model_constrain_cokriging(
            gp_regressor, kernel_bounds, M, input_dim, noise_var)
        if optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    else:
        gp_regressor = GPy_model(X=X, Y=Y, kernel=GPmodel.kern,
                                 noise_var=GPmodel['.*Gaussian_noise.variance'].values, normalizer=GPmodel.normalizer)
        previous_data_num, _ = np.shape(GPmodel.X)
        temp_1 = data_num - data_num % OPTIMIZE_SEPARATOR
        temp_2 = previous_data_num - previous_data_num % OPTIMIZE_SEPARATOR
        if (temp_1 != temp_2 or data_num == previous_data_num)  and optimize:

            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    return gp_regressor

def set_mtgpy_regressor(GPmodel, X_list, Y_list, eval_num, kernel_bounds, M, fidelity_features, fidelity_feature_dim, noise_var=1e-6, optimize_num=10, optimize=True, normalizer=True):
    fidelity_features = np.vstack([np.matlib.repmat(fidelity_features.ravel()[i], eval_num[i], 1)
                                  for i in range(len(eval_num))])
    X = np.c_[np.vstack([X for X in X_list if np.size(X) != 0]), fidelity_features]
    Y = np.vstack([Y for Y in Y_list if np.size(Y) != 0])

    data_num = np.size(Y)
    input_dim = np.shape(X)[1]
    if GPmodel is None:

        k1 = GPy.kern.RBF(input_dim = input_dim-fidelity_feature_dim, active_dims=list(range(input_dim-fidelity_feature_dim)), ARD=True)
        k2 = GPy.kern.RBF(input_dim = fidelity_feature_dim, active_dims=list(range(input_dim-fidelity_feature_dim, input_dim)), ARD=True)
        kernel = k1 * k2

        gp_regressor = GPy_model(X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        gp_regressor = mt_model_constrain(
            gp_regressor, kernel_bounds, M, input_dim, noise_var, fidelity_feature_dim)
        if optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    else:
        gp_regressor = GPy_model(X=X, Y=Y, kernel=GPmodel.kern,
                                 noise_var=GPmodel['.*Gaussian_noise.variance'].values, normalizer=GPmodel.normalizer)
        previous_data_num, _ = np.shape(GPmodel.X)
        temp_1 = data_num - data_num % OPTIMIZE_SEPARATOR
        temp_2 = previous_data_num - previous_data_num % OPTIMIZE_SEPARATOR
        if (temp_1 != temp_2 or data_num == previous_data_num)  and optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)

    return gp_regressor


def mf_model_constrain(GPy_model, kernel_bounds, M, input_dim, noise_var=1e-6):
    GPy_model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)

    GPy_model['.*mul.rbf.variance'].constrain_fixed(1)
    if GPy_model.kern.mul.rbf.ARD:
        for i in range(input_dim):
            GPy_model['.*rbf.lengthscale'][[i]
                                        ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
    else:
        GPy_model['.*rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])

    GPy_model['.*mul.coregion.W'].constrain_bounded(-1, 1)
    GPy_model['.*mul.coregion.kappa'].constrain_bounded(1e-8, 1e-1)

    GPy_model['.*mul_1.rbf.variance'].constrain_fixed(1)

    if GPy_model.kern.mul_1.rbf.ARD:
        for i in range(input_dim):
            GPy_model['.*mul_1.rbf.lengthscale'][[i]
                                                ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
    else:
        GPy_model['.*mul_1.rbf.lengthscale'].constrain_bounded(kernel_bounds[0, -1], kernel_bounds[1, -1])
    GPy_model['.*mul_1.coregion.W'].constrain_bounded(-1, 1)
    GPy_model['.*mul_1.coregion.kappa'].constrain_bounded(1e-8, 1e-1)
    return GPy_model

def mt_model_constrain(GPy_model, kernel_bounds, M, input_dim, noise_var=1e-6, fidelity_feature_dim=1):
    GPy_model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)

    GPy_model['.*mul.rbf.variance'].constrain_fixed(1)
    if GPy_model.kern.rbf.ARD:
        for i in range(input_dim - fidelity_feature_dim):
            GPy_model['.*rbf.lengthscale'][[i]
                                        ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
    else:
        GPy_model['.*rbf.lengthscale'].constrain_bounded(kernel_bounds[0, 0], kernel_bounds[1, 0])

    GPy_model['.*mul.rbf_1.variance'].constrain_fixed(1)
    if GPy_model.kern.rbf_1.ARD:
        for i in range(fidelity_feature_dim):
            GPy_model['.*mul.rbf_1.lengthscale'][[i]
                                                ].constrain_bounded(kernel_bounds[0, i+(input_dim - fidelity_feature_dim)], kernel_bounds[1, i+(input_dim - fidelity_feature_dim)])
    else:
        GPy_model['.*mul.rbf_1.lengthscale'].constrain_bounded(kernel_bounds[0, -1], kernel_bounds[1, -1])

    return GPy_model

def mf_model_constrain_cokriging(GPy_model, kernel_bounds, M, input_dim, noise_var=1e-6):
    GPy_model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
    GPy_model['.*mul.rbf.variance'].constrain_fixed(1)
    for i in range(input_dim):
        GPy_model['.*mul.rbf.lengthscale'][[i]
                                       ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])
    GPy_model['.*mul_1.rbf.variance'].constrain_fixed(1)
    for i in range(input_dim):
        GPy_model['.*mul_1.rbf.lengthscale'][[i]
                                             ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])


    GPy_model['.*mul.coregion.W'].constrain_bounded(np.sqrt(0.75), 1.)
    GPy_model['.*mul.coregion.kappa'].constrain_bounded(1e-2, 1e-1)
    GPy_model['.*mul_1.coregion.W'].constrain_bounded(-0.5, 0.5)
    GPy_model['.*mul_1.coregion.kappa'].constrain_bounded(1e-2, 1e-1)
    return GPy_model



def mf_model_constrain_cokriging_strictly(GPy_model, kernel_bounds, M, input_dim, noise_var=1e-6):
    GPy_model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)

    GPy_model['.*mul.rbf.variance'].constrain_fixed(1)
    for i in range(input_dim):
        GPy_model['.*mul.rbf.lengthscale'][[i]
                                       ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])





    GPy_model['.*mul_1.rbf.variance'].constrain_fixed(1)
    for i in range(input_dim):
        GPy_model['.*mul_1.rbf.lengthscale'][[i]
                                             ].constrain_bounded(kernel_bounds[0, i], kernel_bounds[1, i])













    return GPy_model

def trunc_norm_sampling(mu, cov_chol, upper):
    sampling_num = np.shape(upper)[1]
    standard_normal_samples = np.random.normal(size=(np.size(mu), sampling_num))
    scaled_normal_samples = cov_chol.dot(standard_normal_samples) + np.c_[mu]


    if np.any(scaled_normal_samples > upper):
        violate_index = np.any(scaled_normal_samples > upper, axis=0)
        standard_normal_samples[:, violate_index] = trunc_norm_sampling(mu, cov_chol, upper[:, violate_index])
    return standard_normal_samples

def marginal_pf_tmvn_entropy(mu, sigma, lower, upper, dim):
    '''
    return (dim)-dimention entropy of truncated multi-variate normal distribution truncated by pareto frontier.

    Parameters
    ----------
    mu : numpy array
        mean of original L-dimentional multi-variate nomal (L)
    sigma : numpy array
        covariance matrix of L-dimentional original multi-variate normal (L \times L)
    lower : numpy array
        lower position of M cells in region truncated by pareto frontier (M \times L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (M \times L)
    dim : int
        objective dimention ( in {0, ..., L-1})

    Retruns
    -------
    entropy : float
        dim-dimentional entopy of truncated multi-variate normal
    '''
    M, L = np.shape(lower)

    Z_ms = list()
    for m in range(M):
        Z_m, _ = mvn.mvnun(lower[m, :], upper[m, :], mu,
                           sigma, maxpts=L*1e4, abseps=1e-8, releps=1e-6)
        Z_ms.append(Z_m)
    Z_ms = np.array(Z_ms)
    Z = np.sum(Z_ms)

    dim_cells = np.unique(np.c_[lower[:, dim], upper[:, dim]], axis=0)
    dim_cells_Z = np.zeros(M)
    same_cells_index = list()
    for i in range(np.shape(dim_cells)[0]):
        in_upper_index = dim_cells[i, 1] == upper[:, dim]
        in_lower_index = dim_cells[i, 0] == lower[:, dim]
        same_cells_index.append(np.logical_and(in_lower_index, in_upper_index))
        dim_cells_Z[i] = np.sum(Z_ms[same_cells_index[i]])
    same_cells_index = np.array(same_cells_index)

    index_list = [dim] + [i for i in range(L) if i not in [dim]]
    mu = mu[index_list]
    sigma = sigma[index_list, :][:, index_list]
    lower = lower[:, index_list]
    upper = upper[:, index_list]

    cov_inv = 1./sigma[0, 0]
    temp = sigma[1:, 0]*cov_inv
    conditional_sigma = sigma[1:, 1:] - np.c_[temp].dot(np.c_[sigma[1:, 0]].T)

    def marginal_density_pf_tmvn(x):
        in_lower_index = dim_cells[:, 0] < x
        in_upper_index = x <= dim_cells[:, 1]
        cell_index = same_cells_index[np.logical_and(
            in_lower_index, in_upper_index)].ravel()
        in_lower = lower[cell_index]
        in_upper = upper[cell_index]

        conditional_mu = mu[1:] + temp*(x - mu[0]).ravel()
        pdf = norm.pdf(x, loc=mu[0], scale=np.sqrt(sigma[0, 0]))
        conditional_Z = 0
        for i in range(np.shape(in_lower)[0]):
            conditional_Z_m, _ = mvn.mvnun(in_lower[i, 1:], in_upper[i, 1:], conditional_mu.ravel(
            ), conditional_sigma, maxpts=(L-1)*1e4, abseps=1e-8, releps=1e-6)
            conditional_Z += conditional_Z_m
        marginal_density_tmvn = pdf * conditional_Z
        if marginal_density_tmvn == 0:
            marginal_density_tmvn = 1
        return - marginal_density_tmvn * np.log(marginal_density_tmvn)

    entropy = 0
    for i in range(np.shape(dim_cells)[0]):
        result = integrate.quad(marginal_density_pf_tmvn,
                                dim_cells[i, 0], dim_cells[i, 1])
        entropy += result[0] / Z + dim_cells_Z[i] * np.log(Z) / Z

    return entropy


def pf_tmvn_entropy(mu, sigma, lower, upper):
    '''
    return entropy of truncated multi-variate normal distribution truncated by pareto frontier.

    Parameters
    ----------
    mu : numpy array
        mean of original L-dimentional multi-variate nomal (L)
    sigma : numpy array
        covariance matrix of L-dimentional original multi-variate normal (L \times L)
    lower : numpy array
        lower position of M cells in region truncated by pareto frontier (M \times L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (M \times L)

    Retruns
    -------
    entropy : float
        entopy of truncated multi-variate normal
    Z : float
        truncated normalized constat (truncated by pareto frontier)
    '''



    M, _ = lower.shape
    sigma_inv = np.linalg.inv(sigma)


    Z_ms = list()
    entropys = list()
    for m in range(M):
        entropy_m, Z_m = tmvn_entropy(
            mu, sigma, sigma_inv, lower[m, :], upper[m, :])
        Z_ms.append(Z_m)
        entropys.append(entropy_m)

    Z_ms = np.array(Z_ms)
    Z_ms[Z_ms == 0] = 1
    entropys = np.array(entropys)
    Z = np.sum(Z_ms)



    entropy = np.sum(Z_ms * (entropys - np.log(Z_ms))) / Z + np.log(Z)
    return entropy, Z, Z_ms


def onesided_marginal_density_tmvn(x, mu, sigma, Z, L, upper):
    marginal_dim = int(np.size(x))

    if marginal_dim == L:
        return multivariate_normal.pdf(x, mean=mu, cov=sigma) / Z

    cov_inv = np.linalg.inv(sigma[:marginal_dim, :marginal_dim])
    temp = sigma[marginal_dim:, :marginal_dim].dot(cov_inv)
    conditional_sigma = sigma[marginal_dim:, marginal_dim:] - \
        temp.dot(sigma[marginal_dim:, :marginal_dim].T)
    conditional_mu = mu[marginal_dim:] + \
        temp.dot(np.c_[x - mu[:marginal_dim]]).ravel()



    pdf = multivariate_normal.pdf(x, mean=mu[:marginal_dim], cov=sigma[:marginal_dim, :marginal_dim])
    conditional_Z_m, _ = mvn.mvnun(conditional_mu.ravel() - 1e2*np.sqrt(np.diag(conditional_sigma)), upper[marginal_dim:], conditional_mu.ravel(), conditional_sigma, maxpts=(L - marginal_dim)*1e4, abseps=1e-5, releps=1e-3)
    if np.isnan(conditional_Z_m):
        print('conditional_Z_m is NANs')
        print(conditional_sigma)
        print(np.linalg.inv(conditional_sigma))
        print(upper[marginal_dim:])
        print(conditional_mu.ravel())
    return pdf * conditional_Z_m / Z

def for_BMES(upper, mu, sigma, Z, L):
    upper = upper - mu
    zero_mean = np.zeros(L)

    sigma_plus_dd = sigma.copy()
    F_k_q_bk_bq = np.empty((L,L))

    for k in range(L):
        dims = [k]
        index_list = dims + [i for i in range(L) if i not in dims]
        temp_sigma = sigma[index_list, :][:, index_list]
        temp_upper = upper[index_list]

        F_k_bk = onesided_marginal_density_tmvn(upper[k], zero_mean, temp_sigma, Z, L, temp_upper)

        sigma_plus_dd -= upper[k]*F_k_bk * np.c_[sigma[:, k]] * sigma[:, k] / sigma[k, k]
        latter_sum = 0
        for q in range(L):
            if q != k:
                fir_term = sigma[:, q] - sigma[k, q] * sigma[:, k] / sigma[k, k]

                dims = [k, q]
                index_list = dims + [i for i in range(L) if i not in dims]
                temp_sigma = sigma[index_list, :][:, index_list]
                temp_upper = upper[index_list]

                if q > k:
                    F_k_q_bk_bq[k, q] = onesided_marginal_density_tmvn(upper[dims], zero_mean, temp_sigma, Z, L, temp_upper)
                    F_k_q_bk_bq[q, k] = F_k_q_bk_bq[k, q]

                latter_sum += fir_term * F_k_q_bk_bq[k,q]
        sigma_plus_dd += np.c_[sigma[:, k]] * latter_sum
    return sigma_plus_dd

def for_BMES_fast(upper, mu, sigma, Z, L):
    upper = upper - mu
    zero_mean = np.zeros(L)



    base_list = np.arange(L).tolist()
    F_k = np.zeros(L)
    F_kq = np.zeros((L, L))
    for k in range(L):
        index_list = base_list.copy()
        index_list = [index_list.pop(k)] + index_list
        temp_sigma = sigma[index_list, :][:, index_list]
        temp_upper = upper[index_list]

        F_k[k] = onesided_marginal_density_tmvn(upper[k], zero_mean, temp_sigma, Z, L, temp_upper)

        for q in range(L):
            if q > k:
                index_list = [index_list.pop(q)] + index_list
                temp_sigma = sigma[index_list, :][:, index_list]
                temp_upper = upper[index_list]

                F_kq[q, k] = onesided_marginal_density_tmvn(upper[[q,k]], zero_mean, temp_sigma, Z, L, temp_upper)

                F_kq[k, q] = F_kq[q, k]


    A = - np.sum( (F_k * upper / np.diag(sigma))[None,None,:] * sigma[:,None,:] * sigma[None,:,:], axis=2)
    A += np.sum(sigma[:,None,:,None] * (sigma[None,:,None,:] - sigma[None,None,:,:] * sigma[None,:,:,None] / np.diag(sigma)[None,None,:,None]) * F_kq[None,None,:,:], axis=(2,3))
    return A


def onesided_tmvn_params(upper, mu, sigma, Z, L):
    '''
    return means and covariance matrices of truncated multi-variate normal distribution truncated by pareto frontier.

    Parameters
    ----------
    mu : numpy array
        mean of original L-dimentional multi-variate nomal (L)
    sigma : numpy array
        covariance matrix of L-dimentional original multi-variate normal (L \times L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (L)

    Retruns
    -------
    mu_TN : numpy array
        means of truncated multi-variate normal (L)
    sigma_TN : numpy array
        covariance matrices of truncated multi-variate normal (L \times L)
    d   : numpy array (L)
        diffrence of original mean and mean of trancated multi-variate normal
    '''
    upper = upper - mu
    zero_mean = np.zeros(L)
    d = np.zeros(L)
    sigma_TN = sigma.copy()

    for k in range(L):
        dims = [k]
        index_list = dims + [i for i in range(L) if i not in dims]
        temp_sigma = sigma[index_list, :][:, index_list]
        temp_upper = upper[index_list]

        F_k_bk = onesided_marginal_density_tmvn(upper[k], zero_mean, temp_sigma, Z, L, temp_upper)
        d -= F_k_bk * sigma[:, k]
        sigma_TN -= upper[k]*F_k_bk * np.c_[sigma[:, k]] * sigma[:, k] / sigma[k, k]
        latter_sum = 0
        for q in range(L):
            if q != k:
                fir_term = sigma[:, q] - sigma[k, q] * sigma[:, k] / sigma[k, k]

                dims = [k, q]
                index_list = dims + [i for i in range(L) if i not in dims]
                temp_sigma = sigma[index_list, :][:, index_list]
                temp_upper = upper[index_list]

                F_k_q_bk_bq = onesided_marginal_density_tmvn(upper[dims], zero_mean, temp_sigma, Z, L, temp_upper)
                latter_sum += fir_term * F_k_q_bk_bq
        sigma_TN += np.c_[sigma[:, k]] * latter_sum

    mu_TN = mu + d
    d = np.c_[d]
    sigma_TN = sigma_TN - d.dot(d.T)
    return mu_TN, sigma_TN, d


def marginal_density_tmvn(x, mu, sigma, Z, L, lower, upper, temp=None, conditional_sigma=None):
    marginal_dim = int(np.size(x))

    if marginal_dim == L:
        return multivariate_normal.pdf(x, mean=mu, cov=sigma) / Z, None, None

    if (temp is None) or (conditional_sigma is None):
        cov_inv = np.linalg.inv(sigma[:marginal_dim, :marginal_dim])
        temp = sigma[marginal_dim:, :marginal_dim].dot(cov_inv)
        conditional_sigma = sigma[marginal_dim:, marginal_dim:] - \
            temp.dot(sigma[marginal_dim:, :marginal_dim].T)
    conditional_mu = mu[marginal_dim:] + \
        temp.dot(np.c_[x - mu[:marginal_dim]]).ravel()



    pdf = multivariate_normal.pdf(x, mean=mu[:marginal_dim], cov=sigma[:marginal_dim, :marginal_dim])
    conditional_Z_m, _ = mvn.mvnun(lower[marginal_dim:], upper[marginal_dim:], conditional_mu.ravel(
    ), conditional_sigma, maxpts=(L - marginal_dim)*1e4, abseps=1e-8, releps=1e-3)
    if np.isnan(conditional_Z_m):
        print('conditional_Z_m is NANs')
        print(conditional_sigma)
        print(np.linalg.inv(conditional_sigma))
        print(lower[marginal_dim:])
        print(upper[marginal_dim:])
        print(conditional_mu.ravel())
    return pdf * conditional_Z_m / Z, temp, conditional_sigma

def tmvn_params(lower, upper, mu, sigma, Z, L):
    '''
    return means and covariance matrices of truncated multi-variate normal distribution truncated by pareto frontier.

    Parameters
    ----------
    mu : numpy array
        mean of original L-dimentional multi-variate nomal (L)
    sigma : numpy array
        covariance matrix of L-dimentional original multi-variate normal (L \times L)
    lower : numpy array
        lower position of M cells in region truncated by pareto frontier (L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (L)

    Retruns
    -------
    mu_TN : numpy array
        means of truncated multi-variate normal (L)
    sigma_TN : numpy array
        covariance matrices of truncated multi-variate normal (L \times L)
    d   : numpy array (L)
        diffrence of original mean and mean of trancated multi-variate normal
    '''
    lower = lower - mu
    upper = upper - mu
    zero_mean = np.zeros(L)
    d = np.zeros(L)
    sigma_TN = sigma.copy()

    for k in range(L):
        dims = [k]
        index_list = dims + [i for i in range(L) if i not in dims]
        temp_sigma = sigma[index_list, :][:, index_list]
        temp_lower = lower[index_list]
        temp_upper = upper[index_list]

        F_k_ak, temp, cond_sigma = marginal_density_tmvn(
            lower[k], zero_mean, temp_sigma, Z, L, temp_lower, temp_upper)
        F_k_bk, _, _ = marginal_density_tmvn(
            upper[k], zero_mean, temp_sigma, Z, L, temp_lower, temp_upper, temp=temp, conditional_sigma=cond_sigma)
        d += (F_k_ak - F_k_bk) * sigma[:, k]
        sigma_TN += (lower[k]*F_k_ak - upper[k]*F_k_bk) * \
            np.c_[sigma[:, k]] * sigma[:, k] / sigma[k, k]
        latter_sum = 0
        for q in range(L):
            if q != k:
                fir_term = sigma[:, q] - sigma[k, q] * \
                    sigma[:, k] / sigma[k, k]

                dims = [k, q]
                index_list = dims + [i for i in range(L) if i not in dims]
                temp_sigma = sigma[index_list, :][:, index_list]
                temp_lower = lower[index_list]
                temp_upper = upper[index_list]

                F_k_q_ak_aq, temp, cond_sigma = marginal_density_tmvn(
                    lower[[k, q]], zero_mean, temp_sigma, Z, L, temp_lower, temp_upper)
                F_k_q_ak_bq, _, _ = marginal_density_tmvn(
                    [lower[k], upper[q]], zero_mean, temp_sigma, Z, L, temp_lower, temp_upper, temp=temp, conditional_sigma=cond_sigma)
                F_k_q_bk_aq, _, _ = marginal_density_tmvn(
                    [upper[k], lower[q]], zero_mean, temp_sigma, Z, L, temp_lower, temp_upper, temp=temp, conditional_sigma=cond_sigma)
                F_k_q_bk_bq, _, _ = marginal_density_tmvn(
                    upper[[k, q]], zero_mean, temp_sigma, Z, L, temp_lower, temp_upper, temp=temp, conditional_sigma=cond_sigma)
                latter_sum += fir_term * \
                    (F_k_q_ak_aq - F_k_q_ak_bq - F_k_q_bk_aq + F_k_q_bk_bq)
        sigma_TN += np.c_[sigma[:, k]] * latter_sum

    mu_TN = mu + d
    d = np.c_[d]
    sigma_TN = sigma_TN - d.dot(d.T)







    return mu_TN, sigma_TN, d


def tmvn_entropy(mu, sigma, sigma_inv, lower, upper):
    L = np.size(mu)
    Z, _ = mvn.mvnun(lower, upper, mu, sigma, maxpts=L *
                     1e4, abseps=1e-8, releps=1e-6)
    if Z == 0:
        return 0, 0

    _, sigma_TN, d = tmvn_params(lower, upper, mu, sigma, Z, L)

    entropy = (np.log(np.linalg.det(2*np.pi*sigma)) +
               np.trace(sigma_inv.dot(sigma_TN + d.dot(d.T)))) / 2 + np.log(Z)
    return entropy, Z

def ep_tmvn_params(lower, upper, mu, sigma, L):
    '''
    return means and covariance matrices of truncated multi-variate normal distribution truncated by pareto frontier.

    Parameters
    ----------
    mu : numpy array
        mean of original L-dimentional multi-variate nomal (L)
    sigma : numpy array
        covariance matrix of L-dimentional original multi-variate normal (L \times L)
    lower : numpy array
        lower position of M cells in region truncated by pareto frontier (L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (L)

    Returns
    -------
    mu_TN : numpy array
        means of truncated multi-variate normal (L)
    sigma_TN : numpy array
        covariance matrices of truncated multi-variate normal (L \times L)
    '''
    mu_tilde = np.zeros(L)
    sigma_tilde = np.inf*np.ones(L)
    mu_TN = mu
    sigma_TN = sigma

    sigma_inv = np.linalg.inv(sigma)
    sigma_inv_mu = sigma_inv.dot(mu)


    for i in range(10000):
        sigma_bar =  1./(1./np.diag(sigma_TN) - 1./sigma_tilde)
        mu_bar = sigma_bar * (mu_TN/np.diag(sigma_TN) - mu_tilde/sigma_tilde)

        alpha = (lower - mu_bar) / np.sqrt(sigma_bar)
        beta = (upper - mu_bar) / np.sqrt(sigma_bar)
        Z = norm.cdf(beta) - norm.cdf(alpha)
        alpha_pdf = norm.pdf(alpha)
        beta_pdf = norm.pdf(beta)
        diff_pdf = alpha_pdf - beta_pdf
        diff_pdf_product = alpha*alpha_pdf - beta*beta_pdf

        gamma = diff_pdf_product / Z - (diff_pdf / Z)**2
        gamma_0_index = np.abs(gamma) <= 1e-16
        gamma[gamma_0_index] = 1








        sigma_tilde = -(1./gamma + 1) * sigma_bar
        mu_tilde = mu_bar - 1./gamma * (diff_pdf / Z) * np.sqrt(sigma_bar)


        sigma_tilde[gamma_0_index] = np.inf
        mu_tilde[gamma_0_index] = 0

        sigma_tilde_inv = np.diag(1./sigma_tilde)
        sigma_TN_new = np.linalg.inv(sigma_tilde_inv + sigma_inv)
        mu_TN_new = sigma_TN_new.dot(sigma_tilde_inv.dot(mu_tilde) + sigma_inv_mu)

        change = np.max([np.max(np.abs(sigma_TN_new - sigma_TN)), np.max(np.abs(mu_TN_new - mu_TN))])

        sigma_TN = sigma_TN_new
        mu_TN = mu_TN_new

        if np.isnan(change):
            print('iteration :', i)
            print('mu', mu)
            print('sigma', sigma)
            print('lower', lower)
            print('upper', upper)
            print('mu_TN', mu_TN)
            print('sigma_TN', sigma_TN)
            print(gamma)
            exit()

        if change < 1e-10:

            break

    return mu_TN, sigma_TN

def main():
    np.random.seed(0)
    L = 4

    mu = np.random.rand(L) * 10 - 5
    sigma = 0.001 * np.ones((L, L)) + np.diag(5 * np.ones(L))
    upper = mu + 1e2 * np.sqrt(np.diag(sigma))
    lower = -1e4 * np.ones(L)
    upper[-1] = np.random.rand(1)
    upper[-2] = np.random.rand(1)






    print('mu :', mu)
    print('upper :', upper)
    print('lower :', lower)

    Z, _ = mvn.mvnun(lower, upper, mu, sigma, maxpts=L * 1e4, abseps=1e-8, releps=1e-6)

    start = time.time()
    mu_1, sigma_1, _ = tmvn_params(lower, upper, mu, sigma, Z, L)
    print('1 :', time.time() - start)
    print('mu_1 : \n', mu_1)
    print('sigma_1 ;\n', sigma_1)














    exit()

    for l in range(L):
        print('L =', l)
        mu = np.ones(l)
        sigma = 0.5 * np.ones((l, l)) + np.diag(0.5 * np.ones(l))

        temp = np.arange(10) - 5
        lower = np.array([i*np.ones(l) for i in temp])
        temp = temp + 1
        upper = np.array([i*np.ones(l) for i in temp])


        start = time.time()

        entropy, Z, _ = pf_tmvn_entropy(mu, sigma, lower, upper)
        elapsed_time = time.time() - start

        print('Exact entropy : ', entropy, 'time : ', elapsed_time)














        start = time.time()
        marginal_entropys = list()
        for i in range(l):
            marginal_entropys.append(
                marginal_pf_tmvn_entropy(mu, sigma, lower, upper, i))
        elapsed_time = time.time() - start
        print('marginal_ents : ',
              marginal_entropys[0], 'time(per) : ', elapsed_time / l)

        if l <= 3:
            def minus_plogp(*args):
                x = np.array([args])
                pdf = multivariate_normal.pdf(x, mean=mu, cov=sigma) / Z
                if pdf == 0:
                    pdf = 1
                return - pdf * np.log(pdf)

            integrate_entropy = 0

            start = time.time()
            for i in range(int(np.size(lower) / l)):
                result = integrate.nquad(minus_plogp, [[lower[i, k], upper[i, k]] for k in range(
                    l)], opts=[{'epsabs': 1e-8, 'epsrel': 1e-8} for i in range(l)], full_output=True)

                integrate_entropy += result[0]
            integrate_entropy = integrate_entropy
            elapsed_time = time.time() - start


            print('integ entropy : ', integrate_entropy, 'time : ', elapsed_time)
            print('digit of diff : ', -
                  np.log10(np.abs(entropy - integrate_entropy)))


if __name__ == '__main__':
    main()
