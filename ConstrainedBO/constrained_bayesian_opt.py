# -*- coding: utf-8 -*-
import os
import sys
import time
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipyminimize
from scipydirect import minimize
from scipy.stats import norm
from scipy.stats import mvn
import nlopt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../myutils"))
import myutils as utils
from BO_core import ConstrainedBO_core

class CBO(ConstrainedBO_core):
    __metaclass__ = ABCMeta

    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=None, GPmodel=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.preprocessing_time = 0

    def next_input_pool(self, X):
        if self.model=='independent':
            self.acquisition_values = self.acq(X)
        elif self.model=='correlated':
            self.acquisition_values = self.acq_correlated(X)

        if np.all(self.acquisition_values == 0):
            print('all acquisition function value is zero')
            max_index = np.random.randint(0, np.shape(X)[0], 1)
        else:
            max_index = np.argmin(self.acquisition_values)
        max_index = np.argmin(self.acquisition_values)
        next_input = [np.atleast_2d(X[max_index]) for c in range(self.C+1)]
        X = np.delete(X, max_index, axis=0)
        return next_input, X

    def next_input(self):
        NUM_START = 1000
        x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        if self.x_max is not None:
            x0 = np.r_[x0, np.c_[self.x_max].T]
        if self.max_inputs is not None:
            x0 = np.r_[x0, self.max_inputs]
        max_acq = 0
        max_x = None

        if self.model=='independent':
            acq0 = self.acq(x0)
            x0 = x0[acq0 != 0]
        elif self.model=='correlated':
            temp_x0 = np.c_[np.matlib.repmat(x0, self.C+1, 1), np.c_[np.array([i*np.ones(np.shape(x0)[0]) for i in range(self.C+1)]).ravel()]]
            mean, var = self.GPmodel.predict(temp_x0)
            upper_bound = mean.reshape((np.shape(x0)[0], self.C+1)) + 3 * var.reshape((np.shape(x0)[0], self.C+1))
            x0 = x0[np.any(upper_bound > np.r_[[np.min(self.max_samples)], self.thresholds], axis=1)]
            opt = nlopt.opt(nlopt.LN_BOBYQA, self.input_dim)
            opt.set_lower_bounds(self.bounds[0].tolist())
            opt.set_upper_bounds(self.bounds[1].tolist())
            opt.set_xtol_rel(1e-3)
            opt.set_ftol_rel(1e-2)
            opt.set_maxtime(60*2 / NUM_START)
            opt.set_maxeval(100)
            def f(x, grad):
                return self.acq_correlated(x)
            opt.set_min_objective(f)

        if np.size(x0) != 0:
            for i in range(np.shape(x0)[0]):
                if self.model=='independent':
                    res = scipyminimize(self.acq, x0=x0[i], bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-3})
                elif self.model=='correlated':

                    x = opt.optimize(x0[i])
                    minf = opt.last_optimum_value()


                    res = {'fun': minf, 'x': x}


                if max_acq < - res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):


                    max_acq = - res['fun']
                    max_x = res['x']

        if max_x is None:
            if self.max_inputs is not None:
                print('select by Thompson Sampling')
                max_x = self.max_inputs[0]
                inputs = [np.atleast_2d(max_x) for c in range(self.C+1)]
            else:
                print('select by Uncertainty Sampling')
                inputs = self.uncertainty_sampling()

        else:
            if self.model=='independent':
                res = scipyminimize(self.acq, x0=max_x, bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-16})
            elif self.model=='correlated':
                res = scipyminimize(self.acq_correlated, x0=max_x, bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-16})
            if max_acq < - res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                max_acq = - res['fun']
                max_x = res['x']

            print('max_acq(acquisition function value at selected point):', max_acq)
            inputs = [np.atleast_2d(max_x) for c in range(self.C+1)]

        return inputs

    def uncertainty_sampling(self):
        def uncertainty(x):
            x = np.atleast_2d(x)
            entropy = 0
            for c in range(self.C+1):
                _, var = self.GPmodel.predict_noiseless(np.c_[x, c*np.c_[np.ones(np.shape(x)[0])]])
                entropy += np.log(var)
            return - entropy.ravel()[0]

        NUM_START = 100
        x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        max_acq = -np.inf
        max_x = None

        opt = nlopt.opt(nlopt.LN_BOBYQA, self.input_dim)
        opt.set_lower_bounds(self.bounds[0].tolist())
        opt.set_upper_bounds(self.bounds[1].tolist())
        opt.set_xtol_rel(1e-3)
        opt.set_ftol_rel(1e-2)
        opt.set_maxtime(30 / NUM_START)
        opt.set_maxeval(1000)
        def f(x, grad):
            return uncertainty(x)
        opt.set_min_objective(f)

        for i in range(np.shape(x0)[0]):


            res = opt.last_optimize_result()
            x = opt.optimize(x0[i])
            minf = opt.last_optimum_value()
            res = {'fun': minf, 'x': x}

            if max_acq < - res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                max_acq = - res['fun']
                max_x = res['x']
        inputs = [np.atleast_2d(max_x) for c in range(self.C+1)]
        return inputs



class ConstrainedExpectedImprovement(CBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=None, GPmodel=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)

    def acq(self, x):
        x = np.atleast_2d(x)

        if self.y_max is not None:
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]])
            std = np.sqrt(var)
            Z = (mean - self.y_max) / std
            acq = ((Z * std)*norm.cdf(Z) + std*norm.pdf(Z)).ravel()
        else:
            acq = 1

        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]])
            prob = 1 - norm.cdf((self.thresholds[c] - mean) / np.sqrt(var)).ravel()
            acq = acq * prob
        return - acq

class ConstrainedThompsonSampling(CBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=None, GPmodel=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = 1

    def next_input(self, pool_X=None):
        _, inputs = self.sampling_RFM(pool_X=pool_X, sampling_approach='MinViolation')
        inputs = [np.array(inputs) for c in range(self.C+1)]
        return inputs

    def next_input_pool(self, pool_X=None):
        inputs = self.next_input(pool_X=pool_X)
        selected_index = np.all(pool_X == inputs[0], axis=1)
        pool_X = pool_X[np.logical_not(selected_index)]
        return inputs, pool_X



class ConstrainedMaxvalueEntropySearch(CBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, sampling_num=10, sampling_approach='inf', cost=None, GPmodel=None, pool_X=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = sampling_num
        self.pool_X = pool_X
        self.sampling_approach = sampling_approach
        self.maximum_sampling()


    def maximum_sampling(self):
        start = time.time()
        self.max_samples, self.max_inputs = self.sampling_RFM(pool_X=self.pool_X, sampling_approach=self.sampling_approach)
        print('max_sample time:', time.time() - start)

        self.max_inputs = np.atleast_2d([self.max_inputs[i] for i in range(self.sampling_num) if self.max_inputs[i] is not None])
        self.max_samples = np.c_[self.max_samples].T
        print('sampled maximums:', self.max_samples)
        if self.input_dim <= 10:
            print('sampled_max-inputs:', self.max_inputs.tolist())

        if np.size(self.max_inputs) == 0:
            self.max_inputs = None

    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list=add_X_list, add_Y_list=add_Y_list, optimize=optimize)
        self.maximum_sampling()


    def acq(self, x):
        x = np.atleast_2d(x)




        gammas_c = list()
        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]])
            gammas_c.append(( (self.thresholds[c] - mean) / np.sqrt(var)).ravel())

        gammas_c = np.array(gammas_c)
        survival_funcs_c = 1 - norm.cdf(gammas_c)

        Z_star = np.prod(survival_funcs_c, axis=0)


        index_c = survival_funcs_c==0
        survival_funcs_c[index_c] = 1
        inner_sum_c = gammas_c * norm.pdf(gammas_c) / survival_funcs_c

        inner_sum_c[index_c] = gammas_c[index_c]**2
        inner_sum_c = np.sum(inner_sum_c, axis=0)





        mean, var = self.GPmodel.predict_noiseless(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]])
        gamma_f = (self.max_samples - mean) / np.sqrt(var)
        survival_funcs_f = 1 - norm.cdf(gamma_f)

        Z_star = np.c_[Z_star] * survival_funcs_f


        index_f = survival_funcs_f==0
        survival_funcs_f[index_f] = 1

        gamma_f[np.isinf(gamma_f)] = 0
        inner_sum_f = gamma_f * norm.pdf(gamma_f) / survival_funcs_f

        inner_sum_f[index_f] = gamma_f[index_f]**2
        inner_sum = inner_sum_f + np.c_[inner_sum_c]


        Z_star[Z_star==1] = 1 - 1e-16

        acq = np.sum(Z_star / (2* (1 - Z_star)) * inner_sum - np.log(1-Z_star), axis=1)
        return - acq

class ConstrainedMaxvalueEntropySearch_LB(CBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, sampling_num=10, sampling_approach='inf', cost=None, GPmodel=None, pool_X=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = sampling_num
        self.pool_X = pool_X
        self.sampling_approach = sampling_approach
        self.maximum_sampling()
        self.upper_inf = np.inf*np.ones((self.C+1))
        self.infin = np.ones(self.C+1)


    def maximum_sampling(self):
        start = time.time()
        self.max_samples, self.max_inputs = self.sampling_RFM(pool_X=self.pool_X, sampling_approach=self.sampling_approach)
        print('max_sample time:', time.time() - start)

        self.max_inputs = np.atleast_2d([self.max_inputs[i] for i in range(self.sampling_num) if self.max_inputs[i] is not None])
        self.max_samples = np.c_[self.max_samples].T
        print('sampled maximums:', self.max_samples)
        if self.input_dim <= 10:
            print('sampled_max-inputs:', self.max_inputs.tolist())

        if np.size(self.max_inputs) == 0:
            self.max_inputs = None

        self.lower = np.r_[self.max_samples, np.matlib.repmat(np.c_[self.thresholds], 1, self.sampling_num)]

    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list=add_X_list, add_Y_list=add_Y_list, optimize=optimize)
        self.maximum_sampling()


    def acq(self, x):
        x = np.atleast_2d(x)




        gammas_c = list()
        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]])
            gammas_c.append(( (self.thresholds[c] - mean) / np.sqrt(var)).ravel())

        gammas_c = np.array(gammas_c)
        survival_funcs_c = 1 - norm.cdf(gammas_c)

        Z_star = np.prod(survival_funcs_c, axis=0)




        mean, var = self.GPmodel.predict_noiseless(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]])
        gamma_f = (self.max_samples - mean) / np.sqrt(var)
        survival_funcs_f = 1 - norm.cdf(gamma_f)

        Z_star = np.c_[Z_star] * survival_funcs_f


        Z_star[Z_star==1] = 1 - 1e-16

        acq = - np.sum(np.log(1-Z_star), axis=1)
        return - acq

    def acq_correlated(self, x):
        x = np.atleast_2d(x)

        if np.shape(x)[0] > 1:
            print('correlated version of CMES-IBO acquisition function can not deal with multiple xs')
            exit(1)

        x = np.c_[np.matlib.repmat(x, self.C+1, 1), np.c_[np.arange(self.C+1)]]

        mean, cov = self.GPmodel.predict(x, full_cov=True)







        std = np.sqrt(np.diag(cov))
        correlation_matrix = cov / np.c_[std] / std
        correl = correlation_matrix[np.tril_indices(self.C+1, k=-1)]
        standardized_lower = (self.lower - mean) / np.c_[std]
        Z_star = [mvn.mvndst(standardized_lower[:,i], self.upper_inf, self.infin, correl)[1] for i in range(self.sampling_num)]



        Z_star = np.array(Z_star)

        Z_star[Z_star==1] = 1 - 1e-16

        acq = - np.sum(np.log(1 - Z_star))
        return - acq





'''
class ConstrainedMaxvalueEntropySearch_seqdec_KL(CBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, sampling_num=10, cost=None, GPmodel=None, pool_X=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = sampling_num

        start = time.time()
        self.max_samples, self.max_inputs = self.sampling_RFM(pool_X=pool_X, )
        print('max_sample time:', time.time() - start)

        self.max_inputs = np.atleast_2d([self.max_inputs[i] for i in range(self.sampling_num) if self.max_inputs[i] is not None])
        self.max_samples = np.c_[self.max_samples].T
        print('sampled maximums:', self.max_samples)
        print('sampled_max-inputs:', self.max_inputs.tolist())


        if np.size(self.max_inputs) == 0:
            self.max_inputs = None

    def next_input(self):
        NUM_START = 1000
        x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        if self.x_max is not None:
            x0 = np.r_[x0, np.c_[self.x_max].T]
        if self.max_inputs is not None:
            x0 = np.r_[x0, self.max_inputs]
        max_acq = 0
        max_x = None

        acq0 = self.acq(x0)
        x0 = x0[acq0 != 0]
        if np.size(x0) != 0:
            for i in range(np.shape(x0)[0]):
                res = scipyminimize(self.acq, x0=x0[i], bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-3})
                if max_acq < - res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                    max_acq = - res['fun']
                    max_x = res['x']

        if max_x is None:
            if self.max_inputs is not None:
                print('select by Thompson Sampling')
                max_x = self.max_inputs[0]
            else:
                max_x = self.uncertainty_sampling()[0]
                print('select by Uncertainty Sampling')

        else:
            res = scipyminimize(self.acq, x0=max_x, bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-16})
            if max_acq < - res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                max_acq = - res['fun']
                max_x = res['x']


        max_c = 0
        max_acq = - np.inf
        for c in range(self.C+1):
            self.eval_c = c
            acq_KL_val = - self.acq_dec_KL(max_x)
            if max_acq < acq_KL_val:
                max_acq = acq_KL_val
                max_c = c

        inputs = [np.array([]).reshape((0, self.input_dim)) for c in range(self.C+1)]
        inputs[max_c] = np.atleast_2d(max_x)
        return inputs

    def acq(self, x):
        x = np.atleast_2d(x)




        gammas_c = list()
        for c in range(self.C):
            mean, var = self.GPmodel_list[c+1].predict_noiseless(x)
            gammas_c.append(( (self.thresholds[c] - mean) / np.sqrt(var)).ravel())

        gammas_c = np.array(gammas_c)
        survival_funcs_c = 1 - norm.cdf(gammas_c)

        Z_star = np.prod(survival_funcs_c, axis=0)


        index_c = survival_funcs_c==0
        survival_funcs_c[index_c] = 1
        inner_sum_c = gammas_c * norm.pdf(gammas_c) / survival_funcs_c

        inner_sum_c[index_c] = gammas_c[index_c]**2
        inner_sum_c = np.sum(inner_sum_c, axis=0)





        mean, var = self.GPmodel_list[0].predict_noiseless(x)
        gamma_f = (self.max_samples - mean) / np.sqrt(var)
        survival_funcs_f = 1 - norm.cdf(gamma_f)

        Z_star = np.c_[Z_star] * survival_funcs_f


        index_f = survival_funcs_f==0
        survival_funcs_f[index_f] = 1

        gamma_f[np.isinf(gamma_f)] = 0
        inner_sum_f = gamma_f * norm.pdf(gamma_f) / survival_funcs_f

        inner_sum_f[index_f] = gamma_f[index_f]**2
        inner_sum = inner_sum_f + np.c_[inner_sum_c]


        Z_star[Z_star==1] = 1 - 1e-16

        acq = np.sum(Z_star / (2* (1 - Z_star)) * inner_sum - np.log(1-Z_star), axis=1)
        return - acq

    def acq_dec_KL(self, x):
        x = np.atleast_2d(x)




        gammas_c = list()
        for c in range(self.C):
            mean, var = self.GPmodel_list[c+1].predict_noiseless(x)
            gammas_c.append(( (self.thresholds[c] - mean) / np.sqrt(var)).ravel())

        gammas_c = np.array(gammas_c)
        survival_funcs_c = 1 - norm.cdf(gammas_c)

        Z_star = np.prod(survival_funcs_c, axis=0)




        mean, var = self.GPmodel_list[0].predict_noiseless(x)
        gamma_f = (self.max_samples - mean) / np.sqrt(var)
        survival_funcs_f = 1 - norm.cdf(gamma_f)

        Z_star = np.c_[Z_star] * survival_funcs_f

        if self.eval_c == 0:

            index_f = survival_funcs_f==0
            survival_funcs_f[index_f] = 1
            one_minus_cdf = survival_funcs_f




        else:

            index_c = survival_funcs_c[self.eval_c-1,:]==0
            survival_funcs_c[self.eval_c-1,:][index_c] = 1
            one_minus_cdf = np.c_[survival_funcs_c[self.eval_c-1,:]]





        Z_star[Z_star>=1] = 1 - 1e-16





        acq = np.log(1-Z_star)


        Z_star[Z_star >= one_minus_cdf] = 0
        acq = acq + (one_minus_cdf - Z_star) / (1 - Z_star) * np.log((one_minus_cdf - Z_star) / one_minus_cdf)
        acq = np.sum(acq, axis=1)
        return - acq

'''

class ConstrainedMaxvalueEntropySearch_dec(CBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, sampling_num=10, sampling_approach='inf', cost=None, GPmodel=None, pool_X=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = sampling_num
        self.pool_X = pool_X
        self.sampling_approach = sampling_approach
        self.maximum_sampling()


    def maximum_sampling(self):
        start = time.time()
        self.max_samples, self.max_inputs = self.sampling_RFM(pool_X=self.pool_X, sampling_approach=self.sampling_approach)
        print('max_sample time:', time.time() - start)

        self.max_inputs = np.atleast_2d([self.max_inputs[i] for i in range(self.sampling_num) if self.max_inputs[i] is not None])
        self.max_samples = np.c_[self.max_samples].T
        print('sampled maximums:', self.max_samples)
        if self.input_dim <= 10:
            print('sampled_max-inputs:', self.max_inputs.tolist())

        if np.size(self.max_inputs) == 0:
            self.max_inputs = None

    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list=add_X_list, add_Y_list=add_Y_list, optimize=optimize)
        self.maximum_sampling()


    def next_input(self):
        NUM_START = 1000
        x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        if self.x_max is not None:
            x0 = np.r_[x0, np.c_[self.x_max].T]
        if self.max_inputs is not None:
            x0 = np.r_[x0, self.max_inputs]
        max_acq = 0
        max_x = None

        for c in range(self.C+1):
            self.eval_c = c
            acq0 = self.acq(x0)
            x0_c = x0[acq0 != 0]

            if np.size(x0_c) != 0:
                for i in range(np.shape(x0_c)[0]):
                    res = scipyminimize(self.acq, x0=x0_c[i], bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-3})
                    if max_acq < - res['fun'] / self.cost[c] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                        max_acq = - res['fun'] / self.cost[c]
                        max_x = res['x']
                        max_c = c

        if max_x is None:
            if self.max_inputs is not None:
                print('select by Thompson Sampling')
                max_x = self.max_inputs[0]
                inputs = [np.atleast_2d(max_x) for c in range(self.C+1)]
            else:
                inputs = self.uncertainty_sampling()
                print('select by Uncertainty Sampling')
        else:
            self.eval_c = max_c
            res = scipyminimize(self.acq, x0=max_x, bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-16})
            if max_acq < - res['fun'] / self.cost[max_c] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                max_acq = - res['fun'] / self.cost[max_c]
                max_x = res['x']
            inputs = [np.array([]).reshape((0, self.input_dim)) for c in range(self.C+1)]
            inputs[max_c] = np.atleast_2d(max_x)
        return inputs



    def acq(self, x):
        x = np.atleast_2d(x)




        gammas_c = list()
        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]])
            gammas_c.append(( (self.thresholds[c] - mean) / np.sqrt(var)).ravel())

        gammas_c = np.array(gammas_c)
        survival_funcs_c = 1 - norm.cdf(gammas_c)

        Z_star = np.prod(survival_funcs_c, axis=0)




        mean, var = self.GPmodel.predict_noiseless(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]])
        gamma_f = (self.max_samples - mean) / np.sqrt(var)
        survival_funcs_f = 1 - norm.cdf(gamma_f)

        Z_star = np.c_[Z_star] * survival_funcs_f

        if self.eval_c == 0:

            index_f = survival_funcs_f==0
            survival_funcs_f[index_f] = 1

            gamma_f[np.isinf(gamma_f)] = 0
            xpdfx_cdfx = gamma_f * norm.pdf(gamma_f) / survival_funcs_f
            xpdfx_cdfx[index_f] = gamma_f[index_f]**2
            one_minus_cdf = survival_funcs_f
        else:

            index_c = survival_funcs_c[self.eval_c-1,:]==0
            survival_funcs_c[self.eval_c-1,:][index_c] = 1
            xpdfx_cdfx = gammas_c[self.eval_c-1,:] * norm.pdf(gammas_c[self.eval_c-1,:]) / survival_funcs_c[self.eval_c-1,:]
            xpdfx_cdfx[index_c] = gammas_c[self.eval_c-1,:][index_c]**2
            xpdfx_cdfx = np.c_[xpdfx_cdfx]
            one_minus_cdf = np.c_[survival_funcs_c[self.eval_c-1,:]]


        Z_star[Z_star>=1] = 1 - 1e-16
        acq = Z_star / (2*(1-Z_star)) * xpdfx_cdfx - np.log(1-Z_star)


        Z_star[Z_star >= one_minus_cdf] = 0
        acq = acq + (one_minus_cdf - Z_star) / (1 - Z_star) * np.log((one_minus_cdf - Z_star) / one_minus_cdf)
        acq = np.sum(acq, axis=1)
        return - acq


class ConstrainedMaxvalueEntropySearch_dec_KL(ConstrainedMaxvalueEntropySearch):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, sampling_num=10, sampling_approach='inf', cost=None, GPmodel=None, pool_X=None, optimize=True, kernel_name='linear+rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = sampling_num
        self.pool_X = pool_X
        self.sampling_approach = sampling_approach
        self.maximum_sampling()


    def maximum_sampling(self):
        start = time.time()
        self.max_samples, self.max_inputs = self.sampling_RFM(pool_X=self.pool_X, sampling_approach=self.sampling_approach)
        print('max_sample time:', time.time() - start)

        self.max_inputs = np.atleast_2d([self.max_inputs[i] for i in range(self.sampling_num) if self.max_inputs[i] is not None])
        self.max_samples = np.c_[self.max_samples].T
        print('sampled maximums:', self.max_samples)
        if self.input_dim <= 10:
            print('sampled_max-inputs:', self.max_inputs.tolist())

        if np.size(self.max_inputs) == 0:
            self.max_inputs = None

    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list=add_X_list, add_Y_list=add_Y_list, optimize=optimize)
        self.maximum_sampling()

    def next_input(self):
        NUM_START = 1000
        x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        if self.x_max is not None:
            x0 = np.r_[x0, np.c_[self.x_max].T]
        if self.max_inputs is not None:
            x0 = np.r_[x0, self.max_inputs]
        max_acq = 0
        max_x = None

        for c in range(self.C+1):
            self.eval_c = c
            acq0 = self.acq(x0)
            x0_c = x0[acq0 != 0]

            if np.size(x0_c) != 0:
                for i in range(np.shape(x0_c)[0]):
                    res = scipyminimize(self.acq, x0=x0_c[i], bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-3})
                    if max_acq < - res['fun'] / self.cost[c] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                        max_acq = - res['fun'] / self.cost[c]
                        max_x = res['x']
                        max_c = c


        if max_x is None:
            if self.max_inputs is not None:
                print('select by Thompson Sampling')
                max_x = self.max_inputs[0]
                inputs = [np.atleast_2d(max_x) for c in range(self.C+1)]
            else:
                inputs = self.uncertainty_sampling()
                print('select by Uncertainty Sampling')
        else:
            self.eval_c = max_c
            res = scipyminimize(self.acq, x0=max_x, bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-16})
            if max_acq < - res['fun'] / self.cost[max_c] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                max_acq = - res['fun'] / self.cost[max_c]
                max_x = res['x']
            inputs = [np.array([]).reshape((0, self.input_dim)) for c in range(self.C+1)]
            inputs[max_c] = np.atleast_2d(max_x)
        return inputs



    def acq(self, x):
        x = np.atleast_2d(x)




        gammas_c = list()
        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]])
            gammas_c.append(( (self.thresholds[c] - mean) / np.sqrt(var)).ravel())

        gammas_c = np.array(gammas_c)
        survival_funcs_c = 1 - norm.cdf(gammas_c)

        Z_star = np.prod(survival_funcs_c, axis=0)




        mean, var = self.GPmodel.predict_noiseless(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]])
        gamma_f = (self.max_samples - mean) / np.sqrt(var)
        survival_funcs_f = 1 - norm.cdf(gamma_f)

        Z_star = np.c_[Z_star] * survival_funcs_f

        if self.eval_c == 0:

            index_f = survival_funcs_f==0
            survival_funcs_f[index_f] = 1
            one_minus_cdf = survival_funcs_f





        else:

            index_c = survival_funcs_c[self.eval_c-1,:]==0
            survival_funcs_c[self.eval_c-1,:][index_c] = 1
            one_minus_cdf = np.c_[survival_funcs_c[self.eval_c-1,:]]






        Z_star[Z_star>=1] = 1 - 1e-16






        acq = - np.log(1-Z_star)


        Z_star[Z_star >= one_minus_cdf] = 0
        acq = acq + (one_minus_cdf - Z_star) / (1 - Z_star) * np.log((one_minus_cdf - Z_star) / one_minus_cdf)
        acq = np.sum(acq, axis=1)
        return - acq
