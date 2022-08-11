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
import nlopt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../myutils"))
import myutils as utils
from BO_core import ConstrainedBO_core

class PCBO(ConstrainedBO_core):
    __metaclass__ = ABCMeta

    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, cost=None, GPmodel=None, optimize=True, kernel_name='rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.preprocessing_time = 0
        self.num_worker = num_worker
        self.selected_inputs_list = selected_inputs_list
        self.make_selected_inputs_array()
        self.TS_index = 0

    def make_selected_inputs_array(self):
        if np.size(self.selected_inputs_list) > 0:
            eval_num = [np.shape(inputs)[0] for inputs in self.selected_inputs_list]
            output_indexes = np.hstack([i*np.ones(eval_num[i]) for i in range(len(eval_num))])
            self.selected_inputs = np.c_[np.vstack([inputs for inputs in self.selected_inputs_list if np.size(inputs) > 0]), output_indexes]

    @abstractmethod
    def para_acq(self, x):
        pass

    @abstractmethod
    def para_acq_correlated(self, x):
        pass

    @abstractmethod
    def preparation(self):
        pass

    def next_input_pool(self, pool_X):
        num_remain_worker = self.num_worker - np.shape(self.selected_inputs_list[0])[0]

        new_inputs_list = [np.array([]).reshape((0, self.input_dim)) for c in range(self.C+1)]
        X = pool_X.copy()
        for q in range(num_remain_worker):
            print(q, 'th selection-------------')
            self.q = q
            if np.shape(self.selected_inputs_list[0])[0] == 0 and q == 0:
                if self.model=='independent':
                    acquisition_values = self.acq(X)
                elif self.model=='correlated':
                    acquisition_values = self.acq_correlated(X)

                if np.all(acquisition_values == 0):
                    print('all acquisition function value is zero')
                    max_index = np.random.randint(0, np.shape(X)[0], 1)
                else:
                    max_index = np.argmin(acquisition_values)

                temp_new_inputs = [np.atleast_2d(X[max_index]) for c in range(self.C+1)]
                X = np.delete(X, max_index, axis=0)
                self.selected_inputs_list = [np.r_[self.selected_inputs_list[c], temp_new_inputs[c]] for c in range(self.C+1)]
                self.make_selected_inputs_array()
            else:
                self.preparation()
                if self.model=='independent':
                    acquisition_values = self.para_acq(X)
                elif self.model=='correlated':
                    acquisition_values = self.para_acq_correlated(X)

                if np.all(acquisition_values == 0):
                    print('all acquisition function value is zero')
                    max_index = np.random.randint(0, np.shape(X)[0], 1)
                else:
                    max_index = np.argmin(acquisition_values)

                temp_new_inputs = [np.atleast_2d(X[max_index]) for c in range(self.C+1)]
                X = np.delete(X, max_index, axis=0)
                self.selected_inputs_list = [np.r_[self.selected_inputs_list[c], temp_new_inputs[c]] for c in range(self.C+1)]
                self.make_selected_inputs_array()
        new_inputs_list = [ inputs[-num_remain_worker:, :] for inputs in self.selected_inputs_list]
        return new_inputs_list, X


    def next_input(self):
        num_remain_worker = self.num_worker - np.shape(self.selected_inputs_list[0])[0]

        new_inputs_list = [np.array([]).reshape((0, self.input_dim)) for c in range(self.C+1)]
        for q in range(num_remain_worker):
            print(q, 'th selection-------------')
            self.q = q
            if np.shape(self.selected_inputs_list[0])[0] == 0 and q == 0:
                if self.model=='independent':
                    temp_new_inputs = self.select_one_input(self.acq)
                elif self.model=='correlated':
                    temp_new_inputs = self.select_one_input(self.acq_correlated)
                self.selected_inputs_list = [np.r_[self.selected_inputs_list[c], temp_new_inputs[c]] for c in range(self.C+1)]
                self.make_selected_inputs_array()
            else:
                self.preparation()
                if self.model=='independent':
                    temp_new_inputs = self.select_one_input(self.para_acq)
                elif self.model=='correlated':
                    temp_new_inputs = self.select_one_input(self.para_acq_correlated)
                self.selected_inputs_list = [np.r_[self.selected_inputs_list[c], temp_new_inputs[c]] for c in range(self.C+1)]
                self.make_selected_inputs_array()
        new_inputs_list = [ inputs[-num_remain_worker:, :] for inputs in self.selected_inputs_list]
        self.TS_index = 0
        return new_inputs_list


    def select_one_input(self, acq_func):
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
            upper_bound = mean.reshape((np.shape(x0)[0], self.C+1)) + 5 * var.reshape((np.shape(x0)[0], self.C+1))
            x0 = x0[np.any(upper_bound > np.r_[[np.min(self.max_samples)], self.thresholds], axis=1)]


        if np.size(x0) != 0:
            for i in range(np.shape(x0)[0]):
                res = scipyminimize(acq_func, x0=x0[i], bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-3})
                if max_acq < - res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):









                    max_acq = - res['fun']
                    max_x = res['x']

        if max_x is None:
            if self.max_inputs is not None and np.shape(self.max_inputs)[0] > self.TS_index:
                print('select by Thompson Sampling')
                max_x = self.max_inputs[self.TS_index]
                self.TS_index += 1
                inputs = [np.atleast_2d(max_x) for c in range(self.C+1)]
            else:
                print('select by Uncertainty Sampling')
                inputs = self.uncertainty_sampling()


        else:
            res = scipyminimize(acq_func, x0=max_x, bounds=self.bounds_list, method="L-BFGS-B", options={'ftol': 1e-16})
            if max_acq < - res['fun'] and np.all(res['x'] >= self.bounds[0]) and np.all(res['x'] <= self.bounds[1]):
                max_acq = - res['fun']
                max_x = res['x']
            inputs = [np.atleast_2d(max_x) for c in range(self.C+1)]

        return inputs

    def uncertainty_sampling(self):
        NUM_START = 100
        x0 = utils.lhs(self.input_dim, samples=NUM_START, criterion='maximin') * (self.bounds[1]- self.bounds[0]) + self.bounds[0]
        max_acq = -np.inf
        max_x = None

        if np.size(np.vstack(self.selected_inputs_list)) == 0:
            def uncertainty(x):
                x = np.atleast_2d(x)
                _, var = self.GPmodel.predict_noiseless(np.c_[np.matlib.repmat(x, self.C+1, 1), np.c_[np.arange(self.C+1)]])
                entropy = np.sum(np.log(var))
                return - entropy.ravel()[0]
        else:
            def uncertainty(x):
                x = np.atleast_2d(x)
                X = np.c_[np.matlib.repmat(x, self.C+1, 1), np.c_[np.arange(self.C+1)]]
                _, var = self.GPmodel.predict_noiseless(X)

                for c in range(self.C+1):
                    cov_x_selected = self.GPmodel.posterior_covariance_between_points(np.atleast_2d(X[c]), self.selected_inputs[self.selected_inputs[:,-1]==c])
                    temp = cov_x_selected.dot(self.selected_cov_inv[c])
                    var[c] = var[c] - np.c_[np.einsum('ij,ji->i', temp, cov_x_selected.T)]
                entropy = np.sum(np.log(var))
                return - entropy.ravel()[0]

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



class ParallelConstrainedExpectedImprovement(PCBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, sampling_num=10, cost=None, GPmodel=None, optimize=True, kernel_name='rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = sampling_num
        self.M = np.min(Y_list[0])


    def update(self, add_X_list, add_Y_list, optimize=False):
        super().update(add_X_list=add_X_list, add_Y_list=add_Y_list, optimize=optimize)
        self.M = np.min([self.M, np.min(add_Y_list)])
        self.selected_inputs_list = [np.array([]).reshape((0, self.GPmodel.input_dim)) for c in range(self.C+1)]
        self.make_selected_inputs_array()

    def preparation(self):
        mean, cov = self.GPmodel.predict(self.selected_inputs, full_cov=True)




        temp_size = np.shape(self.selected_inputs_list[0])[0]
        self.selected_mean_list = [mean[c*temp_size:(c+1)*temp_size, :] for c in range(self.C+1)]
        self.selected_cov_inv = [ np.linalg.inv(cov[c*temp_size:(c+1)*temp_size, c*temp_size:(c+1)*temp_size]) for c in range(self.C+1)]

        try:
            cov_chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            print('Cholesky decomposition in PECI preparation output error, thus add the 1e-3 to diagonal elements.')
            cov = cov + 1e-3 * np.eye(np.shape(cov)[0])
            cov_chol = np.linalg.cholesky(cov)

        self.montecarlo_samples = mean + cov_chol.dot(np.random.normal(size=(self.selected_inputs.shape[0], self.sampling_num)))

        self.montecarlo_samples = self.montecarlo_samples.reshape([self.C+1, np.shape(self.selected_inputs_list[0])[0], self.sampling_num])
        feasible_flag = np.all(self.montecarlo_samples[1:,:,:] > self.thresholds[:,np.newaxis, np.newaxis], axis=0)

        temp_ys = self.montecarlo_samples[0].copy()
        temp_ys[np.logical_not(feasible_flag)] = -np.inf
        temp_ymax = np.max(temp_ys, axis=0)
        if self.y_max is not None:
            self.additional_ymax = self.y_max * np.ones(self.sampling_num)
            temp_ymax[self.additional_ymax > temp_ymax] = self.additional_ymax[self.additional_ymax > temp_ymax]
            self.additional_ymax = temp_ymax

        else:
            self.additional_ymax = temp_ymax




    def acq(self, x):
        x = np.atleast_2d(x)

        mean, var = self.GPmodel.predict_noiseless(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]])
        if self.y_max is not None:
            std = np.sqrt(var)
            Z = (mean - self.y_max) / std
            acq = ((Z * std)*norm.cdf(Z) + std*norm.pdf(Z)).ravel()
        else:
            acq = mean.ravel() - self.M

        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]])
            prob = 1 - norm.cdf((self.thresholds[c] - mean) / np.sqrt(var)).ravel()
            acq = acq * prob
        return - acq


    def para_acq(self, x):
        x = np.atleast_2d(x)
        acq = np.zeros((np.shape(x)[0], self.sampling_num))

        mean, var = self.GPmodel.predict_noiseless(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]])
        cov_x_selected = self.GPmodel.posterior_covariance_between_points(np.c_[x, np.c_[np.zeros(np.shape(x)[0])]], np.c_[self.selected_inputs_list[0], np.c_[np.zeros(np.shape(self.selected_inputs_list[0])[0])]])
        temp = cov_x_selected.dot(self.selected_cov_inv[0])
        mean = mean + temp.dot(self.montecarlo_samples[0] - self.selected_mean_list[0])
        var = var - np.c_[np.einsum('ij,ji->i', temp, cov_x_selected.T)]

        std = np.sqrt(var)
        Z = (mean - self.additional_ymax) / std
        index = np.logical_not(np.isinf(self.additional_ymax))

        if np.any(index):
            acq[:, index] = (Z[:,index] * std)*norm.cdf(Z[:,index]) + std*norm.pdf(Z[:,index])
        index = np.logical_not(index)
        if np.any(index):
            acq[:, index] = mean[:, index] - self.M


        for c in range(self.C):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]])
            cov_x_selected = self.GPmodel.posterior_covariance_between_points(np.c_[x, (c+1)*np.c_[np.ones(np.shape(x)[0])]], np.c_[self.selected_inputs_list[c+1], (c+1)*np.c_[np.ones(np.shape(self.selected_inputs_list[c+1])[0])]])
            temp = cov_x_selected.dot(self.selected_cov_inv[c+1])
            mean = mean + temp.dot(self.montecarlo_samples[c+1] - self.selected_mean_list[c+1])
            var = var - np.c_[np.einsum('ij,ji->i', temp, cov_x_selected.T)]

            prob = 1 - norm.cdf((self.thresholds[c] - mean) / np.sqrt(var))
            acq = acq * prob
        acq = np.mean(acq, axis=1)
        return - acq

class ParallelConstrainedThompsonSampling(PCBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, cost=None, GPmodel=None, optimize=True, kernel_name='rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
        self.sampling_num = num_worker - np.shape(selected_inputs_list[0])[0]

    def next_input(self):
        _, inputs = self.sampling_RFM(sampling_approach='MinViolation')
        inputs = np.vstack(inputs)
        inputs = [inputs for c in range(self.C+1)]
        return inputs


    def next_input_pool(self, pool_X):
        _, inputs = self.sampling_RFM(pool_X=pool_X, sampling_approach='MinViolation')
        inputs = np.vstack(inputs)
        for i in range(np.shape(inputs)[0]):
            index = np.all(pool_X == np.atleast_2d(inputs[i]), axis=1)
            pool_X = pool_X[np.logical_not(index)]
        inputs = [inputs for c in range(self.C+1)]
        return inputs, pool_X



class ParallelConstrainedMaxvalueEntropySearch(PCBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, sampling_num=10, sampling_approach='inf', cost=None, GPmodel=None, pool_X=None, optimize=True, kernel_name='rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
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
        self.selected_inputs_list = [np.array([]).reshape((0, self.GPmodel.input_dim)) for c in range(self.C+1)]
        self.make_selected_inputs_array()


    def preparation(self):
        if self.model=='independent':
            self.selected_mean = list()
            self.selected_cov_inv = list()

            self.montecarlo_samples = self.sample_path(self.selected_inputs_list)

            for c in range(self.C + 1):
                mean, cov = self.GPmodel.predict(np.c_[self.selected_inputs_list[c], c*np.c_[np.ones(np.shape(self.selected_inputs_list[c])[0])]], full_cov=True)
                cov_inv = np.linalg.inv(cov)
                self.selected_mean.append(mean)
                self.selected_cov_inv.append(cov_inv)


                if c==1:
                    feasible_flag = self.montecarlo_samples[c] > self.thresholds[c-1]
                elif c >= 2:
                    feasible_flag = (self.montecarlo_samples[c] > self.thresholds[c-1]) & feasible_flag

            temp_ys = self.montecarlo_samples[0].copy()
            temp_ys[np.logical_not(feasible_flag)] = -np.inf
            temp_ymax = np.atleast_2d(np.max(temp_ys, axis=0))
            self.max_samples[self.max_samples < temp_ymax] = temp_ymax[self.max_samples < temp_ymax]

        elif self.model=='correlated':
            return None


    def para_acq(self, x):
        x = np.atleast_2d(x)

        gammas = list()
        for c in range(self.C+1):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, c*np.c_[np.ones(np.shape(x)[0])]])
            cov_x_selected = self.GPmodel.posterior_covariance_between_points(np.c_[x, c*np.c_[np.ones(np.shape(x)[0])]], np.c_[self.selected_inputs_list[c], c*np.c_[np.ones(np.shape(self.selected_inputs_list[c])[0])]])
            temp = cov_x_selected.dot(self.selected_cov_inv[c])
            mean = mean + temp.dot(self.montecarlo_samples[c])
            var = var - np.c_[np.einsum('ij,ji->i', temp, cov_x_selected.T)]

            if c==0:
                gammas.append((self.max_samples - mean) / np.sqrt(var))
            else:
                gammas.append((self.thresholds[c-1] - mean) / np.sqrt(var))

        gammas = np.vstack(gammas).reshape(self.C+1, np.shape(x)[0], self.sampling_num)
        survival_funcs = 1 - norm.cdf(gammas)

        Z_star = np.prod(survival_funcs, axis=0)


        index = survival_funcs==0
        survival_funcs[index] = 1

        gammas[np.isinf(gammas)] = 0
        inner_sum = gammas * norm.pdf(gammas) / survival_funcs

        inner_sum[index] = gammas[index]**2
        inner_sum = np.sum(inner_sum, axis=0)


        Z_star[Z_star==1] = 1 - 1e-16

        acq = np.sum(Z_star / (2* (1 - Z_star)) * inner_sum - np.log(1-Z_star), axis=1)
        return - acq



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


class ParallelConstrainedMaxvalueEntropySearch_LB(PCBO):
    def __init__(self, X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, sampling_num=10, sampling_approach='inf', cost=None, GPmodel=None, pool_X=None, optimize=True, kernel_name='rbf', model='independent'):
        super().__init__(X_list, Y_list, bounds, kernel_bounds, C, thresholds, num_worker, selected_inputs_list, cost=cost, GPmodel=GPmodel, optimize=optimize, kernel_name=kernel_name, model=model)
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
        self.selected_inputs_list = [np.array([]).reshape((0, self.GPmodel.input_dim)) for c in range(self.C+1)]
        self.make_selected_inputs_array()


    def preparation(self):
        if self.model=='independent':
            self.selected_mean = list()
            self.selected_cov_inv = list()

            self.montecarlo_samples = self.sample_path(self.selected_inputs_list)

            for c in range(self.C + 1):
                mean, cov = self.GPmodel.predict(np.c_[self.selected_inputs_list[c], c*np.c_[np.ones(np.shape(self.selected_inputs_list[c])[0])]], full_cov=True)
                cov_inv = np.linalg.inv(cov)
                self.selected_mean.append(mean)
                self.selected_cov_inv.append(cov_inv)


                if c==1:
                    feasible_flag = self.montecarlo_samples[c] > self.thresholds[c-1]
                elif c >= 2:
                    feasible_flag = (self.montecarlo_samples[c] > self.thresholds[c-1]) & feasible_flag

            temp_ys = self.montecarlo_samples[0].copy()
            temp_ys[np.logical_not(feasible_flag)] = -np.inf
            temp_ymax = np.atleast_2d(np.max(temp_ys, axis=0))
            self.max_samples[self.max_samples < temp_ymax] = temp_ymax[self.max_samples < temp_ymax]

        elif self.model=='correlated':
            return None

    def para_acq(self, x):
        x = np.atleast_2d(x)

        gammas = list()
        for c in range(self.C+1):
            mean, var = self.GPmodel.predict_noiseless(np.c_[x, c*np.c_[np.ones(np.shape(x)[0])]])
            cov_x_selected = self.GPmodel.posterior_covariance_between_points(np.c_[x, c*np.c_[np.ones(np.shape(x)[0])]], np.c_[self.selected_inputs_list[c], c*np.c_[np.ones(np.shape(self.selected_inputs_list[c])[0])]])
            temp = cov_x_selected.dot(self.selected_cov_inv[c])
            mean = mean + temp.dot(self.montecarlo_samples[c])
            var = var - np.c_[np.einsum('ij,ji->i', temp, cov_x_selected.T)]

            if c==0:
                gammas.append((self.max_samples - mean) / np.sqrt(var))
            else:
                gammas.append((self.thresholds[c-1] - mean) / np.sqrt(var))

        gammas = np.vstack(gammas).reshape(self.C+1, np.shape(x)[0], self.sampling_num)
        survival_funcs = 1 - norm.cdf(gammas)

        Z_star = np.prod(survival_funcs, axis=0)


        Z_star[Z_star==1] = 1 - 1e-16

        acq = - np.sum(np.log(1-Z_star), axis=1)
        return - acq


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
        Z_star = list()

        for max_sample in range(np.size(self.max_samples)):
            Z_star.append(mvn.mvnun(np.r_[[max_sample], self.thresholds], np.inf*np.ones(self.C+1), mean, cov, maxpts=(self.C+1)*1e4, abseps=1e-8, releps=1e-6)[0])





        Z_star = np.array(Z_star)

        Z_star[Z_star==1] = 1 - 1e-16

        acq = - np.sum(np.log(1 - Z_star))
        return - acq

































































































































































































































































































































































































































