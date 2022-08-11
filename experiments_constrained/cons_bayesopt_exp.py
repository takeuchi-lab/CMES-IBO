# -*- coding: utf-8 -*-
import os
import sys
import signal
import glob
import time
import random
import concurrent.futures
import pickle
import datetime
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import GPy



sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ConstrainedBO"))
import constrained_bayesian_opt as CBO
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ConstrainedBO"))
import parallel_constrained_bayesian_opt as PCBO

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_functions"))
import test_functions

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../myutils"))
import myutils

signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(params):
    (func_name, BO_method, function_seed, initial_seed, parallel_num) = params
    print(params)
    # os.system('echo $OMP_NUM_THREADS')

    # settings for test functions
    test_func = eval('test_functions.'+func_name)()
    if 'SynFun' in func_name:
        test_func.func_sampling(seed=function_seed)
    C = test_func.C
    thresholds = test_func.g_thresholds
    bounds = test_func.bounds
    input_dim = test_func.d
    interval_size = bounds[1] - bounds[0]
    kernel_bounds = np.array([interval_size / 10., interval_size * 10])

    if 'LLTO' in func_name:
        median_X = 1.1405376650072867
        kernel_bounds = np.array([median_X*1e-2*np.ones(input_dim), median_X*1e2*np.ones(input_dim)])
    if 'Bi2O3' in func_name:
        median_X = 44.0
        kernel_bounds = np.array([median_X*1e-2*np.ones(input_dim), median_X*1e2*np.ones(input_dim)])

    if 'cnn' in func_name:
        kernel_bounds = np.array([0.1 * np.ones(input_dim), 2. * np.ones(input_dim)])

    if func_name=='Bi2O3' or func_name=='LLTO' or 'pool' in func_name or 'cnn' in func_name:
        X_all = test_func.X
        pool_X = X_all.copy()
    else:
        X_all = None
        pool_X = None

    # for extend cost setting
    cost = np.ones(C+1)
    coupled_cost = np.sum(cost)

    # for extend correlated setting
    if 'corr' in BO_method:
        model = 'independent'
    else:
        model = 'independent'

    # set the seed
    np.random.seed(initial_seed)
    random.seed(initial_seed)

    # Latin Hypercube Sampling
    if 'SynFun' in func_name:
        save_cost = 5 * (C+1)
        if input_dim == 2:
            FIRST_NUM = 3
            save_cost = (C+1)
        elif input_dim > 2:
            FIRST_NUM = 5 * input_dim
        if 'pool' in func_name or 'test' in func_name:
            ITR_MAX = 100 - FIRST_NUM
        else:
            ITR_MAX = 30
    elif 'Bi2O3' in func_name:
        FIRST_NUM = 5
        # FIRST_NUM = 335
        ITR_MAX = 80
        save_cost = (C+1)
    elif 'LLTO' in func_name:
        FIRST_NUM = 5
        # FIRST_NUM = 1118
        ITR_MAX = 100
        save_cost = (C+1)
    elif 'cnn' in func_name:
        FIRST_NUM = 5
        # FIRST_NUM = 1118
        ITR_MAX = 300
        save_cost = (C+1)
    elif 'Test_MESC' in func_name:
        FIRST_NUM = 3
        ITR_MAX = 30
        save_cost = C+1
    elif input_dim == 2:
        FIRST_NUM = 5
        ITR_MAX = 30
        save_cost = C+1
    elif input_dim > 2:
        if 'HartMann6' in func_name:
            FIRST_NUM = 5 * input_dim
            ITR_MAX = 50*input_dim - FIRST_NUM
        elif re.match('G\d+', func_name):
            FIRST_NUM = 10
            ITR_MAX = 300 - FIRST_NUM
        else:
            FIRST_NUM = np.min([5 * input_dim, 25])
            # FIRST_NUM = 10
            ITR_MAX = 100
        save_cost = 5 * (C+1)

    if 'dec' in BO_method:
        ITR_MAX = ITR_MAX * (C+1)
    ITR_MAX += 1
    # COST_MAX = 50 * (C+1) * input_dim
    COST_MAX = np.inf

    lower_p = 0.95

    if X_all is None:
        training_input = myutils.initial_design(FIRST_NUM, input_dim, bounds)
        training_input = [training_input for c in range(C+1)]
        training_output = test_func.mf_values(training_input)
    else:
        training_input, pool_X = myutils.initial_design_pool_random(FIRST_NUM, input_dim, bounds, pool_X)
        training_input = [training_input for c in range(C+1)]
        training_output = test_func.mf_values(training_input)
    eval_num = FIRST_NUM * np.ones(C+1)

    gp_model = None
    if 'SynFun' in func_name:
        kernel_name = 'rbf'
    else:
        # kernel_name = 'rbf'
        kernel_name = 'linear+rbf'


    # if test function is synthetic, add parameter ell information
    if 'SynFun' in func_name:
        optimize=False

        func_name = func_name+'_ell='
        if 'pool' in func_name or 'test' in func_name:
            func_name = func_name + '_' + str(test_func.ell)
            test_func.ell = test_func.ell * np.ones(C+1)
        # elif not('corr' in func_name):
        #     for c in range(C+1):
        #         func_name = func_name + '_' + str(test_func.ell[c])
        else:
            func_name = func_name + '_' + str(test_func.ell)
            test_func.ell = test_func.ell * np.ones(C+1)
        func_name = func_name +'-d='+str(test_func.d)+'-seed'+str(test_func.seed)
        if model == 'independent':
            gp_model = myutils.GPy_independent_model(training_input, training_output, kernel_name=kernel_name, normalizer=False)
            if kernel_name != 'rbf':
                print('Synthetic function is drawn only using rbf kernel, but specified kernel is', kernel_name)
                exit()
            else:
                for c in range(C+1):
                    gp_model.model_list[c]['.*rbf.variance'].constrain_fixed(1)
                    gp_model.model_list[c]['.*rbf.lengthscale'].constrain_fixed(test_func.ell[c])
        elif model == 'correlated':
            gp_model = myutils.GPy_correlated_model(training_input, training_output, kernel_name=kernel_name, normalizer=False)
            if kernel_name != 'rbf':
                print('Synthetic function is drawn only using rbf kernel, but specified kernel is', kernel_name)
                exit()
            else:
                gp_model['.*mul.rbf.lengthscale'].constrain_fixed(test_func.ell[0])
                gp_model['.*mul.coregion.W'].constrain_fixed(test_func.w_1.T)
                gp_model['.*mul.coregion.kappa'].constrain_fixed(test_func.kappa_1)

                gp_model['.*mul_1.rbf.lengthscale'].constrain_fixed(test_func.ell[0])
                gp_model['.*mul_1.coregion.W'].constrain_fixed(test_func.w_2.T)
                gp_model['.*mul_1.coregion.kappa'].constrain_fixed(test_func.kappa_2)


        else:
            print('this GPmodel is not implemented', model)
            exit(1)
    elif 'Bi2O3' in func_name or 'LLTO' in func_name:
        optimize=True
        if model == 'independent':
            if 'Bi2O3' in func_name:
                gp_model = myutils.GPy_independent_model(training_input, training_output, kernel_name=kernel_name, normalizer=True, ARD=True)
            elif 'LLTO' in func_name:
                gp_model = myutils.GPy_independent_model(training_input, training_output, kernel_name=kernel_name, normalizer=True, ARD=False)
            if not(optimize):
                if 'Bi2O3' in func_name:
                    rbf_var = [0.30885155, 0.55114414]
                    linear_var = [0.99999997, 0.19355267]
                    rbf_lengthscales = [[0.49962045,0.56838710, 0.20763265], [0.67291040,0.70293306, 0.01267357]]
                    gp_model.model_list[-1]['.*sum.rbf.variance'].constrain_fixed(rbf_var[c])
                    gp_model.model_list[-1]['.*sum.linear.variances'].constrain_fixed(linear_var[c])
                    gp_model.model_list[-1]['.*sum.rbf.lengthscale'].constrain_fixed(rbf_lengthscales[c])
                elif 'LLTO' in func_name:
                    rbf_var = [0.46577895,  0.10100402]
                    linear_var = [0.99999999, 1.00000000]
                    rbf_lengthscales = [0.00003638, 0.30184647]
                    gp_model.model_list[-1]['.*sum.rbf.variance'].constrain_fixed(rbf_var[c])
                    gp_model.model_list[-1]['.*sum.linear.variances'].constrain_fixed(linear_var[c])
                    gp_model.model_list[-1]['.*sum.rbf.lengthscale'].constrain_fixed(rbf_lengthscales[c])
            else:
                gp_model.set_hyperparameters_bounds(kernel_bounds)
    else:
        optimize=True

    if '1' in BO_method:
        NUM_SAMPLING = 1
    elif '50' in BO_method:
        NUM_SAMPLING = 50
    else:
        NUM_SAMPLING = 10

    if parallel_num > 1:
        results_path = func_name+'_results/'+BO_method+'_Q='+str(parallel_num)+'/seed='+str(initial_seed)+'/'
    else:
        results_path = func_name+'_results/'+BO_method+'/seed='+str(initial_seed)+'/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(results_path+'optimizer_log/'):
        os.makedirs(results_path+'optimizer_log/')
    cost_list = list()
    InfReg_list = list()
    SimReg_list = list()
    inference_point = None
    previous_cost = 0
    current_cost = np.sum(eval_num * cost)
    remain_cost = [[] for c in range(C+1)]
    selected_inputs_list = [np.array([]).reshape((0, input_dim)) for c in range(C+1)]
    num_worker = parallel_num

    regression_output = training_output.copy()

    # bayesian optimizer
    if 'PECI' in BO_method:
        optimizer = PCBO.ParallelConstrainedExpectedImprovement(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, num_worker=parallel_num, selected_inputs_list=selected_inputs_list, sampling_num = NUM_SAMPLING, cost=cost, GPmodel=gp_model, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'ECI' in BO_method:
        optimizer = CBO.ConstrainedExpectedImprovement(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, cost=cost, GPmodel=gp_model, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'US' in BO_method:
        optimizer = CBO.UncertaintySampling(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, cost=cost, GPmodel=gp_model, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'PTSC' in BO_method:
        optimizer = PCBO.ParallelConstrainedThompsonSampling(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, num_worker=parallel_num, selected_inputs_list=selected_inputs_list, cost=cost, GPmodel=gp_model, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'TSC' in BO_method:
        optimizer = CBO.ConstrainedThompsonSampling(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, GPmodel=gp_model, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'PMESC_LB' in BO_method:
        optimizer = PCBO.ParallelConstrainedMaxvalueEntropySearch_LB(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, num_worker=parallel_num, selected_inputs_list=selected_inputs_list, sampling_num = NUM_SAMPLING, cost=cost, GPmodel=gp_model, pool_X=X_all, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'PMESC' in BO_method:
        optimizer = PCBO.ParallelConstrainedMaxvalueEntropySearch(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, num_worker=parallel_num, selected_inputs_list=selected_inputs_list, sampling_num = NUM_SAMPLING, cost=cost, GPmodel=gp_model, pool_X=X_all, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'MESC_LB' in BO_method:
        optimizer = CBO.ConstrainedMaxvalueEntropySearch_LB(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, sampling_num = NUM_SAMPLING, GPmodel=gp_model, pool_X=X_all, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'MESC_dec_KL' in BO_method:
        optimizer = CBO.ConstrainedMaxvalueEntropySearch_dec_KL(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, sampling_num = NUM_SAMPLING, GPmodel=gp_model, pool_X=X_all, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'MESC_dec' in BO_method:
        optimizer = CBO.ConstrainedMaxvalueEntropySearch_dec(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, sampling_num = NUM_SAMPLING, GPmodel=gp_model, pool_X=X_all, optimize=optimize, kernel_name=kernel_name, model=model)
    elif 'MESC' in BO_method:
        optimizer = CBO.ConstrainedMaxvalueEntropySearch(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, sampling_num = NUM_SAMPLING, GPmodel=gp_model, pool_X=X_all, optimize=optimize, kernel_name=kernel_name, model=model)
    # elif 'MESC_seqdec_KL' in BO_method:
    #     optimizer = CBO.ConstrainedMaxvalueEntropySearch_seqdec_KL(X_list=training_input, Y_list=regression_output, bounds = bounds, kernel_bounds=kernel_bounds, C=C, thresholds=thresholds, sampling_num = NUM_SAMPLING, GPmodel=gp_model, pool_X=X_all, optimize=optimize, kernel_name=kernel_name, model=model)
    else:
        print('not implemented')
        exit(1)

    # print the parameters of GPs
    # print(optimizer.GPmodel.means, optimizer.GPmodel.stds)
    if model=='independent':
        print(optimizer.GPmodel.model_list[0]['.*Gaussian_noise.variance'])
        if kernel_name=='rbf':
            for c in range(C+1):
                print(optimizer.GPmodel.model_list[c]['.*rbf.variance'])
                print(optimizer.GPmodel.model_list[c]['.*rbf.lengthscale'])
        elif kernel_name=='linear+rbf':
            for c in range(C+1):
                print(optimizer.GPmodel.model_list[c]['.*sum.linear.variances'])
                print(optimizer.GPmodel.model_list[c]['.*sum.rbf.variance'])
                print(optimizer.GPmodel.model_list[c]['.*sum.rbf.lengthscale'])
    elif model=='correlated':
        # print(optimizer.GPmodel)
        print(optimizer.GPmodel['.*Gaussian_noise.variance'])
        if kernel_name=='rbf':
            print(optimizer.GPmodel['.*mul.coregion.W'])
            print(optimizer.GPmodel['.*mul_1.coregion.W'])

            print(optimizer.GPmodel['.*mul.coregion.kappa'])
            print(optimizer.GPmodel['.*mul_1.coregion.kappa'])

            print(optimizer.GPmodel['.*sum.mul.rbf.variance'])
            print(optimizer.GPmodel['.*sum.mul_1.rbf.variance'])

            print(optimizer.GPmodel['.*sum.mul.rbf.lengthscale'])
            print(optimizer.GPmodel['.*sum.mul_1.rbf.lengthscale'])
        elif kernel_name=='linear+rbf':
            print(optimizer.GPmodel['.*mul.coregion.W'])
            print(optimizer.GPmodel['.*mul_1.coregion.W'])

            print(optimizer.GPmodel['.*mul.coregion.kappa'])
            print(optimizer.GPmodel['.*mul_1.coregion.kappa'])

            print(optimizer.GPmodel['.*sum.mul.sum.rbf.variance'])
            print(optimizer.GPmodel['.*sum.mul_1.sum.rbf.variance'])

            print(optimizer.GPmodel['.*sum.mul.sum.rbf.lengthscale'])
            print(optimizer.GPmodel['.*sum.mul_1.sum.rbf.lengthscale'])

            print(optimizer.GPmodel['.*sum.mul.sum.linear.variance'])
            print(optimizer.GPmodel['.*sum.mul_1.sum.linear.variance'])

    for i in range(ITR_MAX):

        print('Means and stds of GPmodel:', optimizer.GPmodel.means, optimizer.GPmodel.stds)
        print('Unique input size:', np.shape(np.unique(training_input[0], axis=0))[0])

        if (current_cost - previous_cost) >= save_cost:
            previous_cost = current_cost.copy()
            if X_all is not None:
                f_mean, _ = optimizer.GPmodel.predict(np.c_[X_all, np.c_[np.zeros(np.shape(X_all)[0])]])
                feasible_prob = np.ones(np.shape(X_all)[0])
                for c in range(C):
                    g_mean, g_var = optimizer.GPmodel.predict_noiseless(np.c_[X_all, (c+1)*np.c_[np.ones(np.shape(X_all)[0])]])
                    feasible_prob *= (1 - norm.cdf((thresholds[c]-g_mean) / np.sqrt(g_var))).ravel()
                if np.any(feasible_prob >= lower_p):
                    high_prob_index = feasible_prob >= lower_p
                    inference_point = X_all[high_prob_index][np.argmax(f_mean[high_prob_index].ravel())]
                else:
                    inference_point = None
            else:
                additional_point = None
                if (inference_point is not None) and (optimizer.y_max is not None):
                    additional_point = np.r_[np.c_[inference_point].T, optimizer.feasible_points]
                elif inference_point is not None:
                    additional_point = np.c_[inference_point].T
                elif optimizer.y_max is not None:
                    additional_point = optimizer.feasible_points

                start = time.time()
                inference_point = optimizer.posteriori_const_optimize(lower_p, additional_x=additional_point)
                print('InfReg time:', time.time() - start)


            cost_list.append(current_cost)
            if inference_point is not None:
                inference_outputs = np.vstack(test_func.mf_values([np.atleast_2d(inference_point) for c in range(C+1)]))
                print('InfReg const val', inference_outputs.ravel()[1:])

                print('prediction of inference point:')
                for c in range(C+1):
                    print('c =', c, optimizer.GPmodel.predict_noiseless(np.c_[np.atleast_2d(inference_point), [c]]))
                if np.all(inference_outputs.ravel()[1:] >= thresholds):# - 1e-3):
                    InfReg_list.append(inference_outputs[0])
                else:
                    InfReg_list.append([None])
            else:
                InfReg_list.append([None])

            if optimizer.y_max is None:
                SimReg_list.append([None])
            else:
                # This operation is easy to implement, but may be time-consuming for real-experiment.
                # SimReg_output = test_func.values(np.atleast_2d(optimizer.x_max), fidelity=0)

                SimReg_output = training_output[0][np.all(training_input[0]==np.atleast_2d(optimizer.x_max), axis=1)]
                SimReg_list.append([SimReg_output.ravel()[0]])


            with open(results_path + 'cost.pickle', 'wb') as f:
                pickle.dump(np.array(cost_list), f)

            with open(results_path + 'InfReg.pickle', 'wb') as f:
                pickle.dump(np.array(InfReg_list), f)

            with open(results_path + 'SimReg.pickle', 'wb') as f:
                pickle.dump(np.array(SimReg_list), f)

            with open(results_path + 'EvalNum.pickle', 'wb') as f:
                pickle.dump(np.array(eval_num), f)

            print('Cost, eval_num, InfMax, SimMax :', cost_list[-1], eval_num, InfReg_list[-1], SimReg_list[-1])
            if (i % 50) == 0:
                with open(results_path + 'optimizer_log/' + 'optimizer'+str(int(i))+'.pickle', 'wb') as f:
                    pickle.dump(optimizer, f)

            if (cost_list[-1] >= COST_MAX) or (i==ITR_MAX-1):
                with open(results_path + 'optimizer_log/' + 'optimizer'+str(int(i))+'.pickle', 'wb') as f:
                    pickle.dump(optimizer, f)
                break

        # add new input
        start = time.time()
        if X_all is None:
            new_selected_input_list = optimizer.next_input()
        else:
            new_selected_input_list, pool_X = optimizer.next_input_pool(pool_X)
            # pool_X = X_all.copy()
        print('new input select:', time.time() - start)
        # print(new_selected_input_list)

        remain_cost = [np.r_[remain_cost[c], cost[c] * np.ones(np.shape(new_selected_input_list[c])[0])] for c in range(C+1)]
        selected_inputs_list = [np.r_[selected_inputs_list[c], new_selected_input_list[c]] for c in range(C+1)]

        if 'dec' in BO_method:
            iter_cost = np.min(np.hstack(remain_cost))
            current_cost += iter_cost
        else:
            iter_cost = np.min(np.hstack(remain_cost))
            current_cost += coupled_cost

        remain_cost = [remain_cost[c] - iter_cost for c in range(C+1)]
        new_input_list = [selected_inputs_list[c][remain_cost[c]==0] for c in range(C+1)]
        selected_inputs_list = [selected_inputs_list[c][remain_cost[c]>0] for c in range(C+1)]
        remain_cost = [remain_cost[c][remain_cost[c]>0] for c in range(C+1)]

        num_worker = parallel_num - np.size(np.hstack(remain_cost))


        new_output_list = test_func.mf_values(new_input_list)
        eval_num += np.array([np.size(output) for output in new_output_list])
        print("new_input :\n", new_input_list)

        # new_input_list = [[] for c in range(C+1)]
        # new_output_list = [[] for c in range(C+1)]


        if ( (func_name in ['Gardner1', 'Gardner2', 'Gramacy', 'G24', 'Test_MESC']) or (('SynFun' in func_name) and (test_func.d==2) )) and (BO_method in ['ECI', 'MESC', 'MESC_LB', 'MESC_LB_corr']) and (i % 5 == 0):
            GRID_NUM = 101
            x1 = np.linspace(test_func.bounds[0,0], test_func.bounds[1,0], GRID_NUM,endpoint=True)
            x2 = np.linspace(test_func.bounds[0,1], test_func.bounds[1,1], GRID_NUM,endpoint=True)
            X1, X2 = np.meshgrid(x1,x2)
            X = np.c_[np.c_[X1.ravel()], np.c_[X2.ravel()]]
            fig = plt.figure(figsize=(4*(C+1) + 4, 9))
            for c in range(C+1):
                params = ''
                params = params+'c='+str(c)
                params = params+r',$\mu_{pri} =$'+f'{optimizer.GPmodel.means[c]:.03f}'
                params = params+r',$\sigma_{pri} =$'+f'{optimizer.GPmodel.stds[c]:.03f}'
                if model=='independent':
                    params = params+r',$\ell =[$'+f"{optimizer.GPmodel.model_list[c]['.*rbf.lengthscale'].values[0]:.03f}"
                    params = params+str(',')+f"{optimizer.GPmodel.model_list[c]['.*rbf.lengthscale'].values[1]:.03f}" + ']'
                mean, var = optimizer.GPmodel.predict_noiseless(np.c_[X, c*np.c_[np.ones(np.shape(X)[0])]])
                plt.subplot(3, C+1, c+1)
                plt.pcolor(X1, X2, mean.reshape(GRID_NUM,GRID_NUM))
                plt.colorbar()
                plt.scatter(training_input[c][:,0], training_input[c][:,1], c='gray', marker='x', s=5)
                if np.size(new_input_list[c]) != 0:
                    plt.scatter(new_input_list[c][:,0], new_input_list[c][:,1], c='red', marker='*', s=15)
                plt.xlim(test_func.bounds[0,0], test_func.bounds[1,0])
                plt.ylim(test_func.bounds[0,1], test_func.bounds[1,1])
                plt.title('mean:('+params+')', fontsize=10)

                plt.subplot(3, C+1, (C+1)+c+1)
                plt.pcolor(X1, X2, np.sqrt(var).reshape(GRID_NUM,GRID_NUM))
                plt.colorbar()
                plt.scatter(training_input[c][:,0], training_input[c][:,1], c='black', marker='x', s=10)
                if np.size(new_input_list[c]) != 0:
                    plt.scatter(new_input_list[c][:,0], new_input_list[c][:,1], c='red', marker='*', s=15)
                plt.xlim(test_func.bounds[0,0], test_func.bounds[1,0])
                plt.ylim(test_func.bounds[0,1], test_func.bounds[1,1])
                plt.title('std', fontsize=10)

            if not('dec' in BO_method):
                if model=='independent':
                    acq = optimizer.acq(X)
                elif model=='correlated':
                    acq = list()
                    for j in range(np.shape(X)[0]):
                        acq.append(optimizer.acq_correlated(X[j]))
                    acq = np.array(acq)

                plt.subplot(3, C+1, 2*(C+1)+1)
                plt.pcolor(X1, X2, -acq.reshape(GRID_NUM,GRID_NUM))
                plt.colorbar()
                plt.scatter(training_input[0][:,0], training_input[0][:,1], c='gray', marker='x', s=5)
                if np.size(new_input_list[c]) != 0:
                    plt.scatter(new_input_list[0][:,0], new_input_list[0][:,1], c='red', marker='*', s=15)
                plt.scatter(X1.ravel()[np.argmin(acq)], X2.ravel()[np.argmin(acq)], c='green', marker='*', s=15)
                plt.xlim(test_func.bounds[0,0], test_func.bounds[1,0])
                plt.ylim(test_func.bounds[0,1], test_func.bounds[1,1])
                plt.title('acq', fontsize=10)
            else:
                for c in range(C+1):
                    optimizer.eval_c = c
                    acq = optimizer.acq(X)
                    plt.subplot(3, C+1, 2*(C+1)+c+1)
                    plt.pcolor(X1, X2, -acq.reshape(GRID_NUM,GRID_NUM))
                    plt.colorbar()
                    plt.scatter(training_input[0][:,0], training_input[0][:,1], c='gray', marker='x', s=5)
                    if np.size(new_input_list[c]) != 0:
                        plt.scatter(new_input_list[c][:,0], new_input_list[c][:,1], c='red', marker='*', s=15)
                    if np.max(acq) > 0:
                        cont=plt.contour(X1, X2, -acq.reshape(GRID_NUM,GRID_NUM), colors=['black'], levels=[0, 0.1, 0.2])
                        cont.clabel(fmt='%1.1f', fontsize=10)
                    plt.scatter(X1.ravel()[np.argmin(acq)], X2.ravel()[np.argmin(acq)], c='green', marker='*', s=15)
                    plt.xlim(test_func.bounds[0,0], test_func.bounds[1,0])
                    plt.ylim(test_func.bounds[0,1], test_func.bounds[1,1])
                    plt.title('acq', fontsize=10)


            plt.tight_layout()
            plt.savefig(results_path + func_name+'_'+BO_method+'_'+str(initial_seed) +'_ITR='+str(i)+'.png')
            plt.close()

        training_input = [np.r_[training_input[c], new_input_list[c]] if np.size(new_input_list[c]) > 0 else training_input[c] for c in range(C+1)]
        training_output = [np.r_[training_output[c], new_output_list[c]] if np.size(new_output_list[c]) > 0 else training_output[c] for c in range(C+1)]

        # print("new_input :", new_input_list)
        print("new_output :\n", new_output_list)
        print("the predictions (mean, var):")
        for c in range(C+1):
            print('c =', c, optimizer.GPmodel.predict_noiseless(np.c_[new_input_list[c], c*np.c_[np.ones(np.shape(new_input_list[c])[0])]]))

        if i == 0:
            temp_quotient = 0
        if optimize and (i+1)*parallel_num // 5 > temp_quotient:
            optimizer.update(new_input_list, new_output_list, optimize=True)
            # print the parameters of GPs
            print(optimizer.GPmodel.means, optimizer.GPmodel.stds)
            if model=='independent':
                print(optimizer.GPmodel.model_list[0]['.*Gaussian_noise.variance'])
                if kernel_name=='rbf':
                    for c in range(C+1):
                        print(optimizer.GPmodel.model_list[c]['.*rbf.variance'])
                        print(optimizer.GPmodel.model_list[c]['.*rbf.lengthscale'])
                elif kernel_name=='linear+rbf':
                    for c in range(C+1):
                        print(optimizer.GPmodel.model_list[c]['.*sum.linear.variances'])
                        print(optimizer.GPmodel.model_list[c]['.*sum.rbf.variance'])
                        print(optimizer.GPmodel.model_list[c]['.*sum.rbf.lengthscale'])
            elif model=='correlated':
                # print(optimizer.GPmodel)
                if kernel_name=='rbf':
                    print(optimizer.GPmodel['.*mul.coregion.W'])
                    print(optimizer.GPmodel['.*mul_1.coregion.W'])

                    print(optimizer.GPmodel['.*mul.coregion.kappa'])
                    print(optimizer.GPmodel['.*mul_1.coregion.kappa'])

                    print(optimizer.GPmodel['.*sum.mul.rbf.variance'])
                    print(optimizer.GPmodel['.*sum.mul_1.rbf.variance'])

                    print(optimizer.GPmodel['.*sum.mul.rbf.lengthscale'])
                    print(optimizer.GPmodel['.*sum.mul_1.rbf.lengthscale'])
                elif kernel_name=='linear+rbf':
                    print(optimizer.GPmodel['.*mul.coregion.W'])
                    print(optimizer.GPmodel['.*mul_1.coregion.W'])

                    print(optimizer.GPmodel['.*mul.coregion.kappa'])
                    print(optimizer.GPmodel['.*mul_1.coregion.kappa'])

                    print(optimizer.GPmodel['.*sum.mul.sum.rbf.variance'])
                    print(optimizer.GPmodel['.*sum.mul_1.sum.rbf.variance'])

                    print(optimizer.GPmodel['.*sum.mul.sum.rbf.lengthscale'])
                    print(optimizer.GPmodel['.*sum.mul_1.sum.rbf.lengthscale'])

                    print(optimizer.GPmodel['.*sum.mul.sum.linear.variance'])
                    print(optimizer.GPmodel['.*sum.mul_1.sum.linear.variance'])
        else:
            optimizer.update(new_input_list, new_output_list, optimize=False)
        temp_quotient = (i+1)*parallel_num // 5
        # print('quotient', temp_quotient)


if __name__ == '__main__':
    args = sys.argv
    BO_method = args[1]
    test_func = args[2]
    function_seed = np.int(args[3])
    initial_seed = np.int(args[4])
    parallel_num = np.int(args[5])
    options = [option for option in args if option.startswith('-')]
    if '-l' in options or '--log' in options:
        log_flag = True
    else:
        log_flag = False

    test_funcs = ['const_SynFun', 'const_SynFun_compress', 'const_SynFun_pool', 'const_SynFun_test', 'const_SynFun_plus_corr', 'const_SynFun_plus_corr_pool', 'const_SynFun_minus_corr', 'const_HartMann6', 'Gardner1', 'Gardner2', 'Gramacy', 'WeldedBeam', 'PressureVessel', 'TensionCompressionString', 'Test_MESC']
    test_funcs.extend(['Bi2O3', 'LLTO', 'const_cnn_mnist', 'const_cnn_cifar10'])
    test_funcs.extend(['G1', 'G4' , 'G6', 'G7', 'G8', 'G9', 'G10', 'G18', 'G24'])
    test_funcs.extend(['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6'])
    test_funcs.extend(["HeatExchangerNetworkDesign1", "HeatExchangerNetworkDesign2", "OptimalOperationOfAlkylationUnit", "ReactorNetworkDesign", "WeightedMinimizationOfSpeedReducer", "IndustrialRefrigerationSystem", "Three_barTrussDesign", "MultipleDiskClutchBrakeDesign", "Himmelblau_Function"])

    BO_methods = ['US', 'MESC', 'MESC_MinVio', 'MESC_LB', 'MESC_LB_MinVio', 'MESC_dec', 'MESC_dec_KL', 'MESC_seqdec_KL', 'ECI', 'TSC']
    BO_methods.extend(['MESC1', 'MESC50', 'MESC_LB1', 'MESC_LB50'])
    BO_methods.extend(['MESC_corr', 'MESC_LB_corr', 'TSC_corr'])
    parallel_methods = ['PMESC', 'PMESC_LB', 'PMESC_MinVio', 'PMESC_dec', 'PMESC_dec_KL', 'PECI', 'PTSC']
    BO_methods.extend(parallel_methods)

    if BO_method in parallel_methods:
        if parallel_num < 2:
            print('method is parallellized, but # parallel is lower than 2.')
            exit()
    else:
        if parallel_num >= 2:
            print('method is not parallellized, but # parallel is larger than 2.')
            exit()

    if not(test_func in test_funcs):
        print(test_func + ' is not implemented!')
        exit(1)
    if not(BO_method in BO_methods):
        print(BO_method + ' is not implemented!')
        exit(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.system('echo $OMP_NUM_THREADS')
    NUM_WORKER = 10

    # When seed = -1, experiments of seed of 0-9 is done for parallel
    # When other seed is set, experiments of the seed is done
    if function_seed >= 0:
        if initial_seed >= 0:
            main((test_func, BO_method, function_seed, initial_seed, parallel_num))
            exit()

        function_seeds = [function_seed]
        initial_seeds = np.arange(NUM_WORKER).tolist()
    else:
        function_seeds = np.arange(10).tolist()
        if initial_seed < 0:
            initial_seeds = np.arange(NUM_WORKER).tolist()
        else:
            initial_seeds = [initial_seed]

    params = list()
    for f_seed in function_seeds:
        for i_seed in initial_seeds:
            params.append((test_func, BO_method, f_seed, i_seed, parallel_num))
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
        results = executor.map(main, params)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKER) as executor:
    #     results = executor.map(test, params)
