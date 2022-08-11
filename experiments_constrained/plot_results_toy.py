# -*- coding: utf-8 -*-
import os
import sys
import pickle
import re

import numpy as np
import numpy.matlib
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_functions"))
import test_functions

# plt.rcParams['pdf.fonttype'] = 42 # Type3font回避
# plt.rcParams['ps.fonttype'] = 42 # Type3font回避
matplotlib.rcParams['text.usetex'] = True # Type1 font
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # Type1 font

# plt.rcParams['font.family'] = 'Helvetica' # font familyの設定
# plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 25 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 22 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 22 # 軸だけ変更されます
plt.rcParams['legend.fontsize'] = 20 # 軸だけ変更されます
plt.rcParams['figure.figsize'] = (7,5.5)

plt.rcParams['errorbar.capsize'] = 4.0

plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams['lines.markeredgewidth'] = 2.5

plt.rcParams['legend.borderaxespad'] = 0.15
plt.rcParams['legend.borderpad'] = 0.2
plt.rcParams['legend.columnspacing'] = 0.5
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handleheight'] = 0.5

def plot_toy(q=4, Parallel=False, average=False):

    flag_log = True
    if flag_log:
        str_log = '_log'
    else:
        str_log = ''

    if average:
        str_ave = '_average'
    else:
        str_ave = ''
    STR_NUM_WORKER = 'Q='+str(q)

    if Parallel:
        NCOL=2
    else:
        NCOL=2

    plot_max = 10
    plot_min = 1e-6

    seeds_num = 10
    seeds = np.arange(seeds_num)
    func_names = ['Gardner1', 'Gardner2', 'Gramacy', 'const_HartMann6', 'G1', 'G4', 'G7', 'G9', 'G10', 'G18', "ReactorNetworkDesign"] #, "WeightedMinimizationOfSpeedReducer" , "MultipleDiskClutchBrakeDesign", "Himmelblau_Function"]
    func_names = ['Gardner1', 'Gardner2', 'Gramacy', 'const_HartMann6', 'G1', 'G7', 'G9', 'G10', "ReactorNetworkDesign", "const_cnn_cifar10"]

    # func_names = ['Gardner1', 'Gardner2', 'Gramacy', 'const_HartMann6']
    if Parallel:
        BO_methods = ['PECI_'+STR_NUM_WORKER, 'PTSC_'+STR_NUM_WORKER, 'PMESC_'+STR_NUM_WORKER, 'PMESC_LB_'+STR_NUM_WORKER]
    else:
        BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB', 'PESC']
        # BO_methods = ['MESC1', 'MESC', 'MESC50', 'MESC_LB1', 'MESC_LB', 'MESC_LB50', 'ECI', 'TSC', 'PESC']

    for i, func_name in enumerate(func_names):
        # fig = plt.figure(figsize=(7, 5))
        for method in BO_methods:
            GLOBAL_MAX = None
            GLOBAL_MIN = None
            test_func = eval('test_functions.'+func_name)()

            if func_name=='const_HartMann6':
                COST_INI = 5 * test_func.d
                COST_MAX = COST_INI + 200
            elif 'cnn' in func_name:
                COST_INI = 5
                COST_MAX = 100
            elif re.match('G\d+', func_name):
                COST_INI = 10
                COST_MAX = COST_INI + 50
            elif test_func.d == 2:
                COST_INI = 5
                COST_MAX = COST_INI + 30
            else:
                COST_INI = np.min([25, 5 * test_func.d])
                COST_MAX = COST_INI + 50

            COST_MAX += 10
            plot_cost = np.arange(COST_MAX)
            SimReg_all = np.ones((seeds_num, COST_MAX)) * np.inf
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if func_name=='Gardner1':
                GLOBAL_MAX = 1.8887513614906686
                GLOBAL_MIN = -2.0
            if func_name=='Gardner2':
                GLOBAL_MAX = - 0.2532358975009539
                GLOBAL_MIN = - 7.0
            if func_name=='Gramacy':
                GLOBAL_MAX = - 0.5997880520093705
                GLOBAL_MIN = - 2.0
            if func_name=='const_HartMann6':
                GLOBAL_MAX = 3.322368011415267
                GLOBAL_MIN = 2.8124505439686544e-08

            if re.match('G\d+', func_name):
                GLOBAL_MAX = test_func.f_star
                GLOBAL_MIN = test_func.f_min

            if func_name=='MultipleDiskClutchBrakeDesign':
                GLOBAL_MAX = -0.2352424579008037
                GLOBAL_MIN = -6.2486277879901
            if func_name=='ReactorNetworkDesign':
                GLOBAL_MAX = 0.3898703631283752
                GLOBAL_MIN = 0.0
            if func_name=='WeightedMinimizationOfSpeedReducer':
                GLOBAL_MAX = -2994.4244746545537
                GLOBAL_MIN = -7144.667944998401
            if func_name=='Himmelblau_Function':
                GLOBAL_MAX = 30665.103087618983
                GLOBAL_MIN = 22302.761885500004
            if 'cnn' in func_name:
                GLOBAL_MAX = test_func.CONST_MAX
                GLOBAL_MIN = test_func.GLOBAL_MIN
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'SimReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'SimReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)
                            '''
                            横軸をiterationに変更
                            '''
                            cost = cost / (test_func.C + 1)
                            # print(func_name, method, np.size(cost))

                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg==None] = GLOBAL_MIN
                            # if func_name=='LLTO' and 'MESC_LB' in method:
                            #     print(np.shape(SimReg))

                        if np.any(SimReg[:-1] - SimReg[1:] > 0):
                            print(func_name, method, seed)
                            print(SimReg[np.where(SimReg[:-1] - SimReg[1:] > 0)[0]])
                            print(SimReg[np.where(SimReg[:-1] - SimReg[1:] > 0)[0] + 1])
                            print(np.where(SimReg[:-1] - SimReg[1:] > 0)[0])
                            print(cost[np.where(SimReg[:-1] - SimReg[1:] > 0)[0] + 1])

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(SimReg):
                                if j+1 == np.size(cost):
                                    break
                                else:
                                    SimReg_all[seed, int(cost[j]) : int(cost[j+1])] = SimReg[j]
                else:
                    plot=False
            # 各関数でRegretに変換
            # if GLOBAL_MAX < np.max(SimReg_all[np.logical_not(np.isinf(SimReg_all))]):
            #     print(func_name, np.max(SimReg_all[np.logical_not(np.isinf(SimReg_all))]))

            if 'cnn' in func_name:
                SimReg_all = 1. / (1 + np.exp(- SimReg_all))
                SimReg_all[SimReg_all == 1] = np.inf
                SimReg_all = 1. / (1 + np.exp(- GLOBAL_MAX)) - SimReg_all
            else:
                SimReg_all = GLOBAL_MAX - SimReg_all
            # SimReg_all = GLOBAL_MAX - SimReg_all
            # print(SimReg_all[np.logical_not(np.isinf(SimReg_all))][SimReg_all[np.logical_not(np.isinf(SimReg_all))] < 0])
            # SimReg_all[SimReg_all < 0] = 0

            if plot:
                temp_index = BO_methods.index(method)
                index = np.logical_not(np.any(np.isinf(SimReg_all), axis=0))
                method = method.replace('ECI', 'EIC')
                method = method.replace('_Q='+str(q), '')
                method = method.replace('_', '-')

                method = method.replace('MESC', 'CMES')
                method = method.replace('-LB', '-IBO')
                plot_cost -= COST_INI


                if average:
                    SimReg_ave = np.mean(SimReg_all, axis=0)
                    SimReg_se = np.sqrt(np.sum((SimReg_all - SimReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                    # plt.plot(plot_cost[index], SimReg_ave[index], label=method)
                    # plt.fill_between(plot_cost[index], SimReg_ave[index] - SimReg_se[index], SimReg_ave[index] + SimReg_se[index], alpha=0.3)

                    if 'HartMann6' in func_name:
                        plt.errorbar(plot_cost[index], SimReg_ave[index], yerr=SimReg_se[index], errorevery=(temp_index*5, 25), elinewidth=3, label=method)
                    else:
                        plt.errorbar(plot_cost[index], SimReg_ave[index], yerr=SimReg_se[index], errorevery=(temp_index, 5), elinewidth=3, label=method)
                else:
                    SimReg_median = np.median(SimReg_all, axis=0)
                    SimReg_1_4 = np.quantile(SimReg_all, 1/4., axis=0)
                    SimReg_3_4 = np.quantile(SimReg_all, 3/4., axis=0)
                    plt.plot(plot_cost[index], SimReg_median[index], label=method)
                    plt.fill_between(plot_cost[index], SimReg_1_4[index], SimReg_3_4[index], alpha=0.3)
        func_name = func_name.replace('const_', '')
        func_name = func_name.replace('_', ' ')
        plt.title(func_name + ' (d='+str(test_func.d)+')'  + ' (C='+str(test_func.C)+')')
        plt.xlim(COST_INI - COST_INI, COST_MAX - COST_INI - 10)
        # plt.ylim(plot_min, plot_max)
        if func_name == 'Bi2O3':
            plt.ylim(1e-12, 1e-9)
        elif func_name == 'LLTO':
            plt.ylim(1e-8, 1e-5)

        if flag_log:
            plt.yscale('log')
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xlabel('Iteration')
        plt.ylabel('Utility Gap')
        plt.legend(loc='best')
        plt.tight_layout()


        if Parallel:
            plt.savefig('Results_Sim_'+func_name+str_log+str_ave+'_'+STR_NUM_WORKER+'.pdf')
        else:
            plt.savefig('Results_Sim_'+func_name+str_log+str_ave+'.pdf')
        plt.close()


    # Inference plot
    for i, func_name in enumerate(func_names):

        if not('1' in BO_methods[0]):
            fig = plt.figure(figsize=(5, 7))

        for method in BO_methods:
            GLOBAL_MAX = None
            GLOBAL_MIN = None
            test_func = eval('test_functions.'+func_name)()

            if func_name=='const_HartMann6':
                COST_INI = 5 * test_func.d
                COST_MAX = COST_INI + 200
            elif 'cnn' in func_name:
                COST_INI = 5
                COST_MAX = 100
            elif re.match('G\d+', func_name):
                COST_INI = 10
                COST_MAX = COST_INI + 50
            elif test_func.d == 2:
                COST_INI = 5
                COST_MAX = COST_INI + 30
            else:
                COST_INI = np.min([25, 5 * test_func.d])
                COST_MAX = COST_INI + 50

            COST_MAX += 10
            plot_cost = np.arange(COST_MAX)
            InfReg_all = np.ones((seeds_num, COST_MAX)) * np.inf
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            standardaized_mean = 0
            standardaized_std = 1
            if func_name=='Gardner1':
                GLOBAL_MAX = 1.8887513614906686
                GLOBAL_MIN = -2.0
            if func_name=='Gardner2':
                GLOBAL_MAX = - 0.2532358975009539
                GLOBAL_MIN = - 7.0
            if func_name=='Gramacy':
                GLOBAL_MAX = - 0.5997880520093705
                GLOBAL_MIN = - 2.0
            if func_name=='const_HartMann6':
                GLOBAL_MAX = 3.322368011415267
                GLOBAL_MIN = 2.8124505439686544e-08

            if re.match('G\d+', func_name):
                GLOBAL_MAX = test_func.f_star
                GLOBAL_MIN = test_func.f_min

            if func_name=='MultipleDiskClutchBrakeDesign':
                GLOBAL_MAX = -0.2352424579008037
                GLOBAL_MIN = -6.2486277879901
            if func_name=='ReactorNetworkDesign':
                GLOBAL_MAX = 0.3898703631283752
                GLOBAL_MIN = 0.0
            if func_name=='WeightedMinimizationOfSpeedReducer':
                GLOBAL_MAX = -2994.4244746545537
                GLOBAL_MIN = -7144.667944998401
            if func_name=='Himmelblau_Function':
                GLOBAL_MAX = 30665.103087618983
                GLOBAL_MIN = 22302.761885500004
            if 'cnn' in func_name:
                GLOBAL_MAX = test_func.CONST_MAX
                GLOBAL_MIN = test_func.GLOBAL_MIN
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)

                            '''
                            横軸をiterationに変更
                            '''
                            cost = cost / (test_func.C + 1)

                            # if func_name=='ReactorNetworkDesign' and method=='MESC':
                            #     print(cost)

                            if method=='MESC_LB_corr':
                                print(func_name,seed,cost[-1])
                            # if func_name=='Gardner1':
                            #     print(seed, cost[-1])
                        with open(temp_path + 'InfReg.pickle', 'rb') as f:
                            InfReg = pickle.load(f)
                            InfReg[InfReg==None] = GLOBAL_MIN

                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg==None] = GLOBAL_MIN
                        InfReg[InfReg < SimReg] = SimReg[InfReg < SimReg]

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(InfReg):
                                if j+1 == np.size(cost):
                                    break
                                else:
                                    InfReg_all[seed, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                else:
                    plot=False


            if 'cnn' in func_name:
                InfReg_all = 1. / (1 + np.exp(- InfReg_all))
                InfReg_all[InfReg_all == 1] = np.inf
                InfReg_all = 1. / (1 + np.exp(- GLOBAL_MAX)) - InfReg_all
            else:
                InfReg_all = GLOBAL_MAX - InfReg_all
            # InfReg_all = GLOBAL_MAX - InfReg_all

            if method == 'PESC' and not('Reactor' in func_name):
                if os.path.exists('PESC_results/'+func_name+'.csv'):
                    plot=True
                    InfReg_all = np.loadtxt('PESC_results/'+func_name+'.csv', delimiter=',')
                    InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
                    if np.shape(InfReg_all)[1] < COST_MAX:
                        InfReg_all = np.c_[InfReg_all, np.inf*np.ones( (np.shape(InfReg_all)[0], COST_MAX - np.shape(InfReg_all)[1]) )]
                    else:
                        InfReg_all = InfReg_all[:,:COST_MAX]


            if plot:
                index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))
                plot_cost -= COST_INI

                temp_index = BO_methods.index(method)

                method = method.replace('ECI', 'EIC')
                method = method.replace('_Q='+str(q), '')
                method = method.replace('_', '-')

                method = method.replace('MESC', 'CMES')
                method = method.replace('-LB', '-IBO')
                if Parallel:
                    method = method.replace('P', 'P-')

                if average:
                    InfReg_ave = np.mean(InfReg_all, axis=0)
                    InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                    if np.all(InfReg_all < 0):
                        print(method, func_name)
                        print(np.where(InfReg_all > 0))
                    # plt.plot(plot_cost[index], InfReg_ave[index], label=method)
                    # plt.fill_between(plot_cost[index], InfReg_ave[index] - InfReg_se[index], InfReg_ave[index] + InfReg_se[index], alpha=0.3)
                    if '1' in method or '50' in method:
                        line = '-'
                        if 'IBO' in method:
                            if '1' in method:
                                line = '--'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
                                method='IBO1'
                            else:
                                line = ':'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
                                method='IBO50'
                        else:
                            if '1' in method:
                                line = '--'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
                            else:
                                line = ':'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
                        if func_name=='const_HartMann6':
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(temp_index*5, 25), elinewidth=3., linestyle=line, color=c, label=method)
                        else:
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(temp_index, 5), elinewidth=3., linestyle=line, color=c, label=method)
                    elif 'CMES' in method:
                        if 'IBO' in method:
                            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
                        else:
                            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
                        if func_name=='const_HartMann6':
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(temp_index*5, 25), elinewidth=3., color=c, label=method)
                        else:
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(temp_index, 5), elinewidth=3., color=c, label=method)
                    else:
                        if 'PESC' in method:
                            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][4]
                        else:
                            c = None
                        if func_name=='const_HartMann6':
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(temp_index*5, 25), elinewidth=3., color=c, label=method)
                        else:
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(temp_index, 5), elinewidth=3., color=c, label=method)
                else:
                    InfReg_median = np.median(InfReg_all, axis=0)
                    InfReg_1_4 = np.quantile(InfReg_all, 1/4., axis=0)
                    InfReg_3_4 = np.quantile(InfReg_all, 3/4., axis=0)
                    # plt.plot(plot_cost[index], InfReg_median[index], label=method)
                    # plt.fill_between(plot_cost[index], InfReg_1_4[index], InfReg_3_4[index], alpha=0.3)

                    error_bar_plot = np.r_[np.atleast_2d((InfReg_median - InfReg_1_4)[index]), np.atleast_2d((InfReg_3_4 - InfReg_median)[index])]
                    plt.errorbar(plot_cost[index], InfReg_median[index], yerr=error_bar_plot, errorevery=5, capsize=3, elinewidth=1, label=method)

        # if not Parallel and func_name in ['const_HartMann6', 'Gardner1', 'Gardner2', 'Gramacy']:
        #     plt.vlines(-10, 5, 10, color='grey', linestyle='dotted', label='50')
        #     plt.vlines(-10, 5, 10, color='grey', linestyle='dashed', label='1')

        func_name = func_name.replace('const_', '')
        func_name = func_name.replace('_', ' ')
        if func_name == ('HartMann6'):
            func_name = 'Hartmann6'
        if func_name == 'cnn cifar10':
            func_name = 'CNN'
        if func_name == 'ReactorNetworkDesign':
            func_name = 'ReactorNetwork'

        if not Parallel:
            if func_name=='Gardner1' and not('1' in BO_methods[0]):
                # plt.title('(b) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('(b) '+func_name + ' (C='+str(test_func.C)+')')
            # elif func_name=='Hartmann6':
            #     plt.title('(c) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
            elif func_name=='G1':
                # plt.title('(c) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('(c) '+func_name + ' (C='+str(test_func.C)+')')
            elif func_name=='G7':
                # plt.title('(c) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('(d) '+func_name + ' (C='+str(test_func.C)+')')
            elif func_name=='G10':
                # plt.title('(d) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('(e) '+func_name + ' (C='+str(test_func.C)+')')
            elif 'Reactor' in func_name:
                # plt.title('(e) Reactor network' + ' (C='+str(test_func.C)+')')
                plt.title('(f) Reactor netw.' + ' (C='+str(test_func.C)+')', x = 0.425, y = 1.0)
            elif 'CNN' in func_name:
                # plt.title('(f) CNN' + ' (C='+str(test_func.C)+')')
                plt.title('(g) CNN' + ' (C='+str(test_func.C)+')')
            else:
                # plt.title(func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title(func_name)
        else:
            if func_name=='G1':
                # plt.title('(a) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('(a) '+func_name + ' (C='+str(test_func.C)+')')
            elif func_name=='G7':
                # plt.title('(a) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('(b) '+func_name + ' (C='+str(test_func.C)+')')
            elif func_name=='G10':
                # plt.title('(b) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('(c) '+func_name + ' (C='+str(test_func.C)+')')
            elif 'Reactor' in func_name:
                # plt.title('Reactor network' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('Reactor netw.' + ' (C='+str(test_func.C)+')')
            elif 'CNN' in func_name:
                # plt.title('CNN' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title('CNN')
            else:
                # plt.title(func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
                plt.title(func_name)


        # plt.title(func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        # plt.title(func_name + ' (C='+str(test_func.C)+')')




        # if 'Weighted' in func_name:
        #     plt.title('Speed reducer' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        # if 'Multiple' in func_name:
        #     plt.title('Disk clutch brake' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        # plt.xlim(COST_INI, COST_MAX)
        plt.xlim(COST_INI - COST_INI, COST_MAX - COST_INI - 10)

        if flag_log:
            if re.match('G\d+', func_name):
                if func_name=='G1':
                    if Parallel:
                        plt.ylim(10**-3, 25)
                    else:
                        plt.ylim(7*10**-3, 25)
                elif func_name=='G4':
                    plt.ylim(10**-2, 10**4)
                elif func_name=='G7':
                    if Parallel:
                        plt.ylim(10**-1, 10**4)
                    else:
                        plt.ylim(8*10**-1, 10**4)
                elif func_name=='G9':
                    plt.ylim(10, 2*10**7)
                elif func_name=='G10':
                    plt.ylim(20, 3*10**4)
                elif func_name=='G18':
                    plt.ylim(10**-2, 5*10**2)
            elif 'Reactor' in func_name:
                plt.ylim(2*1e-3, 0.5)
            elif 'Gardner1' in func_name:
                plt.ylim(1e-4, 5.)
            elif 'Gardner2' in func_name:
                plt.ylim(1e-5, 10.)
            elif 'Gramacy' in func_name:
                plt.ylim(1e-5, 5.)
            elif 'Hartmann6' in func_name:
                plt.ylim(1e-5, 5.)
            elif 'Himmelblau' in func_name:
                plt.ylim(1e-2, 2*1e3)
            elif 'Multiple' in func_name:
                plt.ylim(1e-8, 10)
            elif 'Reducer' in func_name:
                plt.ylim(8*1e-3, 3*1e1)
            elif 'CNN' in func_name:
                plt.ylim(5*1e-5, 0.7)
            else:
                plt.ylim(plot_min, plot_max)
        else:
            if re.match('G\d+', func_name):
                if func_name=='G1':
                    plt.ylim(0, 25)
                elif func_name=='G4':
                    plt.ylim(0, 10**4)
                elif func_name=='G7':
                    plt.ylim(0, 30)
                elif func_name=='G9':
                    plt.ylim(0, 2*10**7)
                elif func_name=='G10':
                    plt.ylim(0, 500)
                elif func_name=='G18':
                    plt.ylim(0, 5*10**2)
            elif func_name=='ReactorNetworkDesign':
                plt.ylim(0, 0.4)
            elif func_name=='Hartmann6':
                plt.ylim(0, 0.5)
            elif 'Himmelblau' in func_name:
                plt.ylim(0, 2*1e3)
            elif 'Multiple' in func_name:
                plt.ylim(0, 10)
            elif 'Reducer' in func_name:
                plt.ylim(0, 3*1e1)
            else:
                plt.ylim(plot_min, plot_max)


        if standardaized_mean == 0:
            if func_name == 'Bi2O3':
                plt.ylim(1e-12, 1e-9)
            elif func_name == 'LLTO':
                plt.ylim(1e-8, 1e-5)
        else:
            if func_name == 'Bi2O3':
                plt.ylim(1e-2, 4)
            elif func_name == 'LLTO':
                plt.ylim(1e-2, 4)

        if flag_log:
            plt.yscale('log')
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xlabel('Iteration')

        # if not Parallel and func_name=='G7':
        #     plt.yticks([10000, 1000, 100, 10, 1], [r'$10^4 \quad$', r'$10^3$\ ', r'$10^2$\ ', r'$10^1$\ ', r'$10^0$\ '], )

        if Parallel:
            plt.ylabel('Utility Gap')
        elif not(func_name=='Gardner1') and not(func_name=='G1') and not(func_name=='G7') and not(func_name=='G10') and not('Reactor' in func_name) and not(func_name=='CNN'):
            plt.ylabel('Utility Gap')


        if not Parallel and func_name=='Gardner1':
            plt.legend(loc='best', ncol=1)
        elif Parallel and func_name=='G1':
            plt.legend(loc='best', ncol=1)

        if '1' in BO_methods[0]:
            plt.ylabel('Utility Gap')
            if func_name=='Gardner1':
                plt.legend(loc='upper right', ncol=3)
        # if func_name in ['Gardner1', 'Gramacy']:
        #     plt.legend(loc='best', ncol=NCOL)
        # if func_name == 'G7' and Parallel:
        #     plt.legend(loc='best', ncol=NCOL)
        # plt.legend(loc='best', ncol=NCOL)
        # if func_name == 'G10' and not Parallel:
        #     plt.legend(loc='lower left', ncol=1)
        # if func_name == 'Gardner1':
        #     plt.legend(loc='lower left', ncol=NCOL)
        # elif re.match('G\d+', func_name):
        #     plt.legend(loc='best')
        # plt.legend(loc='best')
        plt.tight_layout()

        if Parallel:
            if standardaized_mean == 0:
                plt.savefig('Results_Inf_'+func_name+str_log+str_ave+'_'+STR_NUM_WORKER+'.pdf')
            else:
                plt.savefig('Standardized_Results_Inf_'+func_name+str_log+str_ave+'_'+STR_NUM_WORKER+'.pdf')
        else:
            if standardaized_mean == 0:
                if '1' in BO_methods[0]:
                    plt.savefig('Results_Inf_'+func_name+str_log+str_ave+'_diffK.pdf')
                else:
                    plt.savefig('Results_Inf_'+func_name+str_log+str_ave+'.pdf')
            else:
                plt.savefig('Standardized_Results_Inf_'+func_name+str_log+str_ave+'.pdf')
        plt.close()



def plot_feasible_rate(q=4, Parallel=False, average=False):
    if average:
        str_ave = '_average'
    else:
        str_ave = ''
    STR_NUM_WORKER = 'Q='+str(q)

    if Parallel:
        NCOL=2
    else:
        NCOL=3


    seeds_num = 10
    seeds = np.arange(seeds_num)
    func_names = ['Gardner1', 'Gardner2', 'Gramacy', 'const_HartMann6', 'G1', 'G4', 'G7', 'G9', 'G10', 'G18', "ReactorNetworkDesign"] #, "WeightedMinimizationOfSpeedReducer" , "MultipleDiskClutchBrakeDesign", "Himmelblau_Function"]
    func_names = ['Gardner1', 'Gardner2', 'Gramacy', 'const_HartMann6', 'G1', 'G7', 'G9', 'G10', "ReactorNetworkDesign"]

    # func_names = ['Gardner1', 'Gardner2', 'Gramacy', 'const_HartMann6']
    if Parallel:
        BO_methods = ['PECI_'+STR_NUM_WORKER, 'PTSC_'+STR_NUM_WORKER, 'PMESC_'+STR_NUM_WORKER, 'PMESC_LB_'+STR_NUM_WORKER]
    else:
        BO_methods = ['MESC1', 'MESC_LB1', 'MESC', 'MESC_LB', 'MESC50', 'MESC_LB50']
        BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB', 'PESC']

    # Inference plot
    for i, func_name in enumerate(func_names):
        # fig = plt.figure(figsize=(7, 5))
        for method in BO_methods:
            GLOBAL_MAX = None
            GLOBAL_MIN = None
            test_func = eval('test_functions.'+func_name)()

            if func_name=='const_HartMann6':
                COST_INI = 5 * test_func.d
                COST_MAX = COST_INI + 200
            elif re.match('G\d+', func_name):
                COST_INI = 10
                COST_MAX = COST_INI + 50
            elif test_func.d == 2:
                COST_INI = 5
                COST_MAX = COST_INI + 30
            else:
                COST_INI = np.min([25, 5 * test_func.d])
                COST_MAX = COST_INI + 50

            COST_MAX += 10
            plot_cost = np.arange(COST_MAX)
            InfReg_all = np.ones((seeds_num, COST_MAX)) * np.inf
            result_path = func_name + '_results/'

            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)

                            '''
                            横軸をiterationに変更
                            '''
                            cost = cost / (test_func.C + 1)


                            if method=='MESC_LB_corr':
                                print(func_name,seed,cost[-1])
                            # if func_name=='Gardner1':
                            #     print(seed, cost[-1])
                        with open(temp_path + 'InfReg.pickle', 'rb') as f:
                            InfReg = pickle.load(f)
                            InfReg[InfReg!=None] = 1
                            InfReg[InfReg==None] = 0

                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg!=None] = 1
                            SimReg[SimReg==None] = 0

                        InfReg[InfReg < SimReg] = SimReg[InfReg < SimReg]

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(InfReg):
                                if j+1 == np.size(cost):
                                    break
                                else:
                                    InfReg_all[seed, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                else:
                    plot=False

            # if method == 'PESC':
            #     if os.path.exists('PESC_results/'+func_name+'.csv'):
            #         plot=True
            #         InfReg_all = np.loadtxt('PESC_results/'+func_name+'.csv', delimiter=',')
            #         InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
            #         if np.shape(InfReg_all)[1] < COST_MAX:
            #             InfReg_all = np.c_[InfReg_all, np.inf*np.ones( (np.shape(InfReg_all)[0], COST_MAX - np.shape(InfReg_all)[1]) )]
            #         else:
            #             InfReg_all = InfReg_all[:,:COST_MAX]


            if plot:
                index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))
                plot_cost -= COST_INI

                temp_index = BO_methods.index(method)

                method = method.replace('ECI', 'EIC')
                method = method.replace('_Q='+str(q), '')
                method = method.replace('_', '-')

                method = method.replace('MESC', 'CMES')
                method = method.replace('-LB', '-IBO')
                if Parallel:
                    method = method.replace('P', 'P-')

                if average:
                    InfReg_ave = np.mean(InfReg_all, axis=0)

                    line = '-'
                    if '1' in method or '50' in method:
                        if 'IBO' in method:
                            if '1' in method:
                                # line = '--'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][6]
                                method='IBO1'
                            else:
                                # line = ':'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][7]
                                method='IBO50'
                        else:
                            if '1' in method:
                                # line = '--'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][8]
                            else:
                                # line = ':'
                                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][9]
                        plt.plot(plot_cost[index], InfReg_ave[index], linestyle=line, color=c, label=method)
                    elif 'CMES' in method:
                        if 'IBO' in method:
                            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
                        else:
                            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]

                        plt.plot(plot_cost[index], InfReg_ave[index], linestyle=line, color=c, label=method)
                    else:
                        if 'PESC' in method:
                            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][4]
                        else:
                            c = None
                        plt.plot(plot_cost[index], InfReg_ave[index], linestyle=line, color=c, label=method)
                else:
                    InfReg_median = np.median(InfReg_all, axis=0)
                    InfReg_1_4 = np.quantile(InfReg_all, 1/4., axis=0)
                    InfReg_3_4 = np.quantile(InfReg_all, 3/4., axis=0)
                    # plt.plot(plot_cost[index], InfReg_median[index], label=method)
                    # plt.fill_between(plot_cost[index], InfReg_1_4[index], InfReg_3_4[index], alpha=0.3)

                    error_bar_plot = np.r_[np.atleast_2d((InfReg_median - InfReg_1_4)[index]), np.atleast_2d((InfReg_3_4 - InfReg_median)[index])]
                    plt.errorbar(plot_cost[index], InfReg_median[index], yerr=error_bar_plot, errorevery=5, capsize=3, elinewidth=1, label=method)

        # if not Parallel and func_name in ['const_HartMann6', 'Gardner1', 'Gardner2', 'Gramacy']:
        #     plt.vlines(-10, 5, 10, color='grey', linestyle='dotted', label='50')
        #     plt.vlines(-10, 5, 10, color='grey', linestyle='dashed', label='1')

        func_name = func_name.replace('const_', '')
        func_name = func_name.replace('_', ' ')
        if func_name == ('HartMann6'):
            func_name = 'Hartmann6'

        # if not Parallel:
        #     if func_name=='Gardner1':
        #         plt.title('(b) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        #     elif func_name=='Hartmann6':
        #         plt.title('(c) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        #     elif func_name=='G7':
        #         plt.title('(d) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        #     elif func_name=='G10':
        #         plt.title('(e) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        #     else:
        #         plt.title(func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        # else:
        #     if func_name=='G7':
        #         plt.title('(a) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        #     elif func_name=='G10':
        #         plt.title('(b) '+func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        #     else:
        #         plt.title(func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')

        # if 'Reactor' in func_name:
        #     if Parallel:
        #         plt.title('Reactor network' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        #     else:
        #         plt.title('(f) Reactor network' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')


        plt.title(func_name + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')


        # if 'Weighted' in func_name:
        #     plt.title('Speed reducer' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        # if 'Multiple' in func_name:
        #     plt.title('Disk clutch brake' + ' (d='+str(test_func.d)+')'  + '(C='+str(test_func.C)+')')
        # plt.xlim(COST_INI, COST_MAX)
        plt.xlim(COST_INI - COST_INI, COST_MAX - COST_INI - 10)

        # if re.match('G\d+', func_name):
        #     if func_name=='G1':
        #         plt.ylim(10**-3, 25)
        #     elif func_name=='G4':
        #         plt.ylim(10**-2, 10**4)
        #     elif func_name=='G7':
        #         plt.ylim(10**-1, 10**4)
        #     elif func_name=='G9':
        #         plt.ylim(10, 2*10**7)
        #     elif func_name=='G10':
        #         plt.ylim(20, 3*10**4)
        #     elif func_name=='G18':
        #         plt.ylim(10**-2, 5*10**2)
        # elif func_name=='ReactorNetworkDesign':
        #     plt.ylim(1e-3, 0.5)
        # elif 'Himmelblau' in func_name:
        #     plt.ylim(1e-2, 2*1e3)
        # elif 'Multiple' in func_name:
        #     plt.ylim(1e-8, 10)
        # elif 'Reducer' in func_name:
        #     plt.ylim(8*1e-3, 3*1e1)
        # else:
        #     plt.ylim(plot_min, plot_max)

        # plt.yscale('log')
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xlabel('Iteration')
        plt.ylabel('Pertantage of get feasible sample')
        # if func_name in ['Gardner1', 'Gramacy']:
        #     plt.legend(loc='best', ncol=NCOL)
        # if func_name == 'G7' and Parallel:
        #     plt.legend(loc='best', ncol=NCOL)
        plt.legend(loc='best', ncol=NCOL)
        if func_name == 'G10' and not Parallel:
            plt.legend(loc='lower left', ncol=1)
        # elif re.match('G\d+', func_name):
        #     plt.legend(loc='best')
        # plt.legend(loc='best')
        plt.tight_layout()

        plt.savefig('Results_Inf_'+func_name+'_feasible_rate.pdf')
        plt.close()




def print_rank_correlation():
    Parallel = False
    q = 3
    STR_NUM_WORKER = str(q)

    seeds_num = 10
    seeds = np.arange(seeds_num)
    func_names = ['Gardner1', 'Gardner2', 'Gramacy', 'const_HartMann6', 'G1', 'G7', 'G9', 'G10', "ReactorNetworkDesign"]

    if Parallel:
        BO_methods = ['PECI_'+STR_NUM_WORKER, 'PTSC_'+STR_NUM_WORKER, 'PMESC_'+STR_NUM_WORKER, 'PMESC_LB_'+STR_NUM_WORKER]
    else:
        BO_methods = ['MESC1', 'MESC_LB1', 'MESC', 'MESC_LB', 'MESC50', 'MESC_LB50']
        BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB', 'PESC']
        BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB']

    for i, func_name in enumerate(func_names):
        SimReg_ave_list = list()
        for method in BO_methods:
            GLOBAL_MAX = None
            GLOBAL_MIN = None
            test_func = eval('test_functions.'+func_name)()

            if func_name=='const_HartMann6':
                COST_INI = 5 * test_func.d
                COST_MAX = COST_INI + 200
            elif re.match('G\d+', func_name):
                COST_INI = 10
                COST_MAX = COST_INI + 50
            elif test_func.d == 2:
                COST_INI = 5
                COST_MAX = COST_INI + 30
            else:
                COST_INI = np.min([25, 5 * test_func.d])
                COST_MAX = COST_INI + 50

            COST_MAX += 10
            plot_cost = np.arange(COST_MAX)
            SimReg_all = np.ones((seeds_num, COST_MAX)) * np.inf
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if func_name=='Gardner1':
                GLOBAL_MAX = 1.8887513614906686
                GLOBAL_MIN = -2.0
            if func_name=='Gardner2':
                GLOBAL_MAX = - 0.2532358975009539
                GLOBAL_MIN = - 7.0
            if func_name=='Gramacy':
                GLOBAL_MAX = - 0.5997880520093705
                GLOBAL_MIN = - 2.0
            if func_name=='const_HartMann6':
                GLOBAL_MAX = 3.322368011415267
                GLOBAL_MIN = 2.8124505439686544e-08

            if re.match('G\d+', func_name):
                GLOBAL_MAX = test_func.f_star
                GLOBAL_MIN = test_func.f_min

            if func_name=='MultipleDiskClutchBrakeDesign':
                GLOBAL_MAX = -0.2352424579008037
                GLOBAL_MIN = -6.2486277879901
            if func_name=='ReactorNetworkDesign':
                GLOBAL_MAX = 0.3898703631283752
                GLOBAL_MIN = 0.0
            if func_name=='WeightedMinimizationOfSpeedReducer':
                GLOBAL_MAX = -2994.4244746545537
                GLOBAL_MIN = -7144.667944998401
            if func_name=='Himmelblau_Function':
                GLOBAL_MAX = 30665.103087618983
                GLOBAL_MIN = 22302.761885500004
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'SimReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'SimReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)
                            '''
                            横軸をiterationに変更
                            '''
                            cost = cost / (test_func.C + 1)
                            # print(func_name, method, np.size(cost))

                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg==None] = GLOBAL_MIN

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(SimReg):
                                if j+1 == np.size(cost):
                                    break
                                else:
                                    SimReg_all[seed, int(cost[j]) : int(cost[j+1])] = SimReg[j]
                else:
                    plot=False
            # 各関数でRegretに変換
            # if GLOBAL_MAX < np.max(SimReg_all[np.logical_not(np.isinf(SimReg_all))]):
            #     print(func_name, np.max(SimReg_all[np.logical_not(np.isinf(SimReg_all))]))
            SimReg_all = GLOBAL_MAX - SimReg_all
            # print(SimReg_all[np.logical_not(np.isinf(SimReg_all))][SimReg_all[np.logical_not(np.isinf(SimReg_all))] < 0])
            # SimReg_all[SimReg_all < 0] = 0


            if plot:
                temp_index = BO_methods.index(method)
                index = np.logical_not(np.any(np.isinf(SimReg_all), axis=0))
                method = method.replace('ECI', 'EIC')
                method = method.replace('_Q='+str(q), '')
                method = method.replace('_', '-')

                method = method.replace('MESC', 'CMES')
                method = method.replace('-LB', '-IBO')
                plot_cost -= COST_INI



                SimReg_ave = np.mean(SimReg_all, axis=0)
                SimReg_se = np.sqrt(np.sum((SimReg_all - SimReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)

            # SimReg_ave_list.append( SimReg_ave[index][:COST_MAX - COST_INI - 10] )
            SimReg_ave_list.append( SimReg_all[:,index][:,:COST_MAX - COST_INI - 10] )

        InfReg_ave_list = list()

        for method in BO_methods:
            plot_cost = np.arange(COST_MAX)
            InfReg_all = np.ones((seeds_num, COST_MAX)) * np.inf
            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)
                            cost = cost / (test_func.C + 1)

                        with open(temp_path + 'InfReg.pickle', 'rb') as f:
                            InfReg = pickle.load(f)
                            InfReg[InfReg==None] = GLOBAL_MIN

                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg==None] = GLOBAL_MIN
                        InfReg[InfReg < SimReg] = SimReg[InfReg < SimReg]

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(InfReg):
                                if j+1 == np.size(cost):
                                    break
                                else:
                                    InfReg_all[seed, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                else:
                    plot=False


            InfReg_all = GLOBAL_MAX - InfReg_all

            if method == 'PESC':
                if os.path.exists('PESC_results/'+func_name+'.csv'):
                    plot=True
                    InfReg_all = np.loadtxt('PESC_results/'+func_name+'.csv', delimiter=',')
                    InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
                    if np.shape(InfReg_all)[1] < COST_MAX:
                        InfReg_all = np.c_[InfReg_all, np.inf*np.ones( (np.shape(InfReg_all)[0], COST_MAX - np.shape(InfReg_all)[1]) )]
                    else:
                        InfReg_all = InfReg_all[:,:COST_MAX]


            if plot:
                index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))
                plot_cost -= COST_INI

                temp_index = BO_methods.index(method)

                method = method.replace('ECI', 'EIC')
                method = method.replace('_Q='+str(q), '')
                method = method.replace('_', '-')

                method = method.replace('MESC', 'CMES')
                method = method.replace('-LB', '-IBO')
                if Parallel:
                    method = method.replace('P', 'P-')

                InfReg_ave = np.mean(InfReg_all, axis=0)
                InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)

            # InfReg_ave_list.append(InfReg_ave[index][:COST_MAX - COST_INI - 10])
            InfReg_ave_list.append(InfReg_all[:,index][:,:COST_MAX - COST_INI - 10])


        # print(np.vstack(SimReg_ave_list))
        # print(np.vstack(InfReg_ave_list))
        # print(np.shape(np.vstack(SimReg_ave_list)))
        # print(np.shape(np.vstack(InfReg_ave_list)))


        SimReg_allmethod = np.vstack(SimReg_ave_list)
        InfReg_allmethod = np.vstack(InfReg_ave_list)

        rho_list = list()
        for k in range(COST_MAX - COST_INI - 10 - 1):
            rho, p_val = stats.spearmanr(SimReg_allmethod[:,k+1], InfReg_allmethod[:,k+1])
            rho_list.append(rho)

            # print(SimReg_allmethod[:,k+1], InfReg_allmethod[:,k+1])
            # print(np.argsort(SimReg_allmethod[:,k+1]), np.argsort(InfReg_allmethod[:,k+1]))
            # print(rho)
            # exit()

        # print(np.mean(np.array(rho_list)[np.logical_not(np.isnan(np.array(rho_list)))]), rho_list)


        print(func_name, np.mean(np.array(rho_list)[np.logical_not(np.isnan(np.array(rho_list)))]))


if __name__ == '__main__':
    plot_toy(Parallel=False, average=True)
    plot_toy(q=3, Parallel=True, average=True)

    # plot_toy(Parallel=False, average=False)
    # plot_toy(q=3, Parallel=True, average=False)
    # for q in [2,3,4]:
    #     plot_toy(q=q, Parallel=True, average=False)

    # plot_feasible_rate(average=True)
    # print_rank_correlation()
