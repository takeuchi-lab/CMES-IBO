# -*- coding: utf-8 -*-
import os
import sys
import pickle
import copy

import numpy as np
import numpy.matlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from cycler import cycler
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_functions"))
import test_functions

# plt.rcParams['pdf.fonttype'] = 42 # Type3font回避
# plt.rcParams['ps.fonttype'] = 42 # Type3font回避
'''
linuxではバグるため回避
'''
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

def plot_synthetic(q=4, Parallel=False, average=False, each_plot=False):
    if average:
        str_ave = '_average'
    else:
        str_ave = ''
    STR_NUM_WORKER = 'Q='+str(q)
    plot_max = 4.
    # plot_max = 1
    plot_min = 5*1e-1
    plot_min = 2*1e-4
    # plot_min = 0

    if Parallel:
        NCOL=2
    else:
        NCOL=1

    seeds_num = 10
    seeds = np.arange(seeds_num)

    func_names = ['const_SynFun_ell=_0.2-d=2-seed0', 'const_SynFun_ell=_0.2-d=2-seed1', 'const_SynFun_ell=_0.2-d=2-seed2', 'const_SynFun_ell=_0.2-d=2-seed3', 'const_SynFun_ell=_0.2-d=2-seed4', 'const_SynFun_ell=_0.2-d=2-seed5', 'const_SynFun_ell=_0.2-d=2-seed6', 'const_SynFun_ell=_0.2-d=2-seed7', 'const_SynFun_ell=_0.2-d=2-seed8', 'const_SynFun_ell=_0.2-d=2-seed9']

    func_compress_names = ['const_SynFun_compress_ell=_0.2-d=2-seed0', 'const_SynFun_compress_ell=_0.2-d=2-seed1', 'const_SynFun_compress_ell=_0.2-d=2-seed2', 'const_SynFun_compress_ell=_0.2-d=2-seed3', 'const_SynFun_compress_ell=_0.2-d=2-seed4', 'const_SynFun_compress_ell=_0.2-d=2-seed5', 'const_SynFun_compress_ell=_0.2-d=2-seed6', 'const_SynFun_compress_ell=_0.2-d=2-seed7', 'const_SynFun_compress_ell=_0.2-d=2-seed8', 'const_SynFun_compress_ell=_0.2-d=2-seed9']
    BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB', 'PESC']
    BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB']


    if 'SynFun' in func_names[0]:
        if 'compress' in func_names[0]:
            test_func = eval('test_functions.const_SynFun_compress')()
        else:
            test_func = eval('test_functions.const_SynFun')()

        if 'd=2' in func_names[0]:
            input_dim = 2
        elif 'd=3' in func_names[0]:
            input_dim = 3
            # test_func.C = 2
        elif 'd=4' in func_names[0]:
            input_dim = 4
        elif 'd=5' in func_names[0]:
            input_dim = 5
            # test_func.C = 5

        # COST_INI = (test_func.C + 1) * 5 * input_dim
        # COST_MAX = (test_func.C + 1) * 200
        COST_INI = 3
        COST_MAX = COST_INI + 30 + 10

        if 'pool' in func_names[0] or 'test' in func_names[0]:
            COST_INI = 3
            COST_MAX = 50 * input_dim

    if not Parallel:
        fig = plt.figure(figsize=(5.5, 7))
    # Simple Regret plot
    # plt.subplot(1, 2, 1)
    for k, method in enumerate(BO_methods):
        plot_cost = np.arange(COST_MAX)
        SimReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf

        test_func = eval('test_functions.const_SynFun')()
        for i, func_name in enumerate(func_names):
            GLOBAL_MAX = None
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if func_name=='const_SynFun_ell=_0.2-d=2-seed0' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed0':
                GLOBAL_MAX = 1.3495785751759248
                GLOBAL_MIN = -1.9995734315897211
            if func_name=='const_SynFun_ell=_0.2-d=2-seed1' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed1':
                GLOBAL_MAX = 1.714594741371426
                GLOBAL_MIN = -3.2542125994062037
            if func_name=='const_SynFun_ell=_0.2-d=2-seed2'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed2':
                GLOBAL_MAX = 1.305631394526752
                GLOBAL_MIN = -2.0967492742119482
            if func_name=='const_SynFun_ell=_0.2-d=2-seed3'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed3':
                GLOBAL_MAX = 0.7974972985949693
                GLOBAL_MIN = -2.4217744019408416
            if func_name=='const_SynFun_ell=_0.2-d=2-seed4' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed4':
                GLOBAL_MAX = 0.7992271587208615
                GLOBAL_MIN = -3.0197591002041078
            if func_name=='const_SynFun_ell=_0.2-d=2-seed5'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed5':
                GLOBAL_MAX = 1.3831150914827797
                GLOBAL_MIN = -1.9623293608602748
            if func_name=='const_SynFun_ell=_0.2-d=2-seed6' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed6':
                GLOBAL_MAX = 1.4534487599168746
                GLOBAL_MIN = -2.1819490752145194
            if func_name=='const_SynFun_ell=_0.2-d=2-seed7' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed7':
                GLOBAL_MAX = 1.433866466509011
                GLOBAL_MIN = -1.1690107268053422
            if func_name=='const_SynFun_ell=_0.2-d=2-seed8' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed8':
                GLOBAL_MAX = 1.2952045360134905
                GLOBAL_MIN = -0.9844915170742703
            if func_name=='const_SynFun_ell=_0.2-d=2-seed9'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed9':
                GLOBAL_MAX = 0.5376284714835798
                GLOBAL_MIN = -2.375095294446008

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'SimReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'SimReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)

                            # print(method, cost[-1])
                            cost = cost / (test_func.C + 1)

                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg==None] = GLOBAL_MIN

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(SimReg):
                                if j+1 == np.size(cost):
                                    SimReg_all[seed+i*seeds_num, int(cost[j])] = SimReg[j]
                                    # break
                                else:
                                    SimReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = SimReg[j]
                else:
                    plot=False
            # # 各関数でRegretに変換
            # print(SimReg_all[i*seeds_num:(i+1)*seeds_num, -1])
            SimReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - SimReg_all[i*seeds_num:(i+1)*seeds_num, :]
        if plot:
            plot_cost -= COST_INI
            index = np.logical_not(np.any(np.isinf(SimReg_all), axis=0))
            method_label = copy.copy(method)
            method_label = method_label.replace('ECI', 'EIC')
            method_label = method_label.replace('_Q='+str(q), '')
            method_label = method_label.replace('_', '-')

            if average:
                SimReg_ave = np.mean(SimReg_all, axis=0)
                SimReg_se = np.sqrt(np.sum((SimReg_all - SimReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))
                # plt.plot(plot_cost[index], SimReg_ave[index], label=method_label)
                # plt.fill_between(plot_cost[index], SimReg_ave[index] - SimReg_se[index], SimReg_ave[index] + SimReg_se[index], alpha=0.3)
                if each_plot:
                    if 'MESC-LB' in method_label:
                        for j in range(len(func_names)):
                            SimReg_ave = np.mean(SimReg_all[j:(j+1)*10, :], axis=0)
                            SimReg_se = np.sqrt(np.sum((SimReg_all[j:(j+1)*10, :] - SimReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                            plt.errorbar(plot_cost[index], SimReg_ave[index], yerr=SimReg_se[index], errorevery=5, elinewidth=1, label=method_label+'-'+str(j))
                else:
                    plt.errorbar(plot_cost[index], SimReg_ave[index], yerr=SimReg_se[index], errorevery=(k, 5), elinewidth=3, label=method_label)



            else:
                SimReg_median = np.median(SimReg_all, axis=0)
                SimReg_1_4 = np.quantile(SimReg_all, 1/4., axis=0)
                SimReg_3_4 = np.quantile(SimReg_all, 3/4., axis=0)
                # plt.plot(plot_cost[index], SimReg_median[index], label=method)
                # plt.fill_between(plot_cost[index], SimReg_1_4[index], SimReg_3_4[index], alpha=0.3)

                error_bar_plot = np.r_[np.atleast_2d((SimReg_median - SimReg_1_4)[index]), np.atleast_2d((SimReg_3_4 - SimReg_median)[index])]
                plt.errorbar(plot_cost[index], SimReg_median[index], yerr=error_bar_plot, errorevery=5, elinewidth=1, label=method)
    func_name = func_name.replace('const_', '')
    # plt.title('Synthetic Function' + ' (d='+str(input_dim)+')' + ' (C='+str(test_func.C)+')')
    plt.title('(a) Synthetic' + ' (C='+str(test_func.C)+')')
    # plt.title('Synthetic Function (d=3)')
    plt.xlim(COST_INI - COST_INI, COST_MAX - COST_INI - 10)
    # plt.ylim(plot_min, plot_max)
    plt.yscale('log')
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.xlabel('Iteration')
    plt.ylabel('Utility Gap')
    plt.legend(loc='best')
    plt.tight_layout()

    if Parallel:
        plt.savefig('Results_Sim_Syn_log'+str_ave+'_'+STR_NUM_WORKER+'_d='+str(input_dim)+'.pdf')
    elif each_plot:
        plt.savefig('Results_Sim_Syn_log'+str_ave+'_each'+'_d='+str(input_dim)+'.pdf')
    else:
        plt.savefig('Results_Sim_Syn_log'+str_ave+'_d='+str(input_dim)+'.pdf')
    plt.close()

    if not Parallel:
        fig = plt.figure(figsize=(5.5, 7))
    # Inference Regret plot
    # plt.subplot(1, 2, 2)
    for k, method in enumerate(BO_methods):
        plot_cost = np.arange(COST_MAX)
        InfReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf
        test_func = eval('test_functions.const_SynFun')()
        for i, func_name in enumerate(func_names):
            GLOBAL_MAX = None
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if func_name=='const_SynFun_ell=_0.2-d=2-seed0' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed0':
                GLOBAL_MAX = 1.3495785751759248
                GLOBAL_MIN = -1.9995734315897211
            if func_name=='const_SynFun_ell=_0.2-d=2-seed1' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed1':
                GLOBAL_MAX = 1.714594741371426
                GLOBAL_MIN = -3.2542125994062037
            if func_name=='const_SynFun_ell=_0.2-d=2-seed2'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed2':
                GLOBAL_MAX = 1.305631394526752
                GLOBAL_MIN = -2.0967492742119482
            if func_name=='const_SynFun_ell=_0.2-d=2-seed3'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed3':
                GLOBAL_MAX = 0.7974972985949693
                GLOBAL_MIN = -2.4217744019408416
            if func_name=='const_SynFun_ell=_0.2-d=2-seed4' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed4':
                GLOBAL_MAX = 0.7992271587208615
                GLOBAL_MIN = -3.0197591002041078
            if func_name=='const_SynFun_ell=_0.2-d=2-seed5'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed5':
                GLOBAL_MAX = 1.3831150914827797
                GLOBAL_MIN = -1.9623293608602748
            if func_name=='const_SynFun_ell=_0.2-d=2-seed6' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed6':
                GLOBAL_MAX = 1.4534487599168746
                GLOBAL_MIN = -2.1819490752145194
            if func_name=='const_SynFun_ell=_0.2-d=2-seed7' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed7':
                GLOBAL_MAX = 1.433866466509011
                GLOBAL_MIN = -1.1690107268053422
            if func_name=='const_SynFun_ell=_0.2-d=2-seed8' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed8':
                GLOBAL_MAX = 1.2952045360134905
                GLOBAL_MIN = -0.9844915170742703
            if func_name=='const_SynFun_ell=_0.2-d=2-seed9'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed9':
                GLOBAL_MAX = 0.5376284714835798
                GLOBAL_MIN = -2.375095294446008

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            if method != 'PESC':
                for seed in seeds:
                    temp_path = result_path + method + '/seed=' + str(seed) + '/'
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = pickle.load(f)
                                cost = cost / (test_func.C + 1)

                            with open(temp_path + 'InfReg.pickle', 'rb') as f:
                                InfReg = pickle.load(f)
                                InfReg[InfReg == None] = GLOBAL_MIN
                            with open(temp_path + 'SimReg.pickle', 'rb') as f:
                                SimReg = pickle.load(f)
                                SimReg[SimReg==None] = GLOBAL_MIN
                            InfReg[InfReg < SimReg] = SimReg[InfReg < SimReg]

                            for j in range(np.size(cost)):
                                if j+1 <= np.size(InfReg):
                                    if j+1 == np.size(cost):
                                        InfReg_all[seed+i*seeds_num, int(cost[j])] = InfReg[j]
                                        # break
                                    else:
                                        InfReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                    else:
                        plot=False
                # 各関数でRegretに変換
                InfReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - InfReg_all[i*seeds_num:(i+1)*seeds_num, :]
            else:
                pesc_path = 'PESC_results/const_SynFun'+str(i)+'.csv'
                if os.path.exists(pesc_path):
                    plot=True
                    if np.all(np.isinf(InfReg_all)):
                        InfReg_all = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
                    else:
                        InfReg_tmp = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_tmp = np.c_[np.inf*np.ones((np.shape(InfReg_tmp)[0], COST_INI)), InfReg_tmp]
                        InfReg_all = np.r_[InfReg_all, InfReg_tmp]


        if plot:
            plot_cost -= COST_INI
            index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))
            method_label = copy.copy(method)
            method_label = method_label.replace('ECI', 'EIC')
            method_label = method_label.replace('_Q='+str(q), '')
            method_label = method_label.replace('_', '-')

            method_label = method_label.replace('MESC', 'CMES')
            method_label = method_label.replace('-LB', '-IBO')

            if Parallel:
                method_label = method_label.replace('P', 'P-')

            if average:
                InfReg_ave = np.mean(InfReg_all, axis=0)
                InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))

                # plt.plot(plot_cost[index], InfReg_ave[index], label=method_label)
                # plt.fill_between(plot_cost[index], InfReg_ave[index] - InfReg_se[index], InfReg_ave[index] + InfReg_se[index], alpha=0.3)
                if each_plot:
                    if 'MESC-LB' in method_label:
                        for j in range(len(func_names)):
                            InfReg_ave = np.mean(InfReg_all[j:(j+1)*10, :], axis=0)
                            InfReg_se = np.sqrt(np.sum((InfReg_all[j:(j+1)*10, :] - InfReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=5, elinewidth=1, label=method_label+'-'+str(j))
                else:
                    plot_cost = plot_cost[:np.size(index)]
                    plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(k, 5), elinewidth=3.1, label=method_label, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][k])

            else:
                InfReg_median = np.median(InfReg_all, axis=0)
                InfReg_1_4 = np.quantile(InfReg_all, 1/4., axis=0)
                InfReg_3_4 = np.quantile(InfReg_all, 3/4., axis=0)

                error_bar_plot = np.r_[np.atleast_2d((InfReg_median- InfReg_1_4)[index]), np.atleast_2d((InfReg_3_4 - InfReg_median)[index])]
                plt.errorbar(plot_cost[index], InfReg_median[index], yerr=error_bar_plot, errorevery=5, elinewidth=1, label=method)

        plot_cost = np.arange(COST_MAX)
        InfReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf
        test_func = eval('test_functions.const_SynFun_compress')()
        for i, func_name in enumerate(func_compress_names):
            GLOBAL_MAX = None
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if func_name=='const_SynFun_ell=_0.2-d=2-seed0' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed0':
                GLOBAL_MAX = 1.3495785751759248
                GLOBAL_MIN = -1.9995734315897211
            if func_name=='const_SynFun_ell=_0.2-d=2-seed1' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed1':
                GLOBAL_MAX = 1.714594741371426
                GLOBAL_MIN = -3.2542125994062037
            if func_name=='const_SynFun_ell=_0.2-d=2-seed2'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed2':
                GLOBAL_MAX = 1.305631394526752
                GLOBAL_MIN = -2.0967492742119482
            if func_name=='const_SynFun_ell=_0.2-d=2-seed3'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed3':
                GLOBAL_MAX = 0.7974972985949693
                GLOBAL_MIN = -2.4217744019408416
            if func_name=='const_SynFun_ell=_0.2-d=2-seed4' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed4':
                GLOBAL_MAX = 0.7992271587208615
                GLOBAL_MIN = -3.0197591002041078
            if func_name=='const_SynFun_ell=_0.2-d=2-seed5'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed5':
                GLOBAL_MAX = 1.3831150914827797
                GLOBAL_MIN = -1.9623293608602748
            if func_name=='const_SynFun_ell=_0.2-d=2-seed6' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed6':
                GLOBAL_MAX = 1.4534487599168746
                GLOBAL_MIN = -2.1819490752145194
            if func_name=='const_SynFun_ell=_0.2-d=2-seed7' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed7':
                GLOBAL_MAX = 1.433866466509011
                GLOBAL_MIN = -1.1690107268053422
            if func_name=='const_SynFun_ell=_0.2-d=2-seed8' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed8':
                GLOBAL_MAX = 1.2952045360134905
                GLOBAL_MIN = -0.9844915170742703
            if func_name=='const_SynFun_ell=_0.2-d=2-seed9'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed9':
                GLOBAL_MAX = 0.5376284714835798
                GLOBAL_MIN = -2.375095294446008

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            if method != 'PESC':
                for seed in seeds:
                    temp_path = result_path + method + '/seed=' + str(seed) + '/'
                    # print(temp_path)
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = pickle.load(f)
                                cost = cost / (test_func.C + 1)

                            with open(temp_path + 'InfReg.pickle', 'rb') as f:
                                InfReg = pickle.load(f)
                                InfReg[InfReg == None] = GLOBAL_MIN
                            with open(temp_path + 'SimReg.pickle', 'rb') as f:
                                SimReg = pickle.load(f)
                                SimReg[SimReg==None] = GLOBAL_MIN
                            InfReg[InfReg < SimReg] = SimReg[InfReg < SimReg]

                            for j in range(np.size(cost)):
                                if j+1 <= np.size(InfReg):
                                    if j+1 == np.size(cost):
                                        InfReg_all[seed+i*seeds_num, int(cost[j])] = InfReg[j]
                                        # break
                                    else:
                                        InfReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                    else:
                        plot=False
                # 各関数でRegretに変換
                InfReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - InfReg_all[i*seeds_num:(i+1)*seeds_num, :]
            else:
                pesc_path = 'PESC_results/const_SynFun'+str(i)+'.csv'
                if os.path.exists(pesc_path):
                    plot=True
                    if np.all(np.isinf(InfReg_all)):
                        InfReg_all = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
                    else:
                        InfReg_tmp = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_tmp = np.c_[np.inf*np.ones((np.shape(InfReg_tmp)[0], COST_INI)), InfReg_tmp]
                        InfReg_all = np.r_[InfReg_all, InfReg_tmp]
                else:
                    plot=False


        if plot:
            plot_cost -= COST_INI
            index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))
            method = method.replace('ECI', 'EIC')
            method = method.replace('_Q='+str(q), '')
            method = method.replace('_', '-')


            method = method.replace('MESC', 'CMES')
            method = method.replace('-LB', '-IBO')
            if Parallel:
                method = method.replace('P', 'P-')

            if average:
                InfReg_ave = np.mean(InfReg_all, axis=0)
                InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))

                # plt.plot(plot_cost[index], InfReg_ave[index], label=method)
                # plt.fill_between(plot_cost[index], InfReg_ave[index] - InfReg_se[index], InfReg_ave[index] + InfReg_se[index], alpha=0.3)
                if each_plot:
                    if 'MESC-LB' in method:
                        for j in range(len(func_names)):
                            InfReg_ave = np.mean(InfReg_all[j:(j+1)*10, :], axis=0)
                            InfReg_se = np.sqrt(np.sum((InfReg_all[j:(j+1)*10, :] - InfReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=5, elinewidth=1, label=method+'-'+str(j))
                else:
                    plot_cost = plot_cost[:np.size(index)]
                    print(k*5, 5)
                    plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(k, 5), elinewidth=3, linestyle='--', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][k])

            else:
                InfReg_median = np.median(InfReg_all, axis=0)
                InfReg_1_4 = np.quantile(InfReg_all, 1/4., axis=0)
                InfReg_3_4 = np.quantile(InfReg_all, 3/4., axis=0)

                error_bar_plot = np.r_[np.atleast_2d((InfReg_median- InfReg_1_4)[index]), np.atleast_2d((InfReg_3_4 - InfReg_median)[index])]
                plt.errorbar(plot_cost[index], InfReg_median[index], yerr=error_bar_plot, errorevery=5, elinewidth=1, linestyle='--')


    test_func = eval('test_functions.const_SynFun')()
    func_name = func_name.replace('const_', '')
    plt.title('(a) Synthetic ' + ' (C='+str(test_func.C)+')')
    # plt.title('Synthetic ' + ' (d='+str(input_dim)+')' + '(C='+str(test_func.C)+')')
    # plt.title('Synthetic ' + ' (C='+str(test_func.C)+')')
    # plt.title('Synthetic Function (d=3)')
    plt.xlim(COST_INI - COST_INI, COST_MAX - COST_INI - 10)
    plt.ylim(plot_min, plot_max)
    plt.yscale('log')
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.xlabel('Iteration')
    plt.ylabel('Utility Gap')
    plt.legend(loc='best', ncol=NCOL)
    plt.tight_layout()

    if Parallel:
        plt.savefig('Results_Inf_Syn_log'+str_ave+'_'+STR_NUM_WORKER+'_d='+str(input_dim)+'.pdf')
    elif each_plot:
        plt.savefig('Results_Inf_Syn_log'+str_ave+'_each'+'_d='+str(input_dim)+'.pdf')
    else:
        plt.savefig('Results_Inf_Syn_log'+str_ave+'_d='+str(input_dim)+'.pdf')
    plt.close()




def plot_feasible_rate(q=4, Parallel=False, average=False, each_plot=False):
    if average:
        str_ave = '_average'
    else:
        str_ave = ''
    STR_NUM_WORKER = 'Q='+str(q)
    plot_max = 4.
    # plot_max = 1
    plot_min = 5*1e-1
    plot_min = 1e-4
    # plot_min = 0

    if Parallel:
        NCOL=2
    else:
        NCOL=2

    seeds_num = 10
    seeds = np.arange(seeds_num)

    func_names = ['const_SynFun_ell=_0.2-d=2-seed0', 'const_SynFun_ell=_0.2-d=2-seed1', 'const_SynFun_ell=_0.2-d=2-seed2', 'const_SynFun_ell=_0.2-d=2-seed3', 'const_SynFun_ell=_0.2-d=2-seed4', 'const_SynFun_ell=_0.2-d=2-seed5', 'const_SynFun_ell=_0.2-d=2-seed6', 'const_SynFun_ell=_0.2-d=2-seed7', 'const_SynFun_ell=_0.2-d=2-seed8', 'const_SynFun_ell=_0.2-d=2-seed9']

    func_compress_names = ['const_SynFun_compress_ell=_0.2-d=2-seed0', 'const_SynFun_compress_ell=_0.2-d=2-seed1', 'const_SynFun_compress_ell=_0.2-d=2-seed2', 'const_SynFun_compress_ell=_0.2-d=2-seed3', 'const_SynFun_compress_ell=_0.2-d=2-seed4', 'const_SynFun_compress_ell=_0.2-d=2-seed5', 'const_SynFun_compress_ell=_0.2-d=2-seed6', 'const_SynFun_compress_ell=_0.2-d=2-seed7', 'const_SynFun_compress_ell=_0.2-d=2-seed8', 'const_SynFun_compress_ell=_0.2-d=2-seed9']
    BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB'] #, 'PESC']


    if 'SynFun' in func_names[0]:
        if 'compress' in func_names[0]:
            test_func = eval('test_functions.const_SynFun_compress')()
        else:
            test_func = eval('test_functions.const_SynFun')()

        if 'd=2' in func_names[0]:
            input_dim = 2
        elif 'd=3' in func_names[0]:
            input_dim = 3
            # test_func.C = 2
        elif 'd=4' in func_names[0]:
            input_dim = 4
        elif 'd=5' in func_names[0]:
            input_dim = 5
            # test_func.C = 5

        # COST_INI = (test_func.C + 1) * 5 * input_dim
        # COST_MAX = (test_func.C + 1) * 200
        COST_INI = 3
        COST_MAX = COST_INI + 30 + 10

        if 'pool' in func_names[0] or 'test' in func_names[0]:
            COST_INI = 3
            COST_MAX = 50 * input_dim

    for k, method in enumerate(BO_methods):
        plot_cost = np.arange(COST_MAX)
        InfReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf
        test_func = eval('test_functions.const_SynFun')()
        for i, func_name in enumerate(func_names):
            result_path = func_name + '_results/'

            plot=True
            if method != 'PESC':
                for seed in seeds:
                    temp_path = result_path + method + '/seed=' + str(seed) + '/'
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = pickle.load(f)
                                cost = cost / (test_func.C + 1)

                            with open(temp_path + 'InfReg.pickle', 'rb') as f:
                                InfReg = pickle.load(f)
                                InfReg[InfReg != None] = 1
                                InfReg[InfReg == None] = 0
                            with open(temp_path + 'SimReg.pickle', 'rb') as f:
                                SimReg = pickle.load(f)
                                SimReg[SimReg!=None] = 1
                                SimReg[SimReg==None] = 0
                            InfReg[InfReg < SimReg] = SimReg[InfReg < SimReg]

                            for j in range(np.size(cost)):
                                if j+1 <= np.size(InfReg):
                                    if j+1 == np.size(cost):
                                        InfReg_all[seed+i*seeds_num, int(cost[j])] = InfReg[j]
                                        # break
                                    else:
                                        InfReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                    else:
                        plot=False
                # # 各関数でRegretに変換
                # InfReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - InfReg_all[i*seeds_num:(i+1)*seeds_num, :]
            else:
                pesc_path = 'PESC_results/const_SynFun'+str(i)+'.csv'
                if os.path.exists(pesc_path):
                    plot=True
                    if np.all(np.isinf(InfReg_all)):
                        InfReg_all = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
                    else:
                        InfReg_tmp = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_tmp = np.c_[np.inf*np.ones((np.shape(InfReg_tmp)[0], COST_INI)), InfReg_tmp]
                        InfReg_all = np.r_[InfReg_all, InfReg_tmp]


        if plot:
            plot_cost -= COST_INI
            index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))
            method_label = copy.copy(method)
            method_label = method_label.replace('ECI', 'EIC')
            method_label = method_label.replace('_Q='+str(q), '')
            method_label = method_label.replace('_', '-')

            method_label = method_label.replace('MESC', 'CMES')
            method_label = method_label.replace('-LB', '-IBO')

            if Parallel:
                method_label = method_label.replace('P', 'P-')

            if average:
                InfReg_ave = np.mean(InfReg_all, axis=0)
                InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))

                # plt.plot(plot_cost[index], InfReg_ave[index], label=method_label)
                # plt.fill_between(plot_cost[index], InfReg_ave[index] - InfReg_se[index], InfReg_ave[index] + InfReg_se[index], alpha=0.3)
                if each_plot:
                    if 'MESC-LB' in method_label:
                        for j in range(len(func_names)):
                            InfReg_ave = np.mean(InfReg_all[j:(j+1)*10, :], axis=0)
                            InfReg_se = np.sqrt(np.sum((InfReg_all[j:(j+1)*10, :] - InfReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=5, elinewidth=1, label=method_label+'-'+str(j))
                else:
                    plot_cost = plot_cost[:np.size(index)]
                    # plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(k, 5), elinewidth=3.1, label=method_label, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][k])
                plt.plot(plot_cost, InfReg_ave, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][k], label=method_label)

            else:
                InfReg_median = np.median(InfReg_all, axis=0)
                InfReg_1_4 = np.quantile(InfReg_all, 1/4., axis=0)
                InfReg_3_4 = np.quantile(InfReg_all, 3/4., axis=0)

                error_bar_plot = np.r_[np.atleast_2d((InfReg_median- InfReg_1_4)[index]), np.atleast_2d((InfReg_3_4 - InfReg_median)[index])]
                plt.errorbar(plot_cost[index], InfReg_median[index], yerr=error_bar_plot, errorevery=5, elinewidth=1, label=method)

    print(InfReg_ave)
    test_func = eval('test_functions.const_SynFun')()
    func_name = func_name.replace('const_', '')
    # plt.title('(a) Synthetic ' + ' (d='+str(input_dim)+')' + '(C='+str(test_func.C)+')')
    plt.title('Synthetic ' + ' (d='+str(input_dim)+')' + '(C='+str(test_func.C)+')')
    # plt.title('Synthetic Function (d=3)')
    plt.xlim(COST_INI - COST_INI, COST_MAX - COST_INI - 10)
    # plt.ylim(plot_min, plot_max)
    # plt.yscale('log')
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.xlabel('Iteration')
    plt.ylabel('feasible rate')
    plt.legend(loc='best', ncol=NCOL)
    plt.tight_layout()

    plt.savefig('Results_Inf_feasible_rate_Syn'+str_ave+'_d='+str(input_dim)+'.pdf')
    plt.close()



def print_rank_correlation():
    average = True
    q = 3
    if average:
        str_ave = '_average'
    else:
        str_ave = ''
    STR_NUM_WORKER = 'Q='+str(q)

    seeds_num = 10
    seeds = np.arange(seeds_num)

    func_names = ['const_SynFun_ell=_0.2-d=2-seed0', 'const_SynFun_ell=_0.2-d=2-seed1', 'const_SynFun_ell=_0.2-d=2-seed2', 'const_SynFun_ell=_0.2-d=2-seed3', 'const_SynFun_ell=_0.2-d=2-seed4', 'const_SynFun_ell=_0.2-d=2-seed5', 'const_SynFun_ell=_0.2-d=2-seed6', 'const_SynFun_ell=_0.2-d=2-seed7', 'const_SynFun_ell=_0.2-d=2-seed8', 'const_SynFun_ell=_0.2-d=2-seed9']

    BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB'] #, 'PESC']


    if 'SynFun' in func_names[0]:
        if 'compress' in func_names[0]:
            test_func = eval('test_functions.const_SynFun_compress')()
        else:
            test_func = eval('test_functions.const_SynFun')()

        if 'd=2' in func_names[0]:
            input_dim = 2
        elif 'd=3' in func_names[0]:
            input_dim = 3
            # test_func.C = 2
        elif 'd=4' in func_names[0]:
            input_dim = 4
        elif 'd=5' in func_names[0]:
            input_dim = 5
            # test_func.C = 5

        # COST_INI = (test_func.C + 1) * 5 * input_dim
        # COST_MAX = (test_func.C + 1) * 200
        COST_INI = 3
        COST_MAX = COST_INI + 30 + 10

        if 'pool' in func_names[0] or 'test' in func_names[0]:
            COST_INI = 3
            COST_MAX = 50 * input_dim

    SimReg_list = list()
    InfReg_list = list()
    for k, method in enumerate(BO_methods):
        plot_cost = np.arange(COST_MAX)
        SimReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf

        test_func = eval('test_functions.const_SynFun')()
        for i, func_name in enumerate(func_names):
            GLOBAL_MAX = None
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if func_name=='const_SynFun_ell=_0.2-d=2-seed0' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed0':
                GLOBAL_MAX = 1.3495785751759248
                GLOBAL_MIN = -1.9995734315897211
            if func_name=='const_SynFun_ell=_0.2-d=2-seed1' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed1':
                GLOBAL_MAX = 1.714594741371426
                GLOBAL_MIN = -3.2542125994062037
            if func_name=='const_SynFun_ell=_0.2-d=2-seed2'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed2':
                GLOBAL_MAX = 1.305631394526752
                GLOBAL_MIN = -2.0967492742119482
            if func_name=='const_SynFun_ell=_0.2-d=2-seed3'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed3':
                GLOBAL_MAX = 0.7974972985949693
                GLOBAL_MIN = -2.4217744019408416
            if func_name=='const_SynFun_ell=_0.2-d=2-seed4' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed4':
                GLOBAL_MAX = 0.7992271587208615
                GLOBAL_MIN = -3.0197591002041078
            if func_name=='const_SynFun_ell=_0.2-d=2-seed5'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed5':
                GLOBAL_MAX = 1.3831150914827797
                GLOBAL_MIN = -1.9623293608602748
            if func_name=='const_SynFun_ell=_0.2-d=2-seed6' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed6':
                GLOBAL_MAX = 1.4534487599168746
                GLOBAL_MIN = -2.1819490752145194
            if func_name=='const_SynFun_ell=_0.2-d=2-seed7' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed7':
                GLOBAL_MAX = 1.433866466509011
                GLOBAL_MIN = -1.1690107268053422
            if func_name=='const_SynFun_ell=_0.2-d=2-seed8' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed8':
                GLOBAL_MAX = 1.2952045360134905
                GLOBAL_MIN = -0.9844915170742703
            if func_name=='const_SynFun_ell=_0.2-d=2-seed9'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed9':
                GLOBAL_MAX = 0.5376284714835798
                GLOBAL_MIN = -2.375095294446008

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'SimReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'SimReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)

                            # print(method, cost[-1])
                            cost = cost / (test_func.C + 1)

                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg==None] = GLOBAL_MIN

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(SimReg):
                                if j+1 == np.size(cost):
                                    SimReg_all[seed+i*seeds_num, int(cost[j])] = SimReg[j]
                                    # break
                                else:
                                    SimReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = SimReg[j]
                else:
                    plot=False
            # # 各関数でRegretに変換
            # print(SimReg_all[i*seeds_num:(i+1)*seeds_num, -1])
            SimReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - SimReg_all[i*seeds_num:(i+1)*seeds_num, :]
        if plot:
            plot_cost -= COST_INI
            index = np.logical_not(np.any(np.isinf(SimReg_all), axis=0))
            method_label = copy.copy(method)
            method_label = method_label.replace('ECI', 'EIC')
            method_label = method_label.replace('_Q='+str(q), '')
            method_label = method_label.replace('_', '-')


            SimReg_ave = np.mean(SimReg_all, axis=0)
            SimReg_se = np.sqrt(np.sum((SimReg_all - SimReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))


        # SimReg_list.append(SimReg_ave[index][:COST_MAX - COST_INI - 10])
        SimReg_list.append(SimReg_all[:,index][:,:COST_MAX - COST_INI - 10])


    for k, method in enumerate(BO_methods):
        plot_cost = np.arange(COST_MAX)
        InfReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf
        test_func = eval('test_functions.const_SynFun')()
        for i, func_name in enumerate(func_names):
            GLOBAL_MAX = None
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if func_name=='const_SynFun_ell=_0.2-d=2-seed0' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed0':
                GLOBAL_MAX = 1.3495785751759248
                GLOBAL_MIN = -1.9995734315897211
            if func_name=='const_SynFun_ell=_0.2-d=2-seed1' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed1':
                GLOBAL_MAX = 1.714594741371426
                GLOBAL_MIN = -3.2542125994062037
            if func_name=='const_SynFun_ell=_0.2-d=2-seed2'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed2':
                GLOBAL_MAX = 1.305631394526752
                GLOBAL_MIN = -2.0967492742119482
            if func_name=='const_SynFun_ell=_0.2-d=2-seed3'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed3':
                GLOBAL_MAX = 0.7974972985949693
                GLOBAL_MIN = -2.4217744019408416
            if func_name=='const_SynFun_ell=_0.2-d=2-seed4' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed4':
                GLOBAL_MAX = 0.7992271587208615
                GLOBAL_MIN = -3.0197591002041078
            if func_name=='const_SynFun_ell=_0.2-d=2-seed5'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed5':
                GLOBAL_MAX = 1.3831150914827797
                GLOBAL_MIN = -1.9623293608602748
            if func_name=='const_SynFun_ell=_0.2-d=2-seed6' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed6':
                GLOBAL_MAX = 1.4534487599168746
                GLOBAL_MIN = -2.1819490752145194
            if func_name=='const_SynFun_ell=_0.2-d=2-seed7' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed7':
                GLOBAL_MAX = 1.433866466509011
                GLOBAL_MIN = -1.1690107268053422
            if func_name=='const_SynFun_ell=_0.2-d=2-seed8' or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed8':
                GLOBAL_MAX = 1.2952045360134905
                GLOBAL_MIN = -0.9844915170742703
            if func_name=='const_SynFun_ell=_0.2-d=2-seed9'  or func_name=='const_SynFun_compress_ell=_0.2-d=2-seed9':
                GLOBAL_MAX = 0.5376284714835798
                GLOBAL_MIN = -2.375095294446008

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            result_path = func_name + '_results/'

            plot=True
            if method != 'PESC':
                for seed in seeds:
                    temp_path = result_path + method + '/seed=' + str(seed) + '/'
                    if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'InfReg.pickle'):
                        if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'InfReg.pickle')>0:
                            with open(temp_path + 'cost.pickle', 'rb') as f:
                                cost = pickle.load(f)
                                cost = cost / (test_func.C + 1)

                            with open(temp_path + 'InfReg.pickle', 'rb') as f:
                                InfReg = pickle.load(f)
                                InfReg[InfReg == None] = GLOBAL_MIN
                            with open(temp_path + 'SimReg.pickle', 'rb') as f:
                                SimReg = pickle.load(f)
                                SimReg[SimReg==None] = GLOBAL_MIN
                            InfReg[InfReg < SimReg] = SimReg[InfReg < SimReg]

                            for j in range(np.size(cost)):
                                if j+1 <= np.size(InfReg):
                                    if j+1 == np.size(cost):
                                        InfReg_all[seed+i*seeds_num, int(cost[j])] = InfReg[j]
                                        # break
                                    else:
                                        InfReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                    else:
                        plot=False
                # 各関数でRegretに変換
                InfReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - InfReg_all[i*seeds_num:(i+1)*seeds_num, :]
            else:
                pesc_path = 'PESC_results/const_SynFun'+str(i)+'.csv'
                if os.path.exists(pesc_path):
                    plot=True
                    if np.all(np.isinf(InfReg_all)):
                        InfReg_all = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
                    else:
                        InfReg_tmp = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_tmp = np.c_[np.inf*np.ones((np.shape(InfReg_tmp)[0], COST_INI)), InfReg_tmp]
                        InfReg_all = np.r_[InfReg_all, InfReg_tmp]


        if plot:
            plot_cost -= COST_INI
            index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))


            InfReg_ave = np.mean(InfReg_all, axis=0)
            InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))

        # InfReg_list.append(InfReg_ave[index][:COST_MAX - COST_INI - 10])
        InfReg_list.append(InfReg_all[:,index][:,:COST_MAX - COST_INI - 10])

    SimReg_allmethod = np.vstack(SimReg_list)
    InfReg_allmethod = np.vstack(InfReg_list)

    print(np.shape(SimReg_allmethod))
    print(np.shape(InfReg_allmethod))

    rho_list = list()
    for k in range(COST_MAX - COST_INI - 10 - 1):
        rho, p_val = stats.spearmanr(SimReg_allmethod[:,k+1], InfReg_allmethod[:,k+1])
        rho_list.append(rho)


    print('SynFun', np.mean(np.array(rho_list)[np.logical_not(np.isnan(np.array(rho_list)))]))






if __name__ == '__main__':
    # plot_synthetic(Parallel=False, average=False)
    # plot_synthetic(q=3, Parallel=True, average=False)

    plot_synthetic(Parallel=False, average=True)
    # plot_synthetic(Parallel=False, average=True, each_plot=True)
    # plot_synthetic(q=3, Parallel=True, average=True)

    # for q in [2,4,8]:
    #     plot_synthetic(q=q, Parallel=True)

    # plot_feasible_rate(Parallel=False, average=True)
    # print_rank_correlation()