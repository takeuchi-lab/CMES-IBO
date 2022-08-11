# -*- coding: utf-8 -*-
import os
import sys
import pickle

import numpy as np
import numpy.matlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_functions"))
import test_functions



# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{sfmath}'



plt.rcParams["font.size"] = 25
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
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


    if Parallel:
        NCOL=2
    else:
        NCOL=3

    seeds_num = 10
    seeds = np.arange(seeds_num)






    func_names = ['const_SynFun_plus_corr_ell=_0.1-d=3-seed0', 'const_SynFun_plus_corr_ell=_0.1-d=3-seed1', 'const_SynFun_plus_corr_ell=_0.1-d=3-seed2', 'const_SynFun_plus_corr_ell=_0.1-d=3-seed3', 'const_SynFun_plus_corr_ell=_0.1-d=3-seed4']


    if Parallel and 'plus_corr' in func_names[0]:
        return None

    if Parallel:
        BO_methods = ['PECI_'+STR_NUM_WORKER, 'PTSC_'+STR_NUM_WORKER, 'PMESC_'+STR_NUM_WORKER, 'PMESC_LB_'+STR_NUM_WORKER]
    else:
        BO_methods = ['ECI', 'TSC', 'MESC', 'MESC_LB', 'MESC_LB_corr', 'TSC_corr']

    if 'plus_corr' in func_names[0]:
        plot_max = 6
        plot_min = 1e-4
        NCOL=2
    else:
        plot_max = 10
        plot_min = 5*1e-1
        if not Parallel:
            BO_methods.append('PESC')


    if 'SynFun' in func_names[0]:
        test_func = eval('test_functions.const_SynFun')()
        if 'd=2' in func_names[0]:
            input_dim = 2
        elif 'd=3' in func_names[0]:
            input_dim = 3

        elif 'd=4' in func_names[0]:
            input_dim = 4
        elif 'd=5' in func_names[0]:
            input_dim = 5


        COST_INI = 5 * input_dim
        COST_MAX = 200
    plot_cost = np.arange(COST_MAX)




    for method in BO_methods:
        SimReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf
        for i, func_name in enumerate(func_names):
            GLOBAL_MAX = None


            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed0':
                GLOBAL_MAX = 2.717605081261938
                GLOBAL_MIN = -3.9583288574696502
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed1':
                GLOBAL_MAX = 3.26958904953565
                GLOBAL_MIN = -3.0248848256595653
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed2':
                GLOBAL_MAX = 3.3108119662204625
                GLOBAL_MIN = -3.254167200005991
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed3':
                GLOBAL_MAX = 3.473180558250239
                GLOBAL_MIN = -3.167877206106251
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed4':
                GLOBAL_MAX = 3.3820964713145703
                GLOBAL_MIN = -3.3536334011994815


            if re.match(r'const_SynFun_ell=(_\d+\.\d+){3}-d=\d-seed\d', func_name):
                test_func.C = 2

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){4}-d=\d-seed\d', func_name):
                test_func.C = 3

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){5}-d=\d-seed\d', func_name):
                test_func.C = 4

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){6}-d=\d-seed\d', func_name):
                test_func.C = 5

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){11}-d=\d-seed\d', func_name):
                test_func.C = 10

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){16}-d=\d-seed\d', func_name):
                test_func.C = 15

            if 'corr' in func_name:
                test_func.C = 4
            if 'plus' in func_name or 'minus' in func_name:
                test_func.C = 3

            result_path = func_name + '_results/'

            plot=True
            for seed in seeds:
                temp_path = result_path + method + '/seed=' + str(seed) + '/'
                if os.path.exists(temp_path + 'cost.pickle') and os.path.exists(temp_path + 'SimReg.pickle'):
                    if os.path.getsize(temp_path + 'cost.pickle')>0 and os.path.getsize(temp_path + 'SimReg.pickle')>0:
                        with open(temp_path + 'cost.pickle', 'rb') as f:
                            cost = pickle.load(f)


                            cost = cost / (test_func.C + 1)
                        with open(temp_path + 'SimReg.pickle', 'rb') as f:
                            SimReg = pickle.load(f)
                            SimReg[SimReg==None] = GLOBAL_MIN

                        for j in range(np.size(cost)):
                            if j+1 <= np.size(SimReg):
                                if j+1 == np.size(cost):
                                    break
                                else:
                                    SimReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = SimReg[j]
                else:
                    plot=False


            SimReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - SimReg_all[i*seeds_num:(i+1)*seeds_num, :]
        if plot:
            index = np.logical_not(np.any(np.isinf(SimReg_all), axis=0))
            method = method.replace('ECI', 'EIC')
            method = method.replace('_Q='+str(q), '')
            method = method.replace('_', '-')

            if average:
                SimReg_ave = np.mean(SimReg_all, axis=0)
                SimReg_se = np.sqrt(np.sum((SimReg_all - SimReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))


                if each_plot:
                    if 'MESC-LB' in method:
                        for j in range(len(func_names)):
                            SimReg_ave = np.mean(SimReg_all[j:(j+1)*10, :], axis=0)
                            SimReg_se = np.sqrt(np.sum((SimReg_all[j:(j+1)*10, :] - SimReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                            plt.errorbar(plot_cost[index], SimReg_ave[index], yerr=SimReg_se[index], errorevery=5, capsize=3, elinewidth=1, label=method+'-'+str(j))
                else:
                    plt.errorbar(plot_cost[index], SimReg_ave[index], yerr=SimReg_se[index], errorevery=5, capsize=3, elinewidth=1, label=method)



            else:
                SimReg_median = np.median(SimReg_all, axis=0)
                SimReg_1_4 = np.quantile(SimReg_all, 1/4., axis=0)
                SimReg_3_4 = np.quantile(SimReg_all, 3/4., axis=0)



                error_bar_plot = np.r_[np.atleast_2d((SimReg_median - SimReg_1_4)[index]), np.atleast_2d((SimReg_3_4 - SimReg_median)[index])]
                plt.errorbar(plot_cost[index], SimReg_median[index], yerr=error_bar_plot, errorevery=5, capsize=3, elinewidth=1, label=method)
    func_name = func_name.replace('const_', '')
    plt.title('Synthetic Function' + ' (d='+str(input_dim)+')' + ' (C='+str(test_func.C)+')')

    plt.xlim(COST_INI, COST_MAX)

    plt.yscale('log')
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.xlabel('Total number of evaluations')
    plt.ylabel('Utility Gap')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.close()












    for method in BO_methods:
        InfReg_all = np.ones((seeds_num*len(func_names), COST_MAX)) * np.inf
        for i, func_name in enumerate(func_names):
            GLOBAL_MAX = None


            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed0':
                GLOBAL_MAX = 2.717605081261938
                GLOBAL_MIN = -3.9583288574696502
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed1':
                GLOBAL_MAX = 3.26958904953565
                GLOBAL_MIN = -3.0248848256595653
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed2':
                GLOBAL_MAX = 3.3108119662204625
                GLOBAL_MIN = -3.254167200005991
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed3':
                GLOBAL_MAX = 3.473180558250239
                GLOBAL_MIN = -3.167877206106251
            if func_name=='const_SynFun_plus_corr_ell=_0.1-d=3-seed4':
                GLOBAL_MAX = 3.3820964713145703
                GLOBAL_MIN = -3.3536334011994815


            if re.match(r'const_SynFun_ell=(_\d+\.\d+){3}-d=\d-seed\d', func_name):
                test_func.C = 2


            if re.match(r'const_SynFun_ell=(_\d+\.\d+){4}-d=\d-seed\d', func_name):
                test_func.C = 3

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){5}-d=\d-seed\d', func_name):
                test_func.C = 4

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){6}-d=\d-seed\d', func_name):
                test_func.C = 5

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){11}-d=\d-seed\d', func_name):
                test_func.C = 10

            if re.match(r'const_SynFun_ell=(_\d+\.\d+){16}-d=\d-seed\d', func_name):
                test_func.C = 15

            if 'corr' in func_name:
                test_func.C = 4

            if 'plus' in func_name or 'minus' in func_name:
                test_func.C = 3

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
                                        break
                                    else:
                                        InfReg_all[seed+i*seeds_num, int(cost[j]) : int(cost[j+1])] = InfReg[j]
                    else:
                        plot=False

                InfReg_all[i*seeds_num:(i+1)*seeds_num, :] = GLOBAL_MAX - InfReg_all[i*seeds_num:(i+1)*seeds_num, :]
            else:
                pesc_path = 'PESC_results/const_SynFun'+str(i)+'.csv'
                if os.path.exists(pesc_path):
                    plot=True
                    if np.all(np.isinf(InfReg_all)):
                        InfReg_all = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_all = np.c_[np.inf*np.ones((np.shape(InfReg_all)[0], COST_INI)), InfReg_all]
                        if np.size(plot_cost) < np.shape(InfReg_all)[1]:
                            InfReg_all = InfReg_all[:,:np.size(plot_cost)]
                    else:
                        InfReg_tmp = np.loadtxt(pesc_path, delimiter=',')
                        InfReg_tmp = np.c_[np.inf*np.ones((np.shape(InfReg_tmp)[0], COST_INI)), InfReg_tmp]
                        if np.size(plot_cost) < np.shape(InfReg_tmp)[1]:
                            InfReg_tmp = InfReg_tmp[:,:np.size(plot_cost)]
                        InfReg_all = np.r_[InfReg_all, InfReg_tmp]

                    if np.size(plot_cost) > np.shape(InfReg_all)[1]:
                        plot_cost = plot_cost[:np.shape(InfReg_all)[1]]


        if plot:
            temp_index = BO_methods.index(method)
            index = np.logical_not(np.any(np.isinf(InfReg_all), axis=0))
            method = method.replace('ECI', 'EIC')
            method = method.replace('_Q='+str(q), '')
            method = method.replace('_', '-')


            method = method.replace('MESC', 'CMES')
            method = method.replace('-LB', '-IBO')
            if Parallel:
                method = method.replace('P', 'P-')

            if '-corr' in method:
                method = 'C-' + method
                method = method.replace('-corr', '')

            if average:
                InfReg_ave = np.mean(InfReg_all, axis=0)
                InfReg_se = np.sqrt(np.sum((InfReg_all - InfReg_ave)**2, axis=0) / (seeds_num*len(func_names) - 1)) / np.sqrt(seeds_num*len(func_names))



                if each_plot:
                    if 'MESC-LB' in method:
                        for j in range(len(func_names)):
                            InfReg_ave = np.mean(InfReg_all[j:(j+1)*10, :], axis=0)
                            InfReg_se = np.sqrt(np.sum((InfReg_all[j:(j+1)*10, :] - InfReg_ave)**2, axis=0) / (seeds_num - 1)) / np.sqrt(seeds_num)
                            plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=5, capsize=3, elinewidth=1, label=method+'-'+str(j))
                else:

                    plt.errorbar(plot_cost[index], InfReg_ave[index], yerr=InfReg_se[index], errorevery=(temp_index*5, 25), elinewidth=3., label=method)

            else:
                InfReg_median = np.median(InfReg_all, axis=0)
                InfReg_1_4 = np.quantile(InfReg_all, 1/4., axis=0)
                InfReg_3_4 = np.quantile(InfReg_all, 3/4., axis=0)

                error_bar_plot = np.r_[np.atleast_2d((InfReg_median- InfReg_1_4)[index]), np.atleast_2d((InfReg_3_4 - InfReg_median)[index])]
                plt.errorbar(plot_cost[index], InfReg_median[index], yerr=error_bar_plot, errorevery=5, capsize=3, elinewidth=1, label=method)
    func_name = func_name.replace('const_', '')
    plt.title('Synthetic Function' + ' (d='+str(input_dim)+')' + ' (C='+str(test_func.C)+')')

    plt.xlim(COST_INI, COST_MAX)
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


if __name__ == '__main__':



    plot_synthetic(Parallel=False, average=True)





