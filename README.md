# Python Code for ***Constrained Max-value Entropy Search via Information lower BOund (CMES-IBO)***
This page provides a python implementation of CMES-IBO (Takeno et. al., ICML2022). 
The code can reproduce the results of (Takeno et. al., ICML2022).


# Environment
* Linux
    * We ran the experiments on CentOS 6.9.
    * We confirmed the scripts running on Ubuntu 20.04 (latest LTS), but for this environment, we have not confirmed that the result of the paper can be completely reproduced (the difference in OS may produce a slight change in the results).
* Python3.6.5 ([Anaconda3-5.2.0](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh))
* Additional packages are as follows. (All packages are in requirements.txt)
    * GPy==1.9.6
    * Scipy=1.4.1
    * nlopt=2.6.1
* Fortran (gfortran)
* C (gcc 8.3.1)
* Our current code is purely for the empirical evaluation of our paper. We are planning to add comments and organize the code when we distribute the code as a python package.

# Instruction

* Methods
    * method names
        * MESC-LB (Corresponding to CMES-IBO)
        * MESC (Corresponding to CMES)
        * EIC
        * TSC
    * Parallel method names
        * PMESC-LB (Corresponding to P-CMES-IBO)
        * PMESC (Corresponding to P-CMES)
        * PEIC (Corresponding to P-EIC)
        * PTSC (Corresponding to P-TSC)

* For the benchmark and real-world function experiments (Benchmark_name: Gardner1, Gardner2, Gramacy, const_HartMann6, G1, G7, G10, ReactorNetworkDesign, const_cnn_cifar10)
    * For 10 parallel experiments of different seeds (0 ~ 9):
        * python cons_bayesopt_exp.py method_name Benchmark_name 0 -1 1
        * python cons_bayesopt_exp.py Parallel_method_name Benchmark_name 0 -1 3
    * If you run an experiment with one specific seed:
        * python bayesopt_exp.py method_name Benchmark_name 0 seed 1
        * python parallel_bayesopt_exp.py Parallel_method_name Benchmark_name 0 seed 3
    * For the plots of the experimental results:
        * python plot_results_toy.py

* For the synthetic function experiments
    * For 10 times 10 parallel experiments (10 generated functions and 10 initial samplings) of different seeds (0 ~ 9 and 0 ~ 9):
        * python cons_bayesopt_exp.py method_name const_SynFun -1 -1 1
    * For `single constraint' counterpart:
        * python cons_bayesopt_exp.py method_name const_SynFun_compress -1 -1 1
    * If you run the experiment with one specific function-generation seed, and initialization seed:
        * python cons_bayesopt_exp.py method_name const_SynFun function_seed initial_seed 1
    * For the plots of the experimental results:
        * python plot_results_synthetic.py

* For the correlated synthetic function experiments
    * For 5 times 10 parallel experiments (5 generated functions and 10 initial samplings) of different seeds (0 ~ 4 and 0 ~ 9):
        * python cons_bayesopt_exp.py method_name const_SynFun_plus_corr -1 -1 1
    * If you run the experiment with one specific function-generation seed, and initialization seed:
        * python cons_bayesopt_exp.py method_name const_SynFun_plus_corr function_seed initial_seed 1
    * For the plots of the experimental results:
        * python plot_results_synthetic_correlated.py
