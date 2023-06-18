import sys
import json
import pickle
import argparse
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(cfg='', setting_idx=0, m=2*10*1000, num_runs=30):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        file = '../' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Online performance
        per_param_setting_performance.append(np.array(bin_m_errs(errs=data['errs'], m=m)))

    print(param_settings[setting_idx], setting_idx)
    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # change the cfg file to get the results for different activation functions, ex. '../cfg/sgd/bp/tanh.json'
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str,
                            default='../cfg/sgd/bp/relu.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    performances = []
    m = int(params['flip_after'])*2

    _, param_settings = get_configurations(params=params)
    labels = param_settings
    num_runs = params['num_runs']

    # --------------------------------------------------------------------------
    labels = []
    
    num_runs = 10

    # Add line for bp learner with relu activation
    performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/bp/relu_test.json', setting_idx=0, m=m, num_runs=num_runs))
    labels.append('bp relu')

    # Add line for bp learner with linear activation
    performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/bp/linear_test.json', setting_idx=0, m=m, num_runs=num_runs))
    labels.append('bp linear')

    # Add line for cbp learner with relu activation
    performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/cbp/relu_test.json', setting_idx=0, m=m, num_runs=num_runs))
    labels.append('cbp relu')

    # Add line for full natural importance learner with relu activation
    #performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/natimp/relu_test.json', setting_idx=0, m=m, num_runs=num_runs))
    #labels.append('natimp relu (1.0)')

    # Add line for 0.5 natural importance learner with relu activation
    #performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/sparse_natimp/relu_test_0.5.json', setting_idx=0, m=m, num_runs=num_runs))
    #labels.append('natimp relu (0.5)')

    # Add line for 0.1 natural importance learner with relu activation
    #performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/sparse_natimp/relu_test_0.1.json', setting_idx=0, m=m, num_runs=num_runs))
    #labels.append('natimp relu (0.1)')

    # Add line for random cbp (i.e. 0.0 natural importance) learner with relu activation
    #performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/cbp/relu_test_random.json', setting_idx=0, m=m, num_runs=num_runs))
    #labels.append('natimp relu (0.0)')

    # Add line for full taylor cbp learner with relu activation
    #performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/natimp_taylor/relu_test.json', setting_idx=0, m=m, num_runs=num_runs))
    #labels.append('natimp taylor relu')

    # Add line for mean update with relu activation
    performances.append(add_cfg_performance(cfg='../cfg/' + params['opt'] + '/mean_update/relu_test.json', setting_idx=0, m=m, num_runs=num_runs))
    labels.append('mean_update relu')

    # --------------------------------------------------------------------------

    performances = np.array(performances)

    if params['hidden_activation'] in ['relu', 'swish', 'leaky_relu']:
        yticks = [0.4, 0.8, 1.2, 1.6, 2.0] # (natimp vs cbp)
        #yticks = [0.4, 0.52, 0.64, 0.76, 0.88, 1.0] # (natimp vs sparse natimp)
        #yticks = [0.4, 0.52, 0.64, 0.76, 0.88, 1.0] # (natimp vs taylor natimp)
    else:
        yticks = [0.4, 0.8, 1.2, 1.6, 2.0] # (natimp vs cbp)
        #yticks = [0.4, 0.52, 0.64, 0.76, 0.88, 1.0] # (natimp vs sparse natimp)
        #yticks = [0.4, 0.52, 0.64, 0.76, 0.88, 1.0] # (natimp vs taylor natimp)

    print(yticks, params['hidden_activation'])
    generate_online_performance_plot(
        performances=performances,
        #colors=['C3', 'C4', 'C5', 'C8'], # (natimp vs cbp)
        #colors=['C8', 'C6', 'C7', 'C9'], # (natimp vs sparse natimp)
        #colors=['C5', 'C8', 'C7', 'C10'], # (natimp vs taylor natimp)
        colors=['C3', 'C4', 'C5', 'C11'],
        yticks=yticks,
        xticks=[0, 500000, 1000000],
        xticks_labels=['0', '0.5M', '1M'],
        m=m,
        labels=labels
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

'''

# This repo was forked from https://github.com/shibhansh/loss-of-plasticity; 
# see setup instructions there, and at 
# https://github.com/shibhansh/loss-of-plasticity/tree/main/lop/slowly_changing_regression

# Initial steps
cd lop/slowly_changing_regression
mkdir env_temp_cfg temp_cfg
python3.8 multi_param_expr.py -c cfg/prob.json

# Generates enough data for up to 100 runs. Warning: this takes 84 GB of space.
# Going up to 5-10 runs should be enough for a decent plots.
for i in {0..9}; do python3.8 slowly_changing_regression.py -c env_temp_cfg/${i}.json; done

# Edit cfg files as you wish, then run the following to generate the plots. We
# provide the commands to generate data when num_runs=10 in each of the relevant
# cfg files. Note that the first command in each of the paired commands below
# generates data in a folder called temp_cfg. The second command uses this temp
# data to run the experiment.

# Generate bp data:
python3.8 multi_param_expr.py -c cfg/sgd/bp/relu_test.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Generate cbp data:
python3.8 multi_param_expr.py -c cfg/sgd/cbp/relu_test.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Generate linear data:
python3.8 multi_param_expr.py -c cfg/sgd/bp/linear_test.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Generate full natimp data:
python3.8 multi_param_expr.py -c cfg/sgd/natimp/relu_test.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Generate sparse natimp (0.5) data:
python3.8 multi_param_expr.py -c cfg/sgd/sparse_natimp/relu_test_0.5.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Generate sparse natimp (0.1) data:
python3.8 multi_param_expr.py -c cfg/sgd/sparse_natimp/relu_test_0.1.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Generate sparse natimp (0.0), i.e. random, data:
python3.8 multi_param_expr.py -c cfg/sgd/cbp/relu_test_random.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Full Taylor natimp
python3.8 multi_param_expr.py -c cfg/sgd/natimp_taylor/relu_test.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# Mean_update
python3.8 multi_param_expr.py -c cfg/sgd/mean_update/relu_test.json
for i in {0..9}; do echo "i: $i"; python3.8 expr.py -c temp_cfg/${i}.json; done

# After all data has been generated, to plot, edit above to decide what lines
# you want to include, then run the following:
cd plots
python3.8 online_performance.py

# Result is saved to comparison.png. Rename this file before generating another
# plot again, else it will be overwritten.

'''