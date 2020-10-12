############################################################
### Forked from https://github.com/cmhcbb/attackbox
############################################################

from Sign_OPT_cpu import OPT_attack_sign_SGD_cpu
from HSJA import HSJA
from OPT_attack_lf import OPT_attack_lf
from cube_attack import Cube
from models_cpu import CPUModel, XGBoostModel, XGBoostTestLoader
import os, argparse
import time
import random
from numpy import linalg as LA
import numpy as np
import json
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='path to the config file')

attack_list = {
    "opt": OPT_attack_lf,
    "signopt": OPT_attack_sign_SGD_cpu,
    "hsja": HSJA,
    "cube": Cube,
}

args = parser.parse_args()

with open(args.config_path) as json_file:
    config = json.load(json_file)

print('Using config:', config)

num_attack = int(config['num_attack_per_point'])
# Cube Attack has built in support.
if config['search_mode'] == 'cube':
    num_attack = 1

print('Using num_attack:', num_attack)

test_loader = XGBoostTestLoader(args.config_path)
norm_order = config['norm_type']
# -1 was used as Inf for other benchmarks
if norm_order == -1:
    norm_order = np.inf

model = XGBoostModel(args.config_path, config['num_threads'])
amodel = CPUModel(model)
attack = attack_list[config['search_mode']](amodel, norm_order)


def run_attack(xi, yi, idx):
    best_norm = np.inf
    best_adv = None
    for i in range(num_attack):
        # Fix random seed for each attack.
        random.seed(8 + i)
        np.random.seed(8 + i)

        succ, adv = attack(xi, yi)
        if not succ:
            print('!!!Failed on example %d attack %d' % (idx, i + 1))
            continue
        current_norm = LA.norm(adv - xi, norm_order)
        print('Example %d attack %d: Norm=%.4f' % (idx, i + 1, current_norm))
        if current_norm < best_norm:
            best_norm = current_norm
            best_adv = adv

    succ = best_adv is not None
    return succ, best_adv


total_Linf = 0.0
total_L1 = 0.0
total_L2 = 0.0
total_success = 0

num_examples = len(test_loader)
print('Attacking %d examples...' % num_examples)
timestart = time.time()

for i, (xi, yi) in enumerate(test_loader):
    print(f"image batch: {i}")
    if (amodel.predict_label(xi) != yi):
        print(f"Fail to classify example {i+1}. No need to attack.")
        continue

    #adv=attack(xi,yi, 0.2)
    single_timestart = time.time()
    succ, adv = run_attack(xi, yi, i + 1)
    single_timeend = time.time()
    if succ:
        adv_check = (amodel.predict_label(adv) != yi)
        assert adv_check, '!!!Attack report success but adv invalid!!!'
        print(
            '\n===== Attack result for example %d/%d Norm(%d)=%lf time=%.4fs ====='
            %
            (i + 1, num_examples, config['norm_type'],
             LA.norm(adv - xi, norm_order), single_timeend - single_timestart))
        total_Linf += LA.norm(adv - xi, np.inf)
        total_L1 += LA.norm(adv - xi, 1)
        total_L2 += LA.norm(adv - xi, 2)
        total_success += 1
    else:
        print('!!!Failed on example %d' % (i + 1))

timeend = time.time()

print('*******************************')
print('*******************************')
print('Results for config:', args.config_path)
print('Num Attacked: %d Actual Examples Tested:%d' %
      (len(test_loader), total_success))

# Hack to avoid div-by-0.
if total_success == 0:
    total_success = 1

print('Norm(-1)=%.4f' % (total_Linf / total_success))
print('Norm(1)=%.4f' % (total_L1 / total_success))
print('Norm(2)=%.4f' % (total_L2 / total_success))
print('Time per point: %.4f s' % ((timeend - timestart) / total_success))
print('XGB Time per point: %.4f s' % (model.xgb_runtime / total_success))
print('XGB Time ratio: %.4f' % (model.xgb_runtime / (timeend - timestart)))
