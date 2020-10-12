# LT-Attack Setup
## Ubuntu 20.04
```
cd lt-attack

# Requires libboost>=1.66 for |thread_pool|.
sudo apt install libboost-all-dev

# We use the pre-trained GBDT models from https://github.com/chenhongge/treeVerification.
# [NeurIPS 2019] H. Chen*, H. Zhang*, S. Si, Y. Li, D. Boning and C.-J. Hsieh, Robustness Verification of Tree-based Models
# The author published their pre-trained models in the URL below.
wget http://download.huan-zhang.com/models/tree-verify/tree_verification_models.tar.bz2
tar jxvf tree_verification_models.tar.bz2

# Attack the standard (natural) GBDT model for the breast_cancer dataset. Construct
# adversarial examples on L-2 norm perturbation, using 20 threads on 500 test examples.
make
./lt_attack configs/breast_cancer_unrobust_20x500_norm2_lt-attack.json

# Attack the standard (natural) RF model for the breast_cancer dataset. Construct
# adversarial examples on L-2 norm perturbation, using 20 threads on 100 test examples.
./lt_attack configs/breast_cancer_unrobust-rf_20x100_norm2_lt-attack.json
```

## Sample Output
```
//...
===== Attack result for example 500/500 Norm(2)=0.235702 =====
All Best Norms: Norm(-1)=0.166667 Norm(1)=0.333579 Norm(2)=0.235702.
Average Norms: Norm(-1)=0.235932 Norm(1)=0.369484 Norm(2)=0.282763.
Best Points for example at line 500
1 1:0.07214075340 2:0.11111100000 4:0.16666650810 6:0.11111100000 7:0.16666650810
Results for config:configs/breast_cancer_unrobust_20x500_norm2_lt-attack.json
Average Norms: Norm(-1)=0.235932 Norm(1)=0.369484 Norm(2)=0.282763
--- Timing Metrics ---
|collect_histogram| disabled
## Actual Examples Tested:496
## Time per point: 0.00141016
```

# Configuration File Parameters
We provide sample config files in `config/` which use the following parameters:

- `search_mode`: The attack method to use. Choose from `'lt-attack'` (ours), `'naive-leaf'`, `'naive-feature'`.
- `norm_type`: The objective norm order. Supports 1, 2, and -1 (for L-Inf).
- `num_point`: Number of test examples to attack. We use 500 test examples in most of our experiments.
- `num_threads`: CPU threads per task. We use 20 physical threads per task in most of our experiments.
- `num_attack_per_point`: Number of initial adversarial examples. Usually set to the same as `num_threads`.
- `enable_early_return`: Use early return to speed up the search in `Neighbor_1(C')`. Usually set to `true`.

Additional dataset related parameters:
- `model`: Path to the JSON file dumped from XGBoost models using `bst.dump_model('bar.json', dump_format='json')`. See https://xgboost.readthedocs.io/en/latest/python/python_intro.html#training
- `inputs`: Path to the test example file in LIBSVM format.
- `num_classes`: Number of classes in the dataset.
- `num_features`: Number of features in the dataset.
- `feature_start`: The index of the first feature, could be 0 or 1 on different datasets.

# Baselines
## SignOPT, HSJA, and Cube
```
pip3 install xgboost==1.0.2 sklearn
# Choose |'search_mode'| from 'signopt', 'hsja', and 'cube'. We provide a few sample configs:
python3 baselines/test_attack_cpu.py --config_path=configs/breast_cancer_unrobust_20x500_norm2_cube.json
```

## MILP
```
# Use |'search_mode': 'milp'|. Requires the Gurobi Solver installed.
python3 baselines/xgbKantchelianAttack.py --config_path=configs/breast_cancer_unrobust_20x500_norm2_milp.json
```

## RBA-Appr
```
# RBA-Appr requires training data which can be downloaded from https://github.com/chenhongge/RobustTrees.
# [ICML 2019] Hongge Chen, Huan Zhang, Duane Boning, and Cho-Jui Hsieh, Robust Decision Trees Against Adversarial Examples
# The author published their datasets in the URL below.
mkdir raw_data
cd raw_data
wget https://raw.githubusercontent.com/chenhongge/RobustTrees/master/data/download_data.sh
sh download_data.sh
cd ..

# Use |'search_mode': 'region'|, and add |"train_data": "raw_data/TRAIN_DATA_NAME"| to the corresponding config file.
# We provide a sample config:
./lt_attack configs/breast_cancer_unrobust_20x500_norm2_region.json
```


# Known Issues
The JSON dump of XGBoost models offer precision up to 8 digits, however the difference between certain feature
split threholds may be smaller than 1e-8 in the original XGBoost model. For this reason the model created from
the JSON dump may produce a different prediction on certain examples than the original XGBoost model, and we
manually verify that each produced adversarial example is valid under the JSON dump.


# Credits
1. `nlohmann/json*`: https://github.com/nlohmann/json.
2. `.clang-format`: https://cs.chromium.org/chromium/src/.clang-format.
3. See paper for the full list of references.
