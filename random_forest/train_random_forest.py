import numpy as np
import xgboost as xgb
import time
import json

available_datasets = {
    "breast_cancer": {
        "train_path": "raw_data/breast_cancer_scale0.train",
        "test_path": "raw_data/breast_cancer_scale0.test",
        "save_path": "random_forest/models/breast_cancer_rf.json",
        "param": {
            "objective": "binary:logistic",
            "eta": 1,
            "gamma": 1.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 4,
            "max_depth": 6,
            'tree_method': 'gpu_hist',
        },
        "is_binary": True,
    },
    "covtype": {
        "test_path": "raw_data/covtype.scale01.test0",
        "train_path": "raw_data/covtype.scale01.train0",
        "save_path": "random_forest/models/covtype_rf.json",
        "param": {
            "objective": "multi:softmax",
            "num_class": 7,
            "eta": 1,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 80,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": False,
    },
    "ori_mnist": {
        "test_path": "raw_data/ori_mnist.test0",
        "train_path": "raw_data/ori_mnist.train0",
        "save_path": "random_forest/models/ori_mnist_rf.json",
        "param": {
            "objective": "multi:softmax",
            "num_class": 10,
            "eta": 1,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 200,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": False,
    },
    "webspam": {
        "test_path": "raw_data/webspam_wc_normalized_unigram.svm0.test",
        "train_path": "raw_data/webspam_wc_normalized_unigram.svm0.train",
        "save_path": "random_forest/models/webspam_rf.json",
        "param": {
            "objective": "binary:logistic",
            "eta": 1,
            "gamma": 1.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 100,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": True,
    },
    # New
    "binary_mnist": {
        "test_path": "raw_data/binary_mnist0.t",
        "train_path": "raw_data/binary_mnist0",
        "save_path": "random_forest/models/binary_mnist_rf.json",
        "param": {
            "objective": "binary:logistic",
            "eta": 1,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 1000,
            "max_depth": 4,
            'tree_method': 'gpu_hist',
        },
        "is_binary": True,
    },
    "diabetes": {
        "test_path": "raw_data/diabetes_scale0.test",
        "train_path": "raw_data/diabetes_scale0.train",
        "save_path": "random_forest/models/diabetes_rf.json",
        "param": {
            "objective": "binary:logistic",
            "eta": 1,
            "gamma": 1.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 25,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": True,
    },
    "fashion_mnist": {
        "test_path": "raw_data/fashion.test0",
        "train_path": "raw_data/fashion.train0",
        "save_path": "random_forest/models/fashion_mnist_rf.json",
        "param": {
            "objective": "multi:softmax",
            "num_class": 10,
            "eta": 1,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 200,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": False,
    },
    "higgs": {
        "test_path": "raw_data/HIGGS_s.test0",
        "train_path": "raw_data/HIGGS_s.train0",
        "save_path": "random_forest/models/higgs_rf.json",
        "param": {
            "objective": "binary:logistic",
            "eta": 1,
            "gamma": 1.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 300,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": True,
    },
    "ijcnn": {
        "test_path": "raw_data/ijcnn1s0.t",
        "train_path": "raw_data/ijcnn1s0",
        "save_path": "random_forest/models/ijcnn_rf.json",
        "param": {
            "objective": "binary:logistic",
            "eta": 1,
            "gamma": 1.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 100,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": True,
    },
    "bosch": {
        "test_path": "raw_data/bosch.test",
        "train_path": "raw_data/bosch.train",
        "save_path": "random_forest/models/bosch_rf.json",
        "param": {
            "objective": "binary:logistic",
            "eta": 1,
            "gamma": 1.0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 400,
            "max_depth": 8,
            'tree_method': 'gpu_hist',
        },
        "is_binary": True,
    },
}


def calculate_accuracy(bst, test_path, is_binary):
    dtest = xgb.DMatrix(test_path, silent=True)
    y = dtest.get_label()
    num_total = y.shape[0]
    num_correct = 0

    y_pred = bst.predict(dtest)
    if is_binary:
        y_pred[y_pred > 0.5] = 1
        y_pred = y_pred.astype(int)
    num_correct = np.sum(y == y_pred)
    return (num_correct / num_total, num_total)


# The random forest implementation puts all trees of class 0 in the first |num_parallel_tree|.
def reorder_trees(json_path, num_class, num_parallel_tree):
    if num_class == 2:
        return
    with open(json_path, "r") as f:
        rf_json = json.load(f)
    assert num_class * num_parallel_tree == len(rf_json)
    tree_per_class = [[] for i in range(num_class)]
    for i, t in enumerate(rf_json):
        tree_per_class[(i // num_parallel_tree) % num_class].append(t)
    num_tree_per_class = len(tree_per_class[0])
    new_json = []
    for j in range(num_tree_per_class):
        for i in range(num_class):
            new_json.append(tree_per_class[i][j])
    with open(json_path, 'w') as outfile:
        json.dump(new_json, outfile, indent=2)


def train_rf(dataset_name):
    ds = available_datasets[dataset_name]
    train_path, save_path = ds['train_path'], ds['save_path']
    dtrain = xgb.DMatrix(train_path, silent=True)
    bst = xgb.train(ds['param'], dtrain, num_boost_round=1)
    model_path = save_path.replace('.json', '.model')
    bst.save_model(model_path)
    bst.dump_model(save_path, dump_format='json')
    if 'num_class' in ds['param']:
        reorder_trees(save_path, ds['param']['num_class'],
                      ds['param']['num_parallel_tree'])

    accuracy, num_total = calculate_accuracy(bst, ds['test_path'],
                                             ds['is_binary'])
    print('  dataset_name:%s, accuracy:%.4f, num_total:%d' %
          (dataset_name, accuracy, num_total))


if __name__ == "__main__":
    for dataset_name in available_datasets:
        timestart = time.time()
        train_rf(dataset_name)
        timeend = time.time()
        print('Finished %s in %.4f s' % (dataset_name, timeend - timestart))
