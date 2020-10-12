############################################################
### Forked from https://github.com/cmhcbb/attackbox
############################################################

import numpy as np
import xgboost as xgb
import sklearn.datasets
import json
import time

class CPUModel(object):
    def __init__(self, model):
        self.model = model
        self.bounds = [0, 1]
        self.num_queries = 0

    def predict_label(self, image_or_images):
        if len(image_or_images.shape) == 1:
            images = np.array([image_or_images])
        else:
            images = image_or_images
        output = self.model.predict_label(images)
        self.num_queries += images.shape[0]
        if len(image_or_images.shape) == 1:
            return output[0]
        return output

    def get_num_queries(self):
        return self.num_queries


class XGBoostModel(object):
    def __init__(self, config_path, num_threads):
        with open(config_path) as json_file:
            config = json.load(json_file)

        model_path = config['model'].replace('.json', '.model')
        self.config = config
        self.model = xgb.Booster({'nthread': num_threads})
        self.model.load_model(model_path)
        self.is_binary = config['num_classes'] == 2
        self.num_classes = config['num_classes']
        self.num_queries = 0
        self.xgb_runtime = 0.0

    def predict_label(self, images):
        assert len(images.shape) == 2
        dtest = xgb.DMatrix(np.array(images), silent=True)
        y_pred = self.prefict_with_timer(dtest)
        if self.is_binary:
            y_pred[y_pred > 0.5] = 1
            y_pred = y_pred.astype(int)
        self.num_queries += images.shape[0]
        return y_pred

    def margin(self, X):
        dtest = xgb.DMatrix(X, silent=True)
        return self.prefict_with_timer(dtest, output_margin=True)

    def prefict_with_timer(self, dtest, **kargs):
        timestart = time.time()
        y_pred = self.model.predict(dtest, **kargs)
        timeend = time.time()
        self.xgb_runtime += timeend - timestart
        return y_pred


    def get_num_queries(self):
        return self.num_queries

class XGBoostTestLoader(object):
    def __init__(self, config_path):
        with open(config_path) as json_file:
            config = json.load(json_file)
        data_path = config['inputs']
        # Always use zero_based so we don't have to insert a zero column
        # manually at the beginning.
        X, y = sklearn.datasets.load_svmlight_file(data_path, zero_based=True)
        X = X.toarray()
        self.offset = config.get('offset', 0)
        self.len = min(X.shape[0] - self.offset, config['num_point'])
        self.X = X[self.offset : self.offset + self.len, :]
        self.y = y[self.offset : self.offset + self.len].astype(int)

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        return (self.X[key], self.y[key])
