import numpy as np
import pandas as pd
try:
    import cudf
except ImportError as e:
    print("Failed to import cuDF: " + str(e))
    print("Skipping this test")
from sklearn import datasets
import sys
import unittest
import xgboost as xgb

from regression_test_utilities import run_suite, parameter_combinations, \
    assert_results_non_increasing, Dataset


def get_gdf(is_y_df=True):
    rng = np.random.RandomState(199)
    n = 50000
    m = 20
    sparsity = 0.25
    X, y = datasets.make_regression(n, m, random_state=rng)
    Xy = (np.ascontiguousarray
    (np.transpose(np.concatenate((X, np.expand_dims(y, axis=1)), axis=1))))
    df = cudf.DataFrame(list(zip(['col%d' % i for i in range(m+1)], Xy)))
    all_columns = list(df.columns)
    cols_X = all_columns[0:len(all_columns)-1]
    
    if is_y_df:
        col_y = [all_columns[len(all_columns)-1]]
    else:
        col_y = all_columns[len(all_columns)-1]
        
    return df[cols_X], df[col_y]


class TestGPU(unittest.TestCase):
    
    gdf_datasets_y_df = [Dataset("GDF_DF", lambda: get_gdf(is_y_df=True), "reg:linear", "rmse")]
    gdf_datasets_y_ser = [Dataset("GDF_SER", lambda: get_gdf(is_y_df=False), "reg:linear", "rmse")]
    
    def test_gdf_y_df(self):
        variable_param = {'n_gpus': [1], 'max_depth': [10], 'max_leaves': [255],
                          'max_bin': [255],
                          'grow_policy': ['lossguide']}
        
        for param in parameter_combinations(variable_param):
            param['tree_method'] = 'gpu_hist'
            gpu_results = run_suite(param, num_rounds = 20,
                                    select_datasets = self.gdf_datasets_y_df)
            assert_results_non_increasing(gpu_results, 1e-2)
    
#     def test_gdf_y_ser(self):
        
#         variable_param = {'n_gpus': [1], 'max_depth': [10], 'max_leaves': [255],
#                           'max_bin': [255],
#                           'grow_policy': ['lossguide']}
        
#         for param in parameter_combinations(variable_param):
#             param['tree_method'] = 'gpu_hist'
#             gpu_results = run_suite(param, num_rounds=20,
#                                     select_datasets = self.gdf_datasets_y_ser)
#             assert_results_non_increasing(gpu_results, 1e-2)
            

