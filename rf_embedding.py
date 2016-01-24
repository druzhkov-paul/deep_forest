import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomTreesEmbedding, BaseForest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_array, issparse, check_random_state
import xgboost
from xgboost import XGBClassifier, DMatrix
from sklearn.externals.joblib import Parallel, delayed
import numpy as np

print("sklearn version", sklearn.__version__)
print("xgboost version", xgboost.__version__)


class RandomTreesEmbeddingSupervised(ExtraTreesClassifier):
    def __init__(self,
                 n_estimators=10,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 sparse_output=True,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 use_one_hot=True):
        super(RandomTreesEmbeddingSupervised, self).__init__(
            n_estimators=n_estimators,
            criterion="gini",
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = 1
        self.max_leaf_nodes = max_leaf_nodes
        self.sparse_output = sparse_output
        self.use_one_hot = use_one_hot

    def fit(self, X, y=None, sample_weight=None):
        super(RandomTreesEmbeddingSupervised,
              self).fit(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse=['csc'], ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        self.fit(X, y, sample_weight=sample_weight)
        coding = self.apply(X)

        if self.use_one_hot:
            self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
            coding = self.one_hot_encoder_.fit_transform(coding)

        return coding

    def transform(self, X):
        coding = self.apply(X)
        if self.use_one_hot:
            coding = self.one_hot_encoder_.transform(coding)
        return coding


class RandomTreesEmbeddingUnsupervised(BaseForest):
    def __init__(self,
                 n_estimators=10,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 sparse_output=True,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 use_one_hot=True):
        super(RandomTreesEmbeddingUnsupervised, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = 'mse'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = 1
        self.max_leaf_nodes = max_leaf_nodes
        self.sparse_output = sparse_output
        self.use_one_hot = use_one_hot

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse=['csc'], ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])
        super(RandomTreesEmbeddingUnsupervised,
              self).fit(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight=sample_weight)
        coding = self.apply(X)
        if self.use_one_hot:
            self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
            coding = self.one_hot_encoder_.fit_transform(coding)

        return coding

    def transform(self, X):
        coding = self.apply(X)
        if self.use_one_hot:
            coding = self.one_hot_encoder_.transform(coding)
        return coding


class GBTreesEmbeddingSupervised(GradientBoostingClassifier):
    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, init=None, random_state=None,
                 max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False, use_one_hot=True,
                 n_jobs=1, sparse_output=True):
        super(GBTreesEmbeddingSupervised, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start)

        self.use_one_hot = use_one_hot
        self.n_jobs = n_jobs
        self.sparse_output = sparse_output

    def fit(self, X, y, sample_weight=None, monitor=None):
        super(GBTreesEmbeddingSupervised,
              self).fit(X, y, sample_weight=sample_weight, monitor=monitor)
        return self

    def fit_transform(self, X, y, sample_weight=None, monitor=None):
        self.fit(X, y, sample_weight=sample_weight, monitor=monitor)
        coding = self.apply(X)
        if self.use_one_hot:
            self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
            coding = self.one_hot_encoder_.fit_transform(coding)
        return coding

    def transform(self, X):
        coding = self.apply(X)
        if self.use_one_hot:
            coding = self.one_hot_encoder_.transform(coding)
        return coding


class XGBTEmbeddingSupervised(XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None, use_one_hot=True,
                 sparse_output=True):
        super(XGBTEmbeddingSupervised, self).__init__(
            max_depth=max_depth, learning_rate=learning_rate,
            n_estimators=n_estimators, silent=silent,
            objective=objective,
            nthread=nthread, gamma=gamma, min_child_weight=min_child_weight,
            max_delta_step=max_delta_step, subsample=subsample,
            colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
            base_score=base_score, seed=seed, missing=missing)

        self.use_one_hot = use_one_hot
        self.sparse_output = sparse_output

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        super(XGBTEmbeddingSupervised,
              self).fit(X, y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,
                        early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        return self

    def transform(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing)
        coding = self.booster().predict(test_dmatrix,
                                        output_margin=output_margin,
                                        ntree_limit=ntree_limit,
                                        pred_leaf=True)
        if self.use_one_hot:
            self.one_hot_encoder_ = OneHotEncoder(sparse=self.sparse_output)
            coding = self.one_hot_encoder_.fit_transform(coding)
        return coding

    def fit_transform(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
                      early_stopping_rounds=None, verbose=True, output_margin=False):
        self.fit(X, y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,
                 early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        coding = self.transform(X, output_margin=output_margin)
        return coding
