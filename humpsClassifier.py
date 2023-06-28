from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.decomposition import PCA
import warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

class HumpsClassifier:
    def __init__(self):
        self.pretty_segments_df = pd.read_pickle('data/pretty_segments_df.pkl')
        self.channels_arr = list(np.load(open('data/channels_arr.npy','rb')))
        self.feature_cols = ['median', 'max', 'middle', 'mean_abs_deriv', 'median_abs_deriv','mean','raw_std', 'dip']
        index_to_aa = [c for c in 'CSAGTVNQMILYWFPHRKDE']
        self.aa_to_index = {aa:i for i, aa in enumerate(index_to_aa)}

    def create_rf_upsample_feature_matrix(self, acids, 
                                    num_features=64, 
                                    include_manual=True, 
                                    include_matrix=True, 
                                    accuracy_metric=accuracy_score,
                                    n_repeats=5,
                                    needs_classes = False,
                                    test_runs = None,
                                    classifiers = [
                                        KNeighborsClassifier(3),
                                        RandomForestClassifier(),
                                        AdaBoostClassifier(), 
                                        GaussianProcessClassifier()
                                    ],
                                    use_proba = False,
                                    return_clf = False,
                                    test_size = 0.2):

        feature_cols = self.feature_cols 
        aa_to_index = self.aa_to_index 

        pretty_segments_df = self.pretty_segments_df
        channels_arr = self.channels_arr

        if not include_manual and not include_matrix:
            print("You want me to predict without any features??")
            return
        accs = []
        best_clfs = []
        for i in range(n_repeats):
            data_df = pretty_segments_df[pretty_segments_df.aa.isin(acids)]
            if test_runs:
                train_df = data_df[~data_df.run.isin(test_runs)]
                test_df = data_df[data_df.run.isin(test_runs)]
                assert len(set(train_df.index.values).intersection(set(test_df.index.values))) == 0
            else: 
                train_df, test_df = train_test_split(data_df, test_size=test_size)
            del data_df

            dfs = [train_df[train_df.aa == AA] for AA in acids]
            lens = [len(x) for x in dfs]
            biggest_aa_i = np.argmax(lens)
            biggest_aa = acids[biggest_aa_i]
            target_cnt = lens[biggest_aa_i]

            # upsample minority residues in the TRAINING SET only
            for i, AA_minority in enumerate(acids):
                if AA_minority == biggest_aa:
                    continue
                assert target_cnt >= lens[i]
                re_balance = target_cnt - lens[i]
                df_minority_upsampled = resample(dfs[i], 
                                                    replace=True,     # sample with replacement
                                                    n_samples=re_balance,    # to match majority class
                                                    random_state=42) # reproducible results
                train_df = pd.concat([train_df, df_minority_upsampled])

            # Convert amino acid letter to number for the classification values
            y_train = [aa_to_index[a] for a in train_df.aa.values]
            y_test = [aa_to_index[a] for a in test_df.aa.values]

            # Extract the features from dataframe, based on arguments include_matrix and include_manual
            if include_matrix:
                pca_fit_df = pd.concat([train_df, pretty_segments_df[~pretty_segments_df.aa.isin(acids)]])
                kosher_indices = [channels_arr.index(i) for i in pca_fit_df.index.values]
                pca = PCA(num_features if num_features < len(pca_fit_df) else len(pca_fit_df))
                pca.fit([x[kosher_indices] for i, x in enumerate(pca_fit_df.features.values) if i in kosher_indices])
                
                pca_X_train = pca.transform([x[kosher_indices] for x in train_df['features'].values])
                pca_X_test = pca.transform([x[kosher_indices] for x in test_df['features'].values])

                if include_manual:
                    X_train = [np.concatenate((pca, x.to_numpy())) for pca, (_,x) in zip(pca_X_train, train_df.loc[:, feature_cols].iterrows())]
                    X_test = [np.concatenate((pca, x.to_numpy())) for pca, (_,x) in zip(pca_X_test, test_df.loc[:, feature_cols].iterrows())]
                else:
                    X_train = pca_X_train
                    X_test = pca_X_test

            else: # only manual features 
                X_train = train_df.loc[:, feature_cols]
                X_test = test_df.loc[:, feature_cols]

            if len(classifiers) == 1:
                clf = classifiers[0]
                clf.fit(X_train, y_train)  
                test_pred = clf.predict_proba(X_test) if use_proba else clf.predict(X_test) 
                accs.append(accuracy_metric(y_test, test_pred, clf.classes_) if needs_classes else accuracy_metric(y_test, test_pred))
                best_clfs.append(clf)
            else:
                best_score = 0
                best_classifier = 'None'
                for clf in classifiers:
                    clf.fit(X_train, y_train)
                    test_pred = clf.predict_proba(X_test) if use_proba else clf.predict(X_test) 
                    acc = accuracy_metric(y_test, test_pred, clf.classes_) if needs_classes else accuracy_metric(y_test, test_pred)
                    if acc > best_score:
                        best_score = acc
                        best_classifier = clf.__class__
                accs.append(best_score)
                best_clfs.append(best_classifier)
        # print(best_clfs)
        # return accs
        if return_clf:
            return np.mean(accs), accs, best_clfs
        return np.mean(accs), accs
