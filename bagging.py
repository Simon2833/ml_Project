import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import scipy
import warnings
warnings.filterwarnings('ignore')


class MyBaggingclassifier(BaseEnsemble, ClassifierMixin):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 random_state=None,
                 boots_type=1,
                 comb_tech=1):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.ensemble_ = []
        self.oob_error = []
        self.random = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            self.ensemble_.append(clone(self.base_estimator))

        # There are 3 types of bootstrapping methods to chose by instances(1), features(2) or both(3),
        # any other will give warning and stop program.
        self.boots_type = boots_type

        # There are 3 types of decision combination methods to chose: hard voting(1), soft voting(2) or weighted hard voting(3),
        # any other will give warning and stop program.
        self.comb_tech = comb_tech

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.n_samples = n_samples
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.subspaces_nr = []

        # Size of bootstrap sample by features
        features_am = int(np.sqrt(X.shape[1]))

        # Size of bootstrap sample by instances with stratified classes
        samples_0 = []
        samples_1 = []
        for i in range(X.shape[0]):
            if(y[i] == 0):
                samples_0.append(i)
            elif(y[i] == 1):
                samples_1.append(i)
        samples_am0 = int(np.sqrt(len(samples_0)))
        samples_am1 = int(np.sqrt(len(samples_1)))

        for cls in self.ensemble_:
            nr_idxy0 = self.random.choice(samples_0, samples_am0)
            nr_idxy1 = self.random.choice(samples_1, samples_am1)
            nr_idxy = np.concatenate((nr_idxy0, nr_idxy1))
            nr_idxX = self.random.choice(X.shape[1], features_am)
            self.subspaces_nr.append(nr_idxX)

            if(self.boots_type == 1):
                cls.fit(X[nr_idxy], y[nr_idxy])

                # 3rd combination method is used only with hard voting.
                # Other bootstrap methods are given soft voting so it can be confusing,
                # but it was made so everything was symmetrical and results are only for my exercise.
                if (self.comb_tech == 3):
                    unidx = np.unique(nr_idxy)
                    oob = []
                    for i in range(X.shape[0]):
                        if i not in unidx: oob.append(i)
                    oobpred = cls.predict(X[oob])
                    self.oob_error.append(accuracy_score(y[oob], oobpred))

            elif(self.boots_type == 2):
                cls.fit(X[:, nr_idxX], y[:])
            elif(self.boots_type == 3):
                cls.fit(X[np.ix_(nr_idxy, nr_idxX)], y[nr_idxy])
            else:
                warnings.warn("Argument boots_type can only be integer between 1 and 3.")
                exit()

        return self

####################################################

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        pred_ = []

        # Everyone in ensemble vote for their prediction and one with most votes is chosen.
        # Combination method 3 with boots method [2,3] are hard voting to fill out rest of combination, because we won't use them in weighted had voting.
        if(self.comb_tech == 1 or (self.comb_tech == 3 and self.boots_type in [2, 3])):
            for i, member_clf in enumerate(self.ensemble_):
                if(self.boots_type == 1):
                    pred_.append(member_clf.predict(X))
                elif(self.boots_type in [2, 3]):
                    pred_.append(member_clf.predict(X[:, self.subspaces_nr[i]]))
                else:
                    warnings.warn("Argument boots_type can only be integer between 1 and 3.")
                    exit()

            pred_ = np.array(pred_)
            prediction = scipy.stats.mode(pred_.T, axis=1, keepdims=True)
            return self.classes_[prediction[0]]

        # Every class is given probability using support matrices.
        elif(self.comb_tech == 2):
            esm = self.ensemble_support_matrix(X)
            average_sup = np.mean(esm, axis=0)
            prediction = np.argmax(average_sup, axis=1)
            return self.classes_[prediction]

        # Basically hard voting but with weights that show how important each prediction is looking at whole.
        elif(self.comb_tech == 3 and self.boots_type == 1):
            score = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X))

            for i in range(len(pred_[0])):
                for_0 = 0
                for_1 = 0
                for j in range(len(pred_)):
                    if(pred_[j][i] == 0):
                        for_0 = for_0 + self.oob_error[j]
                    elif(pred_[j][i] == 1):
                        for_1 = for_1 + self.oob_error[j]
                if(for_0 > for_1):
                    score.append(0)
                else:
                    score.append(1)
            return score

        else:
            warnings.warn("Argument comb_tech can only be integer between 1 and 3. If you want to use 3 option boots_type need to be set as 1.")
            exit()

    # Prediction of probability for each class
    def ensemble_support_matrix(self, X):
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            if(self.boots_type == 1):
                probas_.append(member_clf.predict_proba(X))

            elif(self.boots_type in [2, 3]):
                probas_.append(member_clf.predict_proba(X[:, self.subspaces_nr[i]]))
            else:
                warnings.warn("Argument boots_type can only be integer between 1 and 3.")
                exit()
        return np.array(probas_, dtype=object)


####################################################
