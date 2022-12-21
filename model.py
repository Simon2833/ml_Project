import numpy as np
from bagging import MyBaggingclassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

clfs = {
    'SGD': SGDClassifier(loss="log"),
    'GNB': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'DT': DecisionTreeClassifier()

}

datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes', 'ecoli4', 'glass2', 'heart', 'ionosphere',
            'liver', 'monkthree', 'soybean', 'vowel0', 'wisconsin']

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1234)

scores = np.zeros((len(datasets), 10, len(clfs), 3, 3, 21))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    for i in range(len(y)):
        if(y[i] in [2, 3, 4, 5, 6, 7, 8, 9]): y[i] = 1

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            for boots in range(0, 3):
                for comb in range(0, 3):
                    for n_clf in range(0, 20):
                        clf = clone(MyBaggingclassifier(base_estimator=clfs[clf_name], n_estimators=(n_clf+1)*5, random_state=1234, boots_type=boots+1, comb_tech=comb+1))
                        clf.fit(X[train], y[train])
                        y_pred = clf.predict(X[test])
                        scores[data_id, fold_id, clf_id, boots, comb, n_clf] = accuracy_score(y[test], y_pred)
np.save("results", scores)
