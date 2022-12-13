import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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


# Function that fills out values in plot, used mainly for rounding and visibility.
def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("white", "black"), threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

###################################################


scores = np.load("results.npy")
scores = np.mean(scores, axis=0)
scores = np.mean(scores, axis=0)
scores = np.delete(scores, 20, 3)

x1 = 0
x2 = 0

boots = ["majority vote", "support matrices", "weighted majority vote"]
clsf = ["SGD", "GNB", "KNN", "SVM", "DT"]

comb = ["samples", "features", "both"]
n_clf = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

fig, ((a0, a1, a2),
      (a3, a4, a5),
      (a6, a7, a8),
      (a9, a10, a11),
      (a12, a13, a14)) = plt.subplots(5, 3, figsize=(18, 10))
nr = 0
nrr = 0
a = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14]
for ax in a:

    value = scores[x1, x2]

    im = ax.imshow(value)

    if (x1 == 0):
        ax.set_xlabel(boots[nr], va="bottom", fontweight='bold', labelpad=-110)
        nr += 1
    elif (x1 == 4):
        ax.set_xlabel("n_clf", va="center", labelpad=15)

    if (x2 == 0):
        ax.set_ylabel(clsf[nrr], rotation=90, va="bottom", fontweight='bold')
        nrr += 1
    elif (x2 == 2):
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("comb", rotation=-90, va="bottom", labelpad=10)

    x2 += 1
    if (x2 == 3):
        x2 = 0
        x1 += 1

    annotate_heatmap(im, valfmt="{x:.2f}", size=7)

    ax.xaxis.set_ticks(np.arange(len(n_clf)), labels=n_clf)
    ax.yaxis.set_ticks(np.arange(len(comb)), labels=comb)

fig.tight_layout()
fig.suptitle('Accuracy rate on (Bootstrap_technique*Classifiers) using value of (Number_of_classifiers*Combination_technique)',
             fontsize=16)

plt.savefig("heatmap.png")

# Still working on statistical tests, probably add them in near future.
