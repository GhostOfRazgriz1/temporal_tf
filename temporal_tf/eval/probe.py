import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def linear_probe(features, labels):
    X = features.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))
