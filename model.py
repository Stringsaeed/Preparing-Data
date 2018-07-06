from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def scl(features):
    sc = StandardScaler()
    return sc.fit_transform(features)


def svc(X, y):
    clf = SVC(kernel='linear', random_state=0)
    return clf.fit(X, y)
