import numpy as np

from logistic_python import BinaryLogisticRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def test_constructor():
	clf = BinaryLogisticRegression()

def test_classification():
	# Tests if it can make a classification without failing
	X, y = make_classification(n_samples=20, n_features=5, random_state=0)

	clf = BinaryLogisticRegression()
	clf.fit(X, y)
	probs = clf.predict_proba(X)
	y_pred = clf.predict(X)


def test_compare_sklearn():
	# Tests to see if the same predictor labels are obtained from sklearn
	X, y = make_classification(n_samples=10, n_features=5, random_state=0)

	clf = BinaryLogisticRegression()
	clf.fit(X, y)
	probs = clf.predict_proba(X)
	y_pred = clf.predict(X)

	skclf = LogisticRegression(C=1., solver='lbfgs')
	skclf.fit(X, y)
	sk_probs = skclf.predict_proba(X)
	sk_y_pred = skclf.predict(X)

	print probs
	print sk_probs

	assert(np.all(y_pred == sk_y_pred))
	assert(np.all(probs == sk_probs))

