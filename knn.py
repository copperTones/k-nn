import numpy as np

'''
Consider a single dimension (variable) X. Obtain N = 100 iid samples x1, x2, ... of X
uniformly randomly between 1 and 11, and then obtain the corresponding y values as the
natural logarithm of x plus a Gaussian noise (mean 0, standard deviation 0.2), with different
points having different amounts of noise (and the noise is independent of the x-value).
'''

def default_phi(x):
	return np.ones(x.shape)

def default_metric(x):
	return np.sqrt(np.sum(x*x, 1))

# really easy to screw up, but hella abstracted
def knn(x0, x, y, k,
	phi=default_phi,
	metric=default_metric):
	dist = metric(x - x0)
	lo = np.argsort(dist)[:k]
	w = phi(dist[lo])
	return np.sum(w * y[lo]) / np.sum(w)

if __name__ == "__main__":
	rng = np.random.default_rng(196883)
	x = rng.uniform(1, 11, 100)
	y = np.log(x) + rng.normal(0, .2)
	for label, phi in [("equal", default_phi), ("inverse", lambda x: 1/x)]:
		print(label)
		for k in [1, 3, 50]:
			print("K =", k)
			print([knn([x0], x, y, k, phi, np.abs)
				for x0 in range(1, 11, 2)])
	def gauss(x):
		return np.e ** (-x*x/2)
	print([knn([x0], x, y, 100, gauss, np.abs)
		for x0 in range(1, 11, 2)])
