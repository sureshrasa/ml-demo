from sklearn.datasets.samples_generator import make_blobs, make_moons, make_circles
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np

# generate 2d classification dataset
#X, y = make_blobs(n_samples=1000, centers=3, n_features=25)
#X, y = make_moons(n_samples=1000, noise=0.1)
X, y = make_circles(n_samples=1000, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

print(X.shape)
print(y.shape)

np.savetxt("Xdata.txt", X)
np.savetxt("ydata.txt", y)
