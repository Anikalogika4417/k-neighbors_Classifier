import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


x_blue = np.array([0.3, 0.5, 1, 1.4, 1.7, 2])
y_blue = np.array([1, 4.5, 2.3, 1.9, 8.9, 4.1])

x_red = np.array([3.3, 3.5, 4, 4.4, 5.7, 6])
y_red = np.array([7, 1.5, 6.3, 1.9, 2.9, 7.1])

result_array_blue = np.column_stack((x_blue, y_blue))
result_array_red = np.column_stack((x_red, y_red))

X = np.concatenate((result_array_blue, result_array_red), axis=0)
Y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


plt.plot(x_blue, y_blue, 'ro', color='blue')
plt.plot(x_red, y_red, 'ro', color='red')
plt.plot(3,5, 'ro', color='green', markersize=15)
plt.axis([-0.5, 10, -0.5, 10])

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,Y)

predict = classifier.predict(np.array([[5, 5]]))
print('result: ', predict)
