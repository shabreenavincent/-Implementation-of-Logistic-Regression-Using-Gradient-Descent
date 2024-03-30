# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import numpy as np,matplotlib.pyplot as plt and optimize from scipy.
2. Print the array values of X and Y.
3. By plotting the X and Y values get the Sigmoid function grap.
4. Print the grad values of X and Y.
5. Plot the decision boundary of the given data.
6. Obtain the probability value.
7. Get the prediction value of mean and print it.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHABREENA VINCENT
RegisterNumber:  212222230141
```
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:, [0, 1]]
y = data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10, 10, 100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X, theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

def cost(theta, X, y):
  h= sigmoid(np.dot(X, theta))
  J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
  return J

def gradient(theta, X, y):
  h = sigmoid(np.dot(X, theta))
  grad = np.dot(X.T, h - y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta, X, y):
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0], 1)), X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x, X, y)

prob = sigmoid(np.dot(np.array([1, 45, 85]), res.x))
print(prob)

def predict(theta, X):
  X_train = np.hstack((np.ones((X.shape[0], 1)), X))
  prob = sigmoid(np.dot(X_train, theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x, X) == y)
```
## Output:

Array Value of x:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/8addc02a-f4eb-45c0-93b6-73b1343b5902)

Array Value of y:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/bbc81769-1622-49f5-b996-497017d37fc3)

Exam 1 - score graph:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/93b1d3cc-b255-4c2f-9136-c86947992e08)

Sigmoid function Graph:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/c8683429-c7d7-44f5-b354-a8c0fe64a7cb)

X_train_grad value:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/2b051b91-910c-4a90-bead-855b0983d6e1)

Y_train_grad value:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/3ceb4da8-c5b0-4722-997e-136bb5827b7b)

Print res.x:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/8cd1eabf-3e59-43cd-b2bd-f6941dad674b)

Decision Boundary - graph for exam score:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/5ab72b30-51b5-44fc-87ba-d4c8a343280b)

Probability value:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/9247a2dd-c550-45ba-b2f4-da94d8088ed0)

Prediction value of mean:

![image](https://github.com/DHARINIPV/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119400845/3647e888-bdfa-4bce-bea1-545f0b4e47ac)

## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
