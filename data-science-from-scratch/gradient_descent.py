from linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
	return dot(v, v)

# Estimating the Gradient:

from typing import Callable

def difference_quotient(f: Callable[[float], float],
					x: float,
					h: float) -> float:
	return (f(x + h) - f(x)) / h

# Estimate derivative:

def square(x: float) -> float:
	return x * x

# derivative of square function:
def derivative(x: float) -> float:
	return 2 * x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h = 0.0001) for x in xs]

#Plot to show they are basically the same:
import matplotlib.pyplot as plt
plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label = 'Actual')
plt.plot(xs, estimates, 'b+', label = 'Estimates')
plt.legend(loc = 9)
#plt.show()
plt.close()

def partial_difference_quotient(f: Callable[[Vector], float],
								v: Vector,
								i: int,
								h: float) -> float:
	"""Returns the i-th partial difference quotient of f at v """
	w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]

	return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
					v: Vector,
					h: float = 0.00001):
	return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

import random

from linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
	""" Moves `step_size` in the `gradient` direction from `v`"""
	assert len(v) == len(gradient)

	step = scalar_multiply(step_size, gradient)
	return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
	return [2 * v_i for v_i in v]

# pick a random starting point:
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
	grad = sum_of_squares_gradient(v)
	v = gradient_step(v, grad, -0.01)
	# print(epoch, v)


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual)
    squared_error = error ** 2           # We'll minimize squared error
    grad = [2 * error * x, 2 * error]    # using its gradient.
    return grad

from linear_algebra import vector_mean
# First "Using Gradient Descent to Fit Models" Example:
theta = [random.uniform(-1, 1) , random.uniform(-1, 1)]

learning_rate = 0.001

inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

for epoch in range(5000):
	#Compute the mean of the gradients:
	grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])

	#Take a step in that direction:
	theta = gradient_step(theta, grad, -learning_rate)
	#print(epoch, theta)

slope, intercept = theta
print("slope: ", slope, "intercept: ", intercept)

# MiniBatch and Stochastic Gradient Descent:
from typing import TypeVar, List, Iterator

# this allows us to type "generic" functions:
T = TypeVar('T')

def minibatches(dataset: List[T],
				batch_size: int,
				shuffle: bool = True) -> Iterator[List[T]]:

	"""Generates the `batch_size`-sized minibatches from the dataset"""

	batch_starts = [start for start in range(0, len(dataset), batch_size)]

	if shuffle: random.shuffle(batch_starts)

	for start in batch_starts:
		end = start + batch_size
		yield dataset[start:end]

"""
======================MiniBatches Example===================
"""
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
	for batch in minibatches(inputs, batch_size = 20):
		grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
		theta = gradient_step(theta, grad, -learning_rate)

slope, intercept = theta
print("For minibatches example: slope: ", slope, "intercept: ", intercept)


"""
======================Stochastic Gradient Descent Example===================
"""
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100):
	for x, y in inputs:
		grad = linear_gradient(x, y, theta)
		theta = gradient_step(theta, grad, -learning_rate)


slope, intercept = theta
print("For SGD example: slope: ", slope, "intercept: ", intercept)