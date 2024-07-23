def predict(alpha: float, beta: float, x_i: float) -> float:
	return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
	"""
	error from predicting 
	when the actual value is y_i
	"""
	return predict(alpha, beta, x_i) - y_i

from linear_algebra import Vector

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
	return sum(error(alpha, beta, x_i, y_i) ** 2
			for x_i, y_i in zip(x, y))

#least square solution is to choose alpha and beta
#that make sum of sqerrors as small as possible
from typing import Tuple
from linear_algebra import Vector
from statistics import correlation, standard_deviation, mean

def least_square_fit(x: Vector, y: Vector) -> Tuple[float, float]:
	"""
	Given two vectors x and y,
	find the least-squares values of alpha and beta
	"""
	beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
	alpha = mean(y) - beta * mean(x)
	return alpha, beta

# test example:
x = [i for i in range(-100, 110, 10)]
y = [3 * i -5 for i in x]

assert least_square_fit(x, y) == (-5.0, 3.0)

from statistics import num_friends_good, daily_minutes_good
alpha, beta = least_square_fit(num_friends_good, daily_minutes_good)
print("alpha beta from least_square_fit:", alpha, beta)

#A common measure of how well we've fit the data
#is using the coefficient of determination (R-squared)

from statistics import de_mean

def total_sum_of_squares(y: Vector) -> float:
	"""the total squared variation of y_i from their mean"""
	return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
	"""
	The fraction of variation is y captured by the model,
	which equals to 1 - fraction of variation in y not captured
	by the model
	"""
	return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
					total_sum_of_squares(y))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
#print(rsq)

# Using Gradient descent:
import random
import tqdm

from gradient_descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()]

learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
	for _ in t:
		alpha, beta = guess

		# Partial derivative of loss with respect to alpha
		grad_a = sum(2 * error(alpha, beta, x_i, y_i)
					for x_i, y_i in zip(num_friends_good,
										daily_minutes_good))

		# Partial derivative of loss with respect to beta
		grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
					for x_i, y_i in zip(num_friends_good,
										daily_minutes_good))

		# Compute loss to stick in the tqdm description:
		loss = sum_of_sqerrors(alpha, beta,
								num_friends_good, daily_minutes_good)

		t.set_description(f"loss: {loss:.3f}")

		#Update the guess:
		guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

alpha, beta = guess 
print("alpha beta from gradient_descent:", alpha, beta)