from typing import List

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
	"""Adds corresponding elements"""
	assert len(v) == len(w), "vectors must be the same length"

	return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1,2,3],[4,5,6]) == [5,7,9]

def subtract(v: Vector, w: Vector) -> Vector:
	"""Adds corresponding elements"""
	assert len(v) == len(w), "vectors must be the same length"

	return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
	"""Sums all corresponding elements """
	assert vectors, "no vectors provided!"

	# Check the vectors are all the same size:
	num_elements = len(vectors[0])
	assert all(len(v) == num_elements for v in vectors), "different sizes!"

	# the i-th element of the result is the sum of every vector[i]
	return [sum(vector[i] for vector in vectors)
				for i in range(num_elements)]

assert vector_sum([[1,2],[3,4],[5,6],[7,8]]) == [16,20]


def scalar_multiply(c: float, v: Vector) -> Vector:
	return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
	n = len(Vector)
	return scalar_multiply(1/n, vector_sum(vectors))

def dot(v: Vector, w: Vector) -> float:
	assert len(v) == len(w), "vector must be same length"

	return sum(v_i * w_i for v_i, w_i in zip(v,w))

def sum_of_squares(v: Vector) -> float:
	return dot(v, v)


import math

def magnitude(v: Vector) -> float:
	return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
	"""Computes (v_1 - w_1) ** 2 + ... (v_n - w_n) **2"""
	return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
	return math.sqrt(squared_distance(v, w))

def distance(v: Vector, w: Vector) -> float:
	return magnitude(subtract(v, w))


# Another type Alias:
Matrix = List[List[float]]

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
	num_rows = len(A)
	num_cols = len(A[0]) if A else 0
	return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
	""" Returns the i-th row of A (as a Vector)"""
	return A[i]

def get_column(A: Matrix, j: int) -> Vector:
	""" Returns the j-th column of A (as a Vector) """
	return [A_i[j] for A_i in A]


from typing import Callable

def make_matrix(num_rows: int,
				num_cols: int,
				entry_fn: Callable[[int, int], float]) -> Matrix:

	"""
	Returns a num_rows x num_cols matrix
	whose (i,j)-th entry is entry_fn(i, j)
	"""
	return [[entry_fn(i, j)
			for j in range(num_cols)]
			for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
	"""Returns the n x n identify matrix"""
	return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

from pprint import pprint
#pprint(identity_matrix(5))