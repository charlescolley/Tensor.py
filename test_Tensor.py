import scipy.sparse as sp
from Tensor import Tensor
import pytest


from random import randint
def test_empty_constructor():
  A = Tensor()

  assert A.shape[0] == 0
  assert A.shape[1] == 0
  assert A.shape[2] == 0

  assert  A._slices == []

def test_non_empty_valid_constructor():
  slices = []
  T = 3
  n = 5
  m = 7

  for t in range(T):
    slices.append(sp.random(n,m,density=.8))

  A = Tensor(slices)

  assert A.shape[0] == n
  assert A.shape[1] == m
  assert A.shape[2] == T

  assert A._slices == slices

#case where slices are different sizes
def test_invalid_slices_constructors():
  slices = []
  T = 3
  n = 10
  m = 8

  for t in range(T):
    slices.append(sp.random(randint(1,n),randint(1,m)))
  with pytest.raises(ValueError,match=r'slices must all have the same shape.*'):
    A = Tensor(slices)

def test_inconsistent_matrix_type_constructor():
  slices = []
  T =2
  n = 10
  m = 9
  slices.append(sp.random(n,m,format='dok'))
  slices.append(sp.random(n,m,format='csr'))

  with pytest.warns(UserWarning, match = "slice format .*"):
    A = Tensor(slices)


'''
def test_save_load():
  slices = []

  T = 2
  n = 10
  m = 9
  for t in range(T):
    slices.append(sp.random(n,m))
'''

def test_transpose_in_place():
  slices = []

  T = 2
  n = 10
  m = 9

  for t in range(T):
    slices.append(sp.random(n,m,density=.5))

  A = Tensor(slices)
  A.transpose(InPlace=True)

  assert A.shape == (m,n,T)

  for t in range(T):
    if t == 0:
      assert (A._slices[t] - slices[0].T).nnz == 0
    else:
      assert (A._slices[t] - slices[:0:-1][t-1].T).nnz == 0
