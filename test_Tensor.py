import scipy.sparse as sp
from Tensor import Tensor
import pytest
from random import randint


'''-----------------------------------------------------------------------------
                              constructor tests
-----------------------------------------------------------------------------'''

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

  with pytest.warns(RuntimeWarning, match = "slice format .*"):
    A = Tensor(slices)


'''-----------------------------------------------------------------------------
                              save/load tests
-----------------------------------------------------------------------------'''

'''
def test_save_load():
  slices = []

  T = 2
  n = 10
  m = 9
  for t in range(T):
    slices.append(sp.random(n,m))
'''
'''-----------------------------------------------------------------------------
                              transpose tests
-----------------------------------------------------------------------------'''

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

'''-----------------------------------------------------------------------------
                              get_scalar tests
-----------------------------------------------------------------------------'''
def test_working_get_scalar():
  slices = []

  T = 2
  n = 10
  m = 9

  for t in range(T):
    slices.append(sp.random(n, m, density=.5,format = 'dok'))

  A = Tensor(slices)

  for i in range(n):
    for j in range(m):
      for t in range(T):
        assert A.get_scalar(i,j,t) == slices[t][i,j]

def test_get_scalar_warnings():
  slices = []

  T = 2
  n = 10
  m = 9

  for t in range(T):
    slices.append(sp.random(n, m, density=.5))

  A = Tensor(slices)

  with pytest.warns(RuntimeWarning):
    A.get_scalar(0,0,0)

def test_get_scalar_errors():
  slices = []

  T = 2
  n = 10
  m = 9

  for t in range(T):
    slices.append(sp.random(n, m, density=.5))

  A = Tensor(slices)

  with pytest.raises(ValueError):
    A.get_scalar(2 * n,2 *m,2 * T)
    A.get_scalar(-2* n, -2 * m, -2 * T)

'''-----------------------------------------------------------------------------
                              set_scalar tests
-----------------------------------------------------------------------------'''
def test_working_set_scalar():
  slices = []

  T = 2
  n = 10
  m = 9

  for t in range(T):
    slices.append(sp.random(n, m, density=.5,format='dok'))

  A = Tensor(slices)
  rand_i = randint(0,n-1)
  rand_j = randint(0,m-1)
  rand_t = randint(0,T-1)
  val = randint(0,1232)

  A.set_scalar(rand_i,rand_j,rand_t,val)
  assert A.get_scalar(rand_i,rand_j,rand_t) == val

def test_set_scalar_warnings():
  slices = []

  T = 2
  n = 10
  m = 9

  for t in range(T):
    slices.append(sp.random(n, m, density=.5))
  A = Tensor(slices)

  with pytest.warns(RuntimeWarning):
    A.set_scalar(0,0,0,3)

def test_set_scalar_errors():
  slices = []

  T = 2
  n = 10
  m = 9

  for t in range(T):
    slices.append(sp.random(n, m, density=.5,format='dok'))

  A = Tensor(slices)
  with pytest.raises(TypeError):
    A.set_scalar(0,0,0,[1,2,3])
    A.set_scalar(0, 0, 0, "apples")
    A.set_scalar(0,0,0,sp.random(3,2))


