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
                             convert slices test
-----------------------------------------------------------------------------'''
def test_convert_slices():
  n = 5
  m = 7
  T = 5
  slices = []
  for t in range(T):
    slices.append(sp.random(n,m))

  A = Tensor(slices)
  A.convert_slices('dok')

  assert A._slice_format == 'dok'
  for t in range(T):
    assert A._slices[t].format == 'dok'

'''-----------------------------------------------------------------------------
                            get/set slices tests
-----------------------------------------------------------------------------'''
def test_get_frontal_slice():
  n = 5
  m = 7
  T = 5
  slices = []
  for t in range(T):
    slices.append(sp.random(n,m))
  A = Tensor(slices)

  for t in range(T):
    assert (A.get_frontal_slice(t) - slices[t]).nnz == 0

def test_working_get_frontal_slice():
  n = 5
  m = 7
  T = 5
  slices = []
  for t in range(T):
    slices.append(sp.random(n,m))
  A = Tensor(slices)

  new_X = sp.random(n,m)
  randT = randint(0,T-1)
  A.set_frontal_slice(randT,new_X)
  assert (A.get_frontal_slice(randT) - new_X).nnz == 0

def test_get_frontal_slice_errors_and_warnings():
  n = 5
  m = 7
  T = 5
  slices = []
  for t in range(T):
    slices.append(sp.random(n, m))
  A = Tensor(slices)

  #non-sparse matrix errors
  with pytest.raises(TypeError):
    A.set_frontal_slice(0,'apple')
    A.set_frontal_slice(0,4)

  #wrong shape
  with pytest.raises(ValueError):
    A.set_frontal_slice(0,sp.random(n+1,m+1))

  #warn about wrong type
  with pytest.warns(UserWarning):
    A.set_frontal_slice(0,sp.random(n,m,format='lil'))

def test_expanding_tensor():
  n = 5
  m = 7
  T = 5
  k = 2
  slices = []
  for t in range(T):
    slices.append(sp.random(n,m))

  A = Tensor(slices)
  new_slice = sp.random(n,m)
  A.set_frontal_slice(T+2,new_slice)

  assert A.shape == (n,m,T+k)
  for t in range(T,T+k):
    if t == T+k -1:
      assert (A.get_frontal_slice(t) - new_slice).nnz == 0
    else:
      assert (A.get_frontal_slice(t)).nnz == 0


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
  A.transpose(inPlace=True)

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


'''-----------------------------------------------------------------------------
                              squeeze tests
-----------------------------------------------------------------------------'''
def test_squeeze_passed_in_slice():
  n = 10
  m = 9

  X = sp.random(n,m,format='dok')