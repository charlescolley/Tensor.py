import scipy.sparse as sp
import numpy as np
from Tensor import Tensor
import Tensor as Te
import pytest
from tempfile import NamedTemporaryFile
from random import randint, uniform

#GLOBAL TEST VARIABLES
N = 6
M = 7
T = 5


def set_up_tensor(n,m,k, format='coo',dense = False):
  if dense:
    slices = np.random.rand(n,m,k)
  else:
    slices = []
    for t in range(k):
      slices.append(sp.random(n,m,density=.5,format = format))

  return Tensor(slices), slices


'''-----------------------------------------------------------------------------
                              constructor tests
-----------------------------------------------------------------------------'''

def test_empty_constructor():

  A = Tensor()

  assert A.shape[0] == 0
  assert A.shape[1]  == 0
  assert A.shape[2] == 0

  assert  A._slices == []

def test_file_constructor():

   A, slices = set_up_tensor(N,M,T)

   with NamedTemporaryFile() as tmp_file:
     A.save(tmp_file.name)

     B = Tensor(tmp_file.name)

     assert B.shape[0] == N
     assert B.shape[1] == M
     assert B.shape[2] == T
     assert B._slice_format == "coo"

   assert B == A

def test_non_empty_valid_constructor():
  A,slices = set_up_tensor(N,M,T)

  assert A.shape[0] == N
  assert A.shape[1] == M
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

  with pytest.raises(ValueError,match=r'ndarray must be of order 3, slices.*'):
    Tensor(np.random.rand(1,2,3,4,5))
    Tensor(np.random.rand(1,2))

  with

def test_inconsistent_matrix_type_constructor():
  slices = []
  T =2
  n = 10
  m = 9
  slices.append(sp.random(n,m,format='dok'))
  slices.append(sp.random(n,m,format='csr'))

  with pytest.warns(RuntimeWarning, match = "slice format .*"):
    A = Tensor(slices)

def test_dense_tensor_constructor():
  A, slices = set_up_tensor(N,M,T,dense=True)

  assert A.shape[0] == N
  assert A.shape[1] == M
  assert A.shape[2] == T
  assert A._slice_format == "dense"
  assert (A._slices == slices).all()


'''-----------------------------------------------------------------------------
                             convert slices test
-----------------------------------------------------------------------------'''
def test_convert_slices():
  A,slices = set_up_tensor(N,M,T)
  A.convert_slices('dok')

  assert A._slice_format == 'dok'
  for t in range(T):
    assert A._slices[t].format == 'dok'

'''-----------------------------------------------------------------------------
                            get/set slices tests
-----------------------------------------------------------------------------'''
def test_get_frontal_slice():
  A, slices = set_up_tensor(N, M, T)

  for t in range(T):
    assert (A.get_frontal_slice(t) - slices[t]).nnz == 0

def test_working_get_frontal_slice():
  A, slices = set_up_tensor(N, M, T)

  new_X = sp.random(N,M)
  randT = randint(0,T-1)
  A.set_frontal_slice(randT,new_X)
  assert (A.get_frontal_slice(randT) - new_X).nnz == 0

def test_get_frontal_slice_errors_and_warnings():
  A, slices = set_up_tensor(N, M, T)

  #non-sparse matrix errors
  with pytest.raises(TypeError):
    A.set_frontal_slice(0,'apple')
    A.set_frontal_slice(0,4)

  #wrong shape
  with pytest.raises(ValueError):
    A.set_frontal_slice(0,sp.random(N+1,M+1))

  #warn about wrong type
  with pytest.warns(UserWarning):
    A.set_frontal_slice(0,sp.random(N,M,format='lil'))

def test_expanding_tensor():
  k = 4
  A, slices = set_up_tensor(N, M, T)

  new_slice = sp.random(N,M)
  A.set_frontal_slice(T+k,new_slice)

  assert A.shape == (N,M,T+k)
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
  A, slices = set_up_tensor(N, M, T)
  A.transpose(inPlace=True)

  assert A.shape == (M,N,T)

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
  A, slices = set_up_tensor(N, M, T)

  with pytest.warns(RuntimeWarning):
    A.get_scalar(0,0,0)

def test_get_scalar_errors():
  A, slices = set_up_tensor(N, M, T)

  with pytest.raises(ValueError):
    A.get_scalar(2 * N, 2 * M, 2 * T)
    A.get_scalar(-2* M, -2 * M, -2 * T)

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
  A, slices = set_up_tensor(N, M, T)

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
  X = sp.random(N,M,format='dok',density=.4)
  T = Tensor()
  tensor_X = T.squeeze(X)

  assert tensor_X.shape == (N,1,M)

  for i in range(M):
    assert (tensor_X.get_frontal_slice(i) - X[:,i]).nnz == 0

def test_squeeze_in_place():
  A, slices = set_up_tensor(N,M,T,'csr')

  A.squeeze()

  assert A.shape == (N,T,M)
  assert A._slice_format == 'dok'

  for t in range(M):
    new_frontal_slice = A._slices[t]
    for j in range(T):
      assert (new_frontal_slice[:,j] - slices[j][:,t]).nnz == 0

def test_squeeze_warnings():
  A, slices = set_up_tensor(N,M,T)
  with pytest.warns(RuntimeWarning):
    A.squeeze()



def test_squeeze_errors():
  T = Tensor()

  with pytest.raises(TypeError):
    T.squeeze(['blah',3])
    T.squeeze(4)
    T.squeeze('test')


'''-----------------------------------------------------------------------------
                                t product tests
-----------------------------------------------------------------------------'''
def build_block_circulant_matrix(tensor, transpose = False):
  (N,M,T) = tensor.shape
  if transpose:
    block_circ_matrix = sp.random(M*T,N*T,density=0,format='dok')
  else:
    block_circ_matrix = sp.random(N*T,M*T,density=0,format='dok')

  #populate the matrix
  for i in range(T):
    for j in range(T):
      if transpose:
        block_circ_matrix[i * M:(i + 1) * M, j * N:(j + 1) * N] = \
          (tensor._slices[(j + (T - i)) % T]).T
      else:
        block_circ_matrix[i*N:(i+1)*N, j*M:(j+1)*M] = \
          tensor._slices[(i + (T - j))%T]

  return block_circ_matrix

def test_t_product():
  A, slices = set_up_tensor(N,M,T,'dok')
  bcm = build_block_circulant_matrix(A)
  bcm_T = build_block_circulant_matrix(A,transpose=True)

  X = sp.random(M,T,density=.5,format='dok')
  X2 = sp.random(N, T, density=.5, format='dok')

  flattened_x = np.empty((M * T,1))
  flattened_x2 = np.empty((N * T, 1))

  for t in range(T):
    flattened_x[M*t:M*(t+1)] = X[:,t].todense()
    flattened_x2[N*t:N*(t+1)] = X2[:,t].todense()

  t_prod_x = bcm * flattened_x
  t_prod_transpose_x = bcm_T * flattened_x2

  B = A.t_product(A.squeeze(X))
  C = A.t_product(A.squeeze(X2),transpose=True)

  #check each slice
  for t in range(T):
    print t_prod_x.shape
    assert np.allclose(t_prod_x[N*t:N*(t+1)],
                      B._slices[t].todense().reshape(N,1),atol=1e-12)
    assert np.allclose(t_prod_transpose_x[M*t:M*(t+1)],
                      C._slices[t].todense().reshape(M,1),atol=1e-12)

def test_t_product_errors():
  A, slices = set_up_tensor(N,M,T,'dok')
  B = Tensor()

  #check invalid shape errors
  with pytest.raises(ValueError):
    A.t_product(B)
    A.t_product(B,transpose=True)

  #check invalid type errors
  with pytest.raises(TypeError):
    A.t_product(5)
    A.t_product('test')
    A.t_product([1,23,4,'apple'])




'''-----------------------------------------------------------------------------
                              scale tensor tests
-----------------------------------------------------------------------------'''
def test_scale_tensor():
  A,slices = set_up_tensor(N,M,T)

  scalar = uniform(0,1)
  B = A.scale_tensor(scalar)
  A.scale_tensor(scalar, inPlace=True)

  for t in range(T):
    assert (A._slices[t] - scalar* slices[t]).nnz == 0
    assert (B._slices[t] - scalar * slices[t]).nnz == 0


def test_scale_tensor_errors():
  A,_ = set_up_tensor(N,M,T)

  with pytest.raises(TypeError):
    A.scale_tensor([1,2,3])
    A.scale_tensor('apples',inPlace=True)

'''-----------------------------------------------------------------------------
                              zero tensor tests
-----------------------------------------------------------------------------'''
def test_zeros():
  Z1 = Te.zeros((N, M, T))
  Z2 = Te.zeros([N, M, T])
  Z3 = Te.zeros([N, M, T], format = 'dok')
  Z4 = Te.zeros((N, M, T), format='lil')

  for t in range(T):
    assert Z1.get_frontal_slice(t).nnz == 0
    assert Z2.get_frontal_slice(t).nnz == 0
    assert Z3.get_frontal_slice(t).nnz == 0
    assert Z4.get_frontal_slice(t).nnz == 0

def test_zeros_errors():

  #check for invalid tuple and list sizes
  with pytest.raises(ValueError):
    Te.zeros([N,M])
    Te.zeros((N,M))
    Te.zeros([N, M,T,5])
    Te.zeros((N, M,T,5))

  #check for invalid format
  with pytest.raises(AttributeError):
    Te.zeros([2,3,4],format='apple')

  #check for invalid shape types
  with pytest.raises(TypeError):
    Te.zeros([1,2,'apple'])
    Te.zeros(['apple',2,3])
    Te.zeros([1, 'apple',3])