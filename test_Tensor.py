import scipy.sparse as sp
import numpy as np
from numpy.linalg import norm as np_norm
from Tensor import Tensor
import Tensor as Te
import pytest
from tempfile import NamedTemporaryFile
from random import randint, uniform

#GLOBAL TEST VARIABLES
N = 6
M = 7
T = 5
ERROR_TOL = 1e-12

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

def test_convert_slices_error():
  A,_ = set_up_tensor(N,M,T,dense=True)
  with pytest.raises(AttributeError):
    A.convert_slices('dok')

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
def test_save_load():
  A, slices = set_up_tensor(N, M, T,dense=True)
  C = Tensor()

  with NamedTemporaryFile() as tmp_file:
    A.save(tmp_file.name)

    B = A.load(tmp_file.name,make_new=True)
    C.load(tmp_file.name)

    assert C.shape[0] == N
    assert C.shape[1] == M
    assert C.shape[2] == T
    assert C._slice_format == "dense"

    assert B.shape[0] == N
    assert B.shape[1] == M
    assert B.shape[2] == T
    assert B._slice_format == "dense"

  assert (B._slices == slices).all()
  assert (C._slices == slices).all()



'''-----------------------------------------------------------------------------
                              transpose tests
-----------------------------------------------------------------------------'''

def test_sparse_transpose():
  A, slices = set_up_tensor(N, M, T)
  B = A.transpose()
  A.transpose(inPlace=True)


  assert A.shape == (M,N,T)

  for t in range(T):
    if t == 0:
      assert (A._slices[t] - slices[0].T).nnz == 0
      assert (B._slices[t] - slices[0].T).nnz == 0
    else:
      assert (A._slices[t] - slices[:0:-1][t-1].T).nnz == 0
      assert (B._slices[t] - slices[:0:-1][t - 1].T).nnz == 0

def test_dense_non_square_frontal_slices_transpose():
  A, slices = set_up_tensor(N,M,T,dense=True)
  B = A.transpose()
  A.transpose(inPlace=True)
  for t in range(T):
    if t == 0:
      assert (A._slices[:,:,t] == slices[:,:,0].T).all()
      assert (B._slices[:,:,t] == slices[:,:,0].T).all()
    else:
      assert (A._slices[:,:,t] == slices[:,:,t-1].T).all()
      assert (B._slices[:,:,t] == slices[:,:,t - 1].T).all()



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
        assert A.get_scalar(t,j,i) == slices[t][i,j]

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

  A.set_scalar(rand_t,rand_j,rand_i,val)
  assert A.get_scalar(rand_t,rand_j,rand_i) == val

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
  Y = sp.random(N,M, density=.4)
  Z = np.random.rand(N,M)

  T = Tensor()
  tensor_X = T.squeeze(X)
  tensor_Y = T.squeeze(Y)
  tensor_Z = T.squeeze(Z)

  assert tensor_X.shape == (N, 1, M)
  assert tensor_Y.shape == (N, 1, M)
  assert tensor_Z.shape == (N, 1, M)

  #convert coo matrix to dok for access to elements
  Y = Y.todok()

  for i in range(M):
    assert (tensor_X.get_frontal_slice(i) - X[:,i]).nnz == 0
    assert (tensor_Y.get_frontal_slice(i) - Y[:,i]).nnz == 0
    assert (tensor_Z._slices[:,0,i] == Z[:,i]).all()


def test_squeeze_in_place():
  A, slices = set_up_tensor(N,M,T,'csr')
  B, slices2 = set_up_tensor(N,M,T,'dok')
  C, slices3 = set_up_tensor(N,M,T,dense=True)

  A.squeeze()
  B.squeeze()
  C.squeeze()

  assert A.shape == (N,T,M)
  assert A._slice_format == 'dok'
  assert B.shape == (N,T,M)
  assert B._slice_format == 'dok'
  assert C.shape == (N,T,M)


  for t in range(M):
    new_frontal_slice =  A._slices[t]
    new_frontal_slice2 = B._slices[t]
    for j in range(T):
      assert (new_frontal_slice[:,j] -  slices[j][:,t] ).nnz == 0
      assert (new_frontal_slice2[:,j] - slices2[j][:,t]).nnz == 0
      assert all(C._slices[:,j,t] == slices3[:,t,j])



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
                              twist tests
-----------------------------------------------------------------------------'''
def test_twist():
  matrix = sp.random(N,M,density=.5,format='dok')
  matrix2 = sp.random(N,M,density=0,format='dok')
  matrix3 = sp.random(N,M,density=.5,format='lil')

  slices = []
  slices2 = []
  slices3 = []
  for j in xrange(M):
    slices.append(matrix[:,j])
    slices3.append(matrix3[:,j])
    slices2.append(sp.random(N,1,density=.5,format='coo'))
    matrix2[:,j] = slices2[-1]

  A = Tensor(slices)
  A2 = Tensor(slices2)
  A3 = Tensor(slices3)

  B = A.twist(A)
  B2 = A.twist(A2)
  B3 = A.twist(A3)

  assert  B.shape[0] == N
  assert  B.shape[1] == M
  assert (B - matrix).nnz == 0

  assert  B2.shape[0] == N
  assert  B2.shape[1] == M
  assert (B2 - matrix2).nnz == 0

  assert  B3.shape[0] == N
  assert  B3.shape[1] == M
  assert (B3 - matrix3).nnz == 0

  assert (matrix - A.twist(A.squeeze(matrix))).nnz == 0
  assert (matrix2 - A.twist(A.squeeze(matrix2))).nnz == 0
  assert (matrix3 - A.twist(A.squeeze(matrix3))).nnz == 0

  A.twist()
  A2.twist()
  A3.twist()
  assert A._slice_format == 'dok'
  assert A2._slice_format == 'coo'
  assert A3._slice_format == 'lil'

def test_twist_errors():
  A, slices = set_up_tensor(N,M,T)

  with pytest.raises(TypeError):
    A.twist("apple")
    A.twist(2)
    A.twist([1,2,3,'test'])

  with pytest.raises(ValueError):
    A.twist(A)






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
                      B._slices[t].todense().reshape(N,1),atol=ERROR_TOL)
    assert np.allclose(t_prod_transpose_x[M*t:M*(t+1)],
                      C._slices[t].todense().reshape(M,1),atol=ERROR_TOL)

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
                            frobenius norm tests
-----------------------------------------------------------------------------'''
def test_frobenius_norm():
  B, dense_slices = set_up_tensor(N,M,T,dense=True)

  sparse_slices = []
  for t in range(T):
    sparse_slices.append(sp.coo_matrix(dense_slices[:,:,t]))

  A = Tensor(sparse_slices)
  assert abs(A.frobenius_norm() - np_norm(dense_slices.reshape(N*M*T))) \
         < ERROR_TOL
  assert abs(B.frobenius_norm() - np_norm(dense_slices.reshape(N * M * T))) \
         < ERROR_TOL


'''-----------------------------------------------------------------------------
                              norm tests
-----------------------------------------------------------------------------'''

'''-----------------------------------------------------------------------------
                              __add__/__sub__ tests
-----------------------------------------------------------------------------'''
def test__add__and__sub__():
  A, slices1 = set_up_tensor(N, M, T)
  B, slices2 = set_up_tensor(N, M, T)

  summed_slices = []
  subtracted_slices = []
  for t in range(T):
    summed_slices.append(slices1[t] + slices2[t])
    subtracted_slices.append(slices1[t] - slices2[t])

  C = A + B
  D = A - B
  for t in range(T):
    assert (C._slices[t] - summed_slices[t]).nnz == 0
    print D._slices[t]
    print "gap"
    print subtracted_slices[t]
    assert (D._slices[t] - subtracted_slices[t]).nnz == 0


def test__add__and__sub__errors():
  A, slices1 = set_up_tensor(N, M, T)
  B, slices2 = set_up_tensor(N+1, M+1, T+1)

  with pytest.raises(ValueError):
    A + B
    A - B

  with pytest.raises(TypeError):
    A + "sandwich"
    B + slices
    A - "apple"
    B - slices

'''-----------------------------------------------------------------------------
                              __neg__ tests
-----------------------------------------------------------------------------'''

def test__neg__():
  A, slices = set_up_tensor(N,M,T)
  B = -A

  for t in range(T):
    assert(B._slices[t] + slices[t]).nnz == 0

'''-----------------------------------------------------------------------------
                              __mul__ tests
-----------------------------------------------------------------------------'''
#note that this function defers to scale_tensor and t_product thus only
# errors are tested here
def test__mul__errors():
  A, _ = set_up_tensor(N, M, T)

  with pytest.raises(TypeError):
    A * "test"

'''-----------------------------------------------------------------------------
                              find max tests
-----------------------------------------------------------------------------'''
def test_find_max():
  A, _ = set_up_tensor(N,M,T,format='dok')
  B, _ = set_up_tensor(N, M, T,format='csr')
  C, _ = set_up_tensor(N, M, T, format='lil')
  D, _ = set_up_tensor(N, M, T, dense=True)

  A.set_scalar(0,0,0,2)
  B.set_scalar(0,0,0,2)
  C.set_scalar(0,0,0,2)


  assert A.find_max() == 2
  assert B.find_max() == 2
  assert C.find_max() == 2

'''-----------------------------------------------------------------------------
                              scale tensor tests
-----------------------------------------------------------------------------'''
def test_scale_tensor():
  A,slices = set_up_tensor(N,M,T)
  C,slices2 = set_up_tensor(N,M,T,dense=True)

  scalar = uniform(0,1)
  B = A.scale_tensor(scalar)
  A.scale_tensor(scalar, inPlace=True)
  D = C.scale_tensor(scalar)
  C.scale_tensor(scalar, inPlace=True)


  for t in range(T):
    assert (A._slices[t] - scalar* slices[t]).nnz == 0
    assert (B._slices[t] - scalar * slices[t]).nnz == 0

  for i in range(N):
    for j in range(M):
      for t in range(T):
        assert C._slices[i, j, t] == slices2[i, j, t]
        assert D._slices[i, j, t] == slices2[i, j, t]


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