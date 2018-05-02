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
ERROR_TOL = 1e-15

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

  with pytest.raises(ValueError):
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

  #test 3rd order tensor
  assert A.shape[0] == N
  assert A.shape[1] == M
  assert A.shape[2] == T
  assert A._slice_format == "dense"
  assert (A._slices == slices).all()

  #test transverse slice
  A = Tensor(slices[0,:,:],set_lateral=False)
  assert (A._slices == slices[0,:,:]).all()
  assert A.shape[0] == 1
  assert A.shape[1] == M
  assert A.shape[2] == T
  assert A._slice_format == "dense"

  #test lateral slice
  A = Tensor(slices[:, 0, :])
  assert (A._slices[:,0,:] == slices[:, 0, :]).all()
  assert A.shape[0] == N
  assert A.shape[1] == 1
  assert A.shape[2] == T
  assert A._slice_format == "dense"

  #test tubal scalars
  A = Tensor(slices[:, 0, 0])
  assert (A._slices == slices[:, 0, 0]).all()
  assert A.shape[0] == 1
  assert A.shape[1] == 1
  assert A.shape[2] == N
  assert A._slice_format == "dense"

  A = Tensor(slices[0, :, 0])
  assert (A._slices == slices[0, :, 0]).all()
  assert A.shape[0] == 1
  assert A.shape[1] == 1
  assert A.shape[2] == M
  assert A._slice_format == "dense"

  A = Tensor(slices[0, 0, :])
  assert (A._slices == slices[0, 0, :]).all()
  assert A.shape[0] == 1
  assert A.shape[1] == 1
  assert A.shape[2] == T
  assert A._slice_format == "dense"


'''-----------------------------------------------------------------------------
                             convert_slices test
-----------------------------------------------------------------------------'''
def test_convert_slices_dense_to_sparse():
  A,slices = set_up_tensor(N,M,T,dense=True)
  B = Tensor(slices)
  formats = ['dok','coo','dia','lil','csc','csr','bsr']
  for format in formats:
    A.convert_slices(format)
    assert A._slice_format == format
    for t in range(T):
      assert A._slices[t].format == format
    assert A == B
    #convert back for reformatting
    A = Tensor(slices)

def test_convert_slices_sparse_to_dense():

  formats = ['dok','coo','dia','lil','csc','csr','bsr']
  for format in formats:
    A, slices = set_up_tensor(N, M, T,format=format)
    A.convert_slices('dense')
    assert A._slice_format == 'dense'
    assert isinstance(A._slices,np.ndarray)
    B = Tensor(slices)
    assert A == B

'''-----------------------------------------------------------------------------
                              __get_item__ tests
-----------------------------------------------------------------------------'''
def test__get_item__scalar():
  A, dense_slices = set_up_tensor(N,M,T,dense=True)
  B, sparse_slices = set_up_tensor(N, M, T,format='dok')

  i = randint(0,N-1)
  j = randint(0,M-1)
  k = randint(0,T-1)

  assert A[i,j,k] == dense_slices[i,j,k]
  assert B[i,j,k] == sparse_slices[k][i,j]

def test__get_item__tubal_scalar():
  A, dense_slices = set_up_tensor(N,M,T,dense=True)
  B, sparse_slices = set_up_tensor(N, M, T,format='dok')

  i = randint(0, N-1)
  j = randint(0, M-1)

  assert (A[i,j]._slices[0,0] == dense_slices[i,j,:]).all()
  tubal_scalar = B[i,j]._slices
  for t in xrange(T):
    assert tubal_scalar[0,0,t] == sparse_slices[t][i,j]


def test__get_item__slice():
  A, dense_slices = set_up_tensor(N,M,T,dense=True)
  B, sparse_slices = set_up_tensor(N, M, T,format='dok')

  i = randint(0,N-1)
  j = randint(0,M-1)
  k = randint(0,T-1)

  assert (A[i: ,j ,k:]._slices[: ,0 ,:] == dense_slices[i:,j,k:]).all()
  assert (A[i, j:, k:]._slices[0, :, :] == dense_slices[i, j:, k:]).all()
  assert (A[i:, j:, k] == dense_slices[i:, j:, k]).all()

  assert all(map(lambda (x,y): (x != y[i:,j]).nnz == 0, \
            zip(B[i:,j,k:]._slices,sparse_slices[k:])))
  assert all(map(lambda (x,y): (x != y[i,j:]).nnz == 0, \
            zip(B[i,j:,k:]._slices,sparse_slices[k:])))
  assert (B[i:,j:,k] != sparse_slices[k][i:,j:]).nnz == 0

def test__get_item__subtensor():
  A, dense_slices = set_up_tensor(N,M,T,dense=True)
  B, sparse_slices = set_up_tensor(N, M, T,format='dok')

  i = randint(0,N-1)
  j = randint(0,M-1)
  k = randint(0,T-1)

  assert A[i:,j:,k:] == Tensor(dense_slices[i:,j:,k:])
  assert B[i:,j:,k:] == Tensor(map(lambda x: x[i:,j:],sparse_slices[k:]))

def test__get_item_errors():
  # wrong key count tests
  A,_ = set_up_tensor(N,M,T, dense=True)
  with pytest.raises(ValueError):
    A[1,2,3,4,5]
  A,_ = set_up_tensor(N, M, T,format='dok')
  with pytest.raises(ValueError):
    A[1, 2, 3, 4, 5]

  #wrong sparse matrix format
  A,_ = set_up_tensor(N,M,T)
  with pytest.raises(TypeError):
    A[1,2,3,4,5]
  A,_ = set_up_tensor(N,M,T,format='dia')
  with pytest.raises(TypeError):
    A[1,2,3,4,5]
  A, _ = set_up_tensor(N, M, T,format='bsr')
  with pytest.raises(TypeError):
    A[1, 2, 3, 4, 5]


'''-----------------------------------------------------------------------------
                              __set_item__ tests
-----------------------------------------------------------------------------'''
def test__set_item__scalar():
  #dense tensor
  A, _ = set_up_tensor(N,M,T,dense=True)
  B, _ = set_up_tensor(N,M,T,format='dok')

  N2 = randint(0,N-1)
  M2 = randint(0,M-1)
  T2 = randint(0,T-1)

  val = 10
  A[N2,M2,T2] = val

  B[N2, M2, T2] = val

  assert A[N2,M2,T2] == val
  assert B[N2,M2,T2] == val

def test__set_item__tubal_scalar():
  A, _ = set_up_tensor(N,M,T,dense=True)
  B, _ = set_up_tensor(N,M,T,format='dok')

  i = randint(0,N-1)
  j = randint(0,M-1)

  #test list assignment
  tubal_scalar = range(T)

  A[i,j] = tubal_scalar
  B[i,j] = tubal_scalar
  tubal_scalar = np.array(tubal_scalar)
  assert (A[i,j]._slices[0,0,:] == tubal_scalar).all()
  assert (B[i,j]._slices[0,0,:] == tubal_scalar).all()

  #reset values to zeros
  A[i,j] = [0] *T
  B[i,j] = [0] *T

  #test np.array assignment

  A[i, j] = tubal_scalar
  B[i, j] = tubal_scalar
  tubal_scalar = np.array(tubal_scalar)
  assert (A[i, j]._slices[0, 0, :] == tubal_scalar).all()
  assert (B[i, j]._slices[0, 0, :] == tubal_scalar).all()


def test__set_item__dense_tensor_slice():
  #dense tensor dense slice
  A, _ = set_up_tensor(N, M, T, dense=True)

  N2_start = randint(0, N-1)
  M2_start = randint(0, M-1)
  T2_start = randint(0, T-1)

  dense_lateral_slice = np.random.rand(N - N2_start,T - T2_start)
  dense_frontal_slice = np.random.rand(N - N2_start,M - M2_start)
  dense_transverse_slice = np.random.rand(M - M2_start,T - T2_start)

  A[N2_start:,M2_start,T2_start:] = dense_lateral_slice
  assert (A[N2_start:, M2_start, T2_start:]._slices[:, 0, :] ==
          dense_lateral_slice).all()

  A[N2_start:,M2_start:,T2_start] = dense_frontal_slice
  assert (A[N2_start:,M2_start:,T2_start] == dense_frontal_slice).all()

  A[N2_start,M2_start:,T2_start:] = dense_transverse_slice
  assert (A[N2_start,M2_start:,T2_start:]._slices[0,:,:] ==
          dense_transverse_slice).all()

  sparse_lateral_slice = sp.random(N - N2_start,T - T2_start,format='dok')

  A[N2_start:,M2_start,T2_start:] = sparse_lateral_slice
  print A[N2_start:, M2_start, T2_start:]._slices[:, 0, :] == sparse_lateral_slice
  assert (A[N2_start:, M2_start, T2_start:]._slices[:, 0, :] ==
          sparse_lateral_slice).all()

def test__set_item__sparse_tensor_slice():
  #dense tensor dense slice
  A, _ = set_up_tensor(N, M, T,format='dok')

  N2_start = randint(0, N-1)
  M2_start = randint(0, M-1)
  T2_start = randint(0, T-1)

  #insert dense slices

  dense_lateral_slice = np.random.rand(N - N2_start,T - T2_start)
  dense_frontal_slice = np.random.rand(N - N2_start,M - M2_start)
  dense_transverse_slice = np.random.rand(M - M2_start,T - T2_start)

  A[N2_start,M2_start:,T2_start:] = dense_transverse_slice
  slices = A[N2_start, M2_start:, T2_start:]._slices
  for (t, slice) in enumerate(slices):
    for ((_,j), v) in slice.iteritems():
      assert dense_transverse_slice[j,t] == v


  A[N2_start:,M2_start,T2_start:] = dense_lateral_slice

  for (t,slice) in enumerate(A[N2_start:, M2_start, T2_start:]._slices):
    for ((i,_),v) in slice.iteritems():
      assert dense_lateral_slice[i,t] == v


  A[N2_start:,M2_start:,T2_start] = dense_frontal_slice
  assert (A[N2_start:,M2_start:,T2_start] == dense_frontal_slice).all()

  #insert sparse slices

  sparse_lateral_slice = sp.random(N - N2_start,T - T2_start,density=.5)
  sparse_frontal_slice = sp.random(N - N2_start,M - M2_start,density=.5)
  sparse_transverse_slice = sp.random(M - M2_start,T - T2_start,density=.5)

  A[N2_start, M2_start:, T2_start:] = sparse_transverse_slice
  slices = A[N2_start, M2_start:, T2_start:]._slices
  sparse_transverse_slice = sparse_transverse_slice.todok()
  for (t, slice) in enumerate(slices):
    assert (slice != sparse_transverse_slice[:, t].T).nnz == 0

  A[N2_start:, M2_start, T2_start:] = sparse_lateral_slice
  sparse_lateral_slice = sparse_lateral_slice.todok()
  slices = A[N2_start:,M2_start,T2_start:]._slices
  for i,slice_t in enumerate(slices):
    assert (slice_t != sparse_lateral_slice[:,i]).nnz == 0

  A[N2_start:, M2_start:, T2_start] = sparse_frontal_slice
  assert (A[N2_start:, M2_start:, T2_start] != sparse_frontal_slice).nnz == 0


def test__set_item_general_errors():
  #matrix format errors
  A, _ = set_up_tensor(N,M,T)
  with pytest.raises(TypeError):
    A[0,0,0] = 2
  A, _ = set_up_tensor(N, M, T,format='dia')
  with pytest.raises(TypeError):
    A[0, 0, 0] = 2
  A, _ = set_up_tensor(N, M, T,format='bsr')
  with pytest.raises(TypeError):
    A[0, 0, 0] = 2

  #wrong setting type
  A, _ = set_up_tensor(N, M, T,format='dok')
  with pytest.raises(TypeError):
    A[0, 0, 0] = 'apple'

  #scalars
  with pytest.raises(ValueError):
    A[0,0] = 3
    A[0] = 3

def test__set_item_dense_errors():
  A,_ = set_up_tensor(N,M,T,format='dok')


  #tubal scalars
  ts = range(T)
  ts_array = np.array(ts)

  with pytest.raises(ValueError):
    # wrong indices
    A[0] = ts
    A[0] = ts_array
    A[:,0] = ts
    A[:, 0] = ts_array
    A[0, :] = ts
    A[0, :] = ts_array
    A[0,0,0,0] = ts_array


  B,_ = set_up_tensor(N,M,2*T,format='dok')

  large_ts = range(2*T)
  large_ts_array = np.array(large_ts)
  with pytest.raises(ValueError):
    B[0,0] = ts
    B[0,0] = ts_array

    A[0,0] = large_ts
    A[0,0] = large_ts_array

  X = np.random.rand(M,T)

  with pytest.raises(ValueError):
    A[0,0,0,0] = X
    A[0,0,0] = X
    A[0,0,:] = X

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

def test_dense_transpose():
  A, slices = set_up_tensor(N,M,T,dense=True)
  B = A.transpose()
  A.transpose(inPlace=True)

  assert A.shape == (M,N,T)
  assert B.shape == (M,N,T)

  for t in range(T):
    assert (A._slices[:,:,t] == slices[:,:,-t%T].T).all()
    assert (B._slices[:,:,t] == slices[:,:,-t%T].T).all()

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
    assert (tensor_X[i] - X[:,i]).nnz == 0
    assert (tensor_Y[i] - Y[:,i]).nnz == 0
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
  matrix4 = np.random.rand(N,M)

  slices = []
  slices2 = []
  slices3 = []
  slices4 = np.ndarray((N, 1, M))

  for j in xrange(M):
    slices4[:,0,j] = matrix4[:,j]
    slices.append(matrix[:,j])
    slices3.append(matrix3[:,j])
    slices2.append(sp.random(N,1,density=.5,format='coo'))
    matrix2[:,j] = slices2[-1]

  A = Tensor(slices)
  A2 = Tensor(slices2)
  A3 = Tensor(slices3)
  A4 = Tensor(slices4)
  B = A.twist(A)
  B2 = A.twist(A2)
  B3 = A.twist(A3)
  B4 = A.twist(A4)

  assert  B.shape[0] == N
  assert  B.shape[1] == M
  assert (B - matrix).nnz == 0

  assert  B2.shape[0] == N
  assert  B2.shape[1] == M
  assert (B2 - matrix2).nnz == 0

  assert  B3.shape[0] == N
  assert  B3.shape[1] == M
  assert (B3 - matrix3).nnz == 0

  assert B4.shape[0] == N
  assert B4.shape[1] == M
  assert (B4 == matrix4).all()

  assert (matrix - A.twist(A.squeeze(matrix))).nnz == 0
  assert (matrix2 - A.twist(A.squeeze(matrix2))).nnz == 0
  assert (matrix3 - A.twist(A.squeeze(matrix3))).nnz == 0
  assert (matrix4 == A.twist(A.squeeze(matrix4))).all()

  A.twist()
  A2.twist()
  A3.twist()
  A4.twist()
  assert A._slice_format == 'dok'
  assert A2._slice_format == 'coo'
  assert A3._slice_format == 'lil'
  assert A4._slice_format == 'dense'

def test_twist_errors():
  A, slices = set_up_tensor(N,M,T)

  with pytest.raises(TypeError):
    A.twist("apple")
    A.twist(2)
    A.twist([1,2,3,'test'])

  with pytest.raises(ValueError):
    A.twist(A)



'''
    if transpose:
      block_circ_matrix = sp.random((M*T,N*T)
    else:
      block_circ_matrix = np.empty((N*T,M*T)))
    
    for i in xrange(T):
      for j in xrange(T):
        if transpose:
          block_circ_matrix[i * M:(i + 1) * M, j * N:(j + 1) * N] = \
            (tensor._slices[:,:,(j + (T - i)) % T]).T
        else:
          

'''


'''-----------------------------------------------------------------------------
                                t product tests
-----------------------------------------------------------------------------'''
def build_block_circulant_matrix(tensor, transpose = False):
  (N,M,T) = tensor.shape

  if tensor._slice_format == 'dense':
    if transpose:
      block_circ_matrix = np.empty((M * T, N * T))
    else:
      block_circ_matrix = np.empty((N * T, M * T))
  else:
    if transpose:
      block_circ_matrix = sp.random(M*T,N*T,density=0,format='dok')
    else:
      block_circ_matrix = sp.random(N*T,M*T,density=0,format='dok')

  #populate the matrix
  for i in range(T):
    for j in range(T):
      if tensor._slice_format == "dense":
        if transpose:
          block_circ_matrix[i * M:(i + 1) * M, j * N:(j + 1) * N] = \
            (tensor._slices[:,:,(j + (T - i)) % T]).T
        else:
          block_circ_matrix[i * N:(i + 1) * N, j * M:(j + 1) * M] = \
            tensor._slices[:,:,(i + (T - j)) % T]
      else:
        if transpose:
          block_circ_matrix[i * M:(i + 1) * M, j * N:(j + 1) * N] = \
            (tensor._slices[(j + (T - i)) % T]).T
        else:
          block_circ_matrix[i*N:(i+1)*N, j*M:(j+1)*M] = \
            tensor._slices[(i + (T - j))%T]

  return block_circ_matrix

def test_sparse_t_product():
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
    assert np.allclose(t_prod_x[N*t:N*(t+1)],
                      B._slices[t].todense().reshape(N,1),atol=ERROR_TOL)
    assert np.allclose(t_prod_transpose_x[M*t:M*(t+1)],
                      C._slices[t].todense().reshape(M,1),atol=ERROR_TOL)

def test_dense_t_prod():
  A, slices2 = set_up_tensor(N,M,T,dense=True)

  bcm = build_block_circulant_matrix(A)
  bcm_T = build_block_circulant_matrix(A,transpose=True)

  X = np.random.rand(M,T)
  X2 = np.random.rand(N, T)

  flattened_x = np.empty((M * T,1))
  flattened_x2 = np.empty((N * T, 1))

  for t in range(T):
    flattened_x[M*t:M*(t+1)] = X[:,t].reshape((M,1))
    flattened_x2[N*t:N*(t+1)] = X2[:,t].reshape((N,1))

  t_prod_x = np.dot(bcm,flattened_x)
  t_prod_transpose_x = np.dot(bcm_T,flattened_x2)

  B = A.t_product(A.squeeze(X))
  C = A.t_product(A.squeeze(X2),transpose=True)

  #check each slice
  for t in range(T):
    assert np.allclose(t_prod_x[N*t:N*(t+1)],
                      B._slices[:,:,t].reshape((N,1)),atol=ERROR_TOL)
    assert np.allclose(t_prod_transpose_x[M*t:M*(t+1)],
                      C._slices[:,:,t].reshape((M,1)),atol=ERROR_TOL)


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
  np_fro_norm = np_norm(dense_slices.reshape(N*M*T))

  assert (abs(A.frobenius_norm() - np_fro_norm)/np_fro_norm) < ERROR_TOL
  assert (abs(B.frobenius_norm() - np_fro_norm)/np_fro_norm) < ERROR_TOL


'''-----------------------------------------------------------------------------
                              norm tests
-----------------------------------------------------------------------------'''





'''-----------------------------------------------------------------------------
                              __add__/__sub__ tests
-----------------------------------------------------------------------------'''
def test__add__and__sub__sparse():
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
    assert (D._slices[t] - subtracted_slices[t]).nnz == 0

def test__add__and__sub__dense():
  A, slices1 = set_up_tensor(N, M, T,dense=True)
  B, slices2 = set_up_tensor(N, M, T,dense=True)

  summed_slices = slices1 + slices2
  subtracted_slices = slices1 - slices2

  C = A + B
  D = A - B

  assert C == Tensor(summed_slices)
  assert D == Tensor(subtracted_slices)

def test_add__and_sub__crossed():
  A, slices1 = set_up_tensor(N, M, T,dense=True)
  B, slices2 = set_up_tensor(N, M, T)



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

  A[0,0,0] = 2
  B[0,0,0] = 2
  C[0,0,0] = 2

  assert A.find_max() == 2
  assert B.find_max() == 2
  assert C.find_max() == 2

'''-----------------------------------------------------------------------------
                              is_equal_to_tensor tests
-----------------------------------------------------------------------------'''
def test_is_equal_to_tensor():
  A, slices1 = set_up_tensor(N, M, T, format='csr')
  B, slices2 = set_up_tensor(N, M, T, format='dok')
  C, slices3 = set_up_tensor(N, M, T, dense=True)

  assert A.is_equal_to_tensor(A)
  assert not A.is_equal_to_tensor(B)
  assert not A.is_equal_to_tensor(C)

  assert B.is_equal_to_tensor(B)
  assert not B.is_equal_to_tensor(C)

  assert C.is_equal_to_tensor(C)

  assert not A.is_equal_to_tensor([1,2,3])
  assert not A.is_equal_to_tensor(3)
  assert not A.is_equal_to_tensor('test')


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
                              normalize tests
-----------------------------------------------------------------------------'''
def test_dense_normalize():

  A, lateral_slice = set_up_tensor(N,M,T,dense=True)

  V,a = Te.normalize(A)

  for j in range(M):
    assert (V[:,j,:] * a[j,:,:]).is_equal_to_tensor(A[:,j,:],ERROR_TOL)


'''-----------------------------------------------------------------------------
                              zero tensor tests
-----------------------------------------------------------------------------'''
def test_zeros():
  #sparse case
  Z1 = Te.zeros((N, M, T))
  Z2 = Te.zeros([N, M, T])
  Z3 = Te.zeros([N, M, T], format = 'dok')
  Z4 = Te.zeros((N, M, T), format='lil')

  assert Z1.shape == (N,M,T)
  assert Z2.shape == (N,M,T)
  assert Z3.shape == (N,M,T)
  assert Z4.shape == (N,M,T)

  for t in range(T):
    assert Z1[t].nnz == 0
    assert Z2[t].nnz == 0
    assert Z3[t].nnz == 0
    assert Z4[t].nnz == 0

  #dense case
  Z5 = Te.zeros((N,M,T),format='dense')
  Z6 = Te.zeros([N,M,T],format='dense')

  assert Z5.shape == (N,M,T)
  assert Z6.shape == (N,M,T)

  for i in xrange(N):
    for j in xrange(M):
      for k in xrange(T):
        assert Z5[i,j,k] == 0
        assert Z6[i,j,k] == 0

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
    Te.zeros([1, 2, 'apple'])
    Te.zeros(['apple', 2, 3])
    Te.zeros([1, 'apple', 3])

'''-----------------------------------------------------------------------------
                              random tensor tests
-----------------------------------------------------------------------------'''
def test_random():
  #test dense case
  A = Te.random((N,M,T),format='dense',random_state=1)

  assert A.shape == (N,M,T)
  assert A._slice_format == 'dense'
  assert A.find_max() < 1
  assert (-A).find_max() <= 0

  #test sparse case
  density = .5
  A = Te.random([N,M,T],format='dok',random_state=1,density=density)

  assert A.shape == (N,M,T)
  assert A._slice_format == 'dok'
  assert A.find_max() < 1
  assert (-A).find_max() <= 0
  assert reduce(lambda x,y: x + y.nnz, A._slices,0) == density*N*M*T

def test_random_errors():

  with pytest.raises(ValueError):
    A = Te.random([1,2,3,4])
    A = Te.random((1,2,3,4))
    A = Te.random((1,2))
    A = Te.random([1,2])

  with pytest.raises(TypeError):
    A = Te.random('blah')

'''-----------------------------------------------------------------------------
                              empty tensor tests
-----------------------------------------------------------------------------'''
def test_empty():
  #dense case
  A = Te.empty((N,M,T))

  assert A.shape == (N,M,T)
  assert A._slice_format == 'dense'
  assert isinstance(A._slices,np.ndarray) #doesn't matter what the elements are

  A = Te.empty([N, M, T])

  assert A.shape == (N, M, T)
  assert A._slice_format == 'dense'
  assert isinstance(A._slices,
                    np.ndarray)  # doesn't matter what the elements are

  #sparse case
  A = Te.empty((N, M, T),sparse=True)

  assert A.shape == (N, M, T)
  assert A._slice_format == 'dok'
  for t in xrange(T):
    assert A[t].nnz == 0  #check for 0 matrix

  A = Te.empty([N, M, T],sparse=True)

  assert A.shape == (N, M, T)
  assert A._slice_format == 'dok'
  for t in xrange(T):
    assert A[t].nnz == 0  # doesn't matter what the elements are

def test_empty_errors():
  #check invalid shape types
  with pytest.raises(TypeError):
    Te.empty('apple')
    Te.empty(np.array([1,2,3]))

  #check invalid shape values
  with pytest.raises(ValueError):
    Te.empty((1,2,1))
    Te.empty((1,2))
    Te.empty((1,2,3,4))

'''-----------------------------------------------------------------------------
                              identity tensor tests
-----------------------------------------------------------------------------'''
def test_identity():
  #dense case
  A, _ = set_up_tensor(N,M,T,format='csr')
  I = Te.identity(N, T, format='dense')
  assert (I * A).is_equal_to_tensor(A,ERROR_TOL)
  I = Te.identity(M, T, format='dense')
  assert (A * I).is_equal_to_tensor(A,ERROR_TOL)

  #sparse case
  I = Te.identity(N, T)
  assert (I * A).is_equal_to_tensor(A,ERROR_TOL)
  I = Te.identity(M, T)
  assert (A * I).is_equal_to_tensor(A,ERROR_TOL)


if __name__ == '__main__':
  test_identity()