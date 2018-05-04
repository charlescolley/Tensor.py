#  Dependencies
#-----------------------------------------------------------------------------
import scipy.sparse as sp
import pickle
from copy import deepcopy
from itertools import izip
from scipy.sparse.linalg import norm as sp_norm
from scipy.fftpack import fft, ifft,rfft, fftshift, ifftshift, irfft
from math import sqrt, hypot
from numpy import ndarray, conj, NINF, array, dot
from numpy.random import seed, rand
from numpy import zeros as np_zeros
from numpy import empty as np_empty
from numpy.linalg import norm as np_norm
from warnings import warn
from numbers import Number


#import test_Tensor as T


'''-----------------------------------------------------------------------------
  Core Class
-----------------------------------------------------------------------------'''

class Tensor:
  '''
      This is a class of 3rd order sparse tensors which is designed to support
      the t-product.

      Tensor Class
        Private:
          _slices - (list of sparse matrices or 3rd order ndarray)
           a list of sparse scipy matrices which represents the frontal slices of
           the tensor, i.e. the t-th element of the list will be the matrix
           A[:,:,t]. All slices will be the same type of sparse matrix. If
           slices is passed in as an ndarray, then it must only have 3
           dimensions.
          _slice_format - (string)
           a string indicating the sparse matrix format of each of the frontal
           slices in the tensor. If _slices is an ndarray it will be set to
           'dense'
          shape - (tuple of ints)
           a tuple with the shape of the tensor. The ith element of shape
           corresponds to the dimension of the ith mode.
        Public Methods:
          save
          load
          convert_slices
          set_frontal_slice
          get_front_slice
          set_scalar
          get_scalar
          resize                SPARSE UNFINISHED
          transpose
          squeeze
          twist
          t-product
          scale_tensor
          find_max
          is_equal_to_tensor
          frobenius_norm
          norm                         UNTESTED
          cos_distance m
        Overloaded Methods:
          __add__
          __sub__
          __mul__
          __neg__
          __eq__
          __ne
          __getitem__
          __setitem__
      Class utilization
        zeros
        empty
        random
        identity
        normalize
        MGS


    TODO: -write a reshape function
          -write print overloading
          -add in non-zero count private element?
          -add in complex mode
  '''

  def __init__(self, slices = None, set_lateral = True):

    if slices is not None:

      if isinstance(slices,list) and isinstance(slices[0],Number):
        slices = array(slices)

      #if string passed in, assumed to be a file path
      if isinstance(slices,str):
        self.load(slices)
      elif isinstance(slices,ndarray):
        #check order of tensor
        if len(slices.shape) > 3:
          raise ValueError("ndarray must be at most order 3, slices passed in "
                           "has order {}\n".format(len(slices.shape)))
        else:
          self._set_ndarray_as_slices(slices,lateral=set_lateral)
      else:
        #check for valid slice array
        slice_shape = slices[0].shape
        slice_format = slices[0].getformat()
        for t,slice in enumerate(slices[1:],1):
          if slice.shape != slice_shape:
            raise ValueError("slices must all have the same shape, slice {} "
                             "has shape {}, but slice 0 has shape {}\n".
                             format(t,slice.shape,slice_shape))
          if slice.getformat() != slice_format:
            warn("slice format {} is different from first slice, "
                              "coverting to format {},\n this may make "
                              "initialization slow. pass in list of same type "
                              "sparse matrix for \nfaster "
                              "initialization\n".
                              format(slice.getformat(),slice_format),
                          RuntimeWarning)
            slices[t] = slice.asformat(slice_format)

        self._slices = slices
        self.shape = (slice_shape[0],slice_shape[1],len(slices))
        self._slice_format = slice_format
    else:
      self._slices = []
      self.shape = (0, 0, 0)
      self._slice_format = None

  def __mul__(self, other):
    if isinstance(other, Tensor):
      return self.t_product(other)
    elif isinstance(other, Number):
      return self.scale_tensor(other)
    else:
      raise TypeError("{} is not a subclass of Number, or a Tensor instance,\n "
                      "parameter is of type {}\n".format(other,type(other)))

  def _add_sub_helper(self,other, add):
    if isinstance(other, Tensor):
      # check dimensions
      if self.shape != other.shape:
        raise ValueError("invalid shape, input tensor must be of shape {}, \n"
                         "input tensor is of shape {}.\n".format(self.shape,
                                                                 other.shape))
      else:
        if add:
          if self._slice_format == 'dense':
            if other._slice_format == 'dense':
              return Tensor(self._slices + other._slices)
            else:
              new_slices = deepcopy(self._slices)
              if other._slice_format == 'dok':
                for t in xrange(self.shape[2]):
                  for ((i,j),v) in other._slices[t].iteritems():
                    new_slices[i,j,t] += v
              else:
                for t in xrange(self.shape[2]):
                  slice_t = other._slices[t]
                  if other._slice_format != 'coo':
                    slice_t = slice_t.tocoo()
                  for (i,j,v) in izip(slice_t.row,slice_t.col,slice_t.data):
                    new_slices[i,j,t] += v
              return Tensor(new_slices)
          else:
            if other._slice_format == 'dense':
              return other + self
            else:
              return Tensor(
                map(lambda (x, y): x + y, zip(self._slices, other._slices)))
        else:
          if self._slice_format == 'dense':
            if other._slice_format == 'dense':
              return Tensor(self._slices - other._slices)
            else:
              new_slices = deepcopy(self._slices)
              if other._slice_format == 'dok':
                for t in xrange(self.shape[2]):
                  for ((i,j),v) in other._slices[t].iteritems():
                    new_slices[i,j,t] -= v
              else:
                for t in xrange(self.shape[2]):
                  slice_t = other._slices[t]
                  if other._slice_format != 'coo':
                    slice_t = slice_t.tocoo()
                  for (i,j,v) in izip(slice_t.row,slice_t.col,slice_t.data):
                    new_slices[i,j,t] -= v
              return Tensor(new_slices)
          else:
            if other._slice_format == 'dense':
              return -other + self
            else:
              return Tensor(
                map(lambda (x, y): x - y, zip(self._slices, other._slices)))
    else:
      raise TypeError("input {} passed in is not an instance of a Tensor, "
                      "parameter passed in is of type {}".
                      format(other, type(other)))

  def __add__(self, other):
    return self._add_sub_helper(other,add =True)

  def __sub__(self,other):
    return self._add_sub_helper(other,add =False)

  def __neg__(self):
    return self.scale_tensor(-1)

  def __eq__(self, other):
    return self.is_equal_to_tensor(other)

  def __ne__(self,other):
    return not self.__eq__(other)

  def __getitem__(self, key):
    if self._slice_format == 'dense':
      if isinstance(key,slice):
        new_slices = self._slices[:, :, key]
      elif isinstance(key,int):
        return self._slices[:,:,key]
      elif len(key) == 2:
        if isinstance(key[0],int):
          return Tensor(self._slices[key[0],key[1],:],set_lateral=False)
        else:
          new_slices = self._slices[key[0],key[1],:]
      elif len(key) == 3:
        new_slices = self._slices[key]
        if isinstance(new_slices,Number) or isinstance(key[2],int):
          return new_slices
        elif isinstance(key[0],int):
          return Tensor(new_slices,set_lateral=False)
      else:
        raise(ValueError("invalid amount of indices/slices, {} found, must add "
                         "at most 3 indices and/or slices.".format(len(key))))
      return Tensor(new_slices)
    else:
      if isinstance(key,slice):
        return Tensor(self._slices[key])
      elif isinstance(key,int):
        return self._slices[key]
      else:
        if self._slice_format not in ['coo', 'dia', 'bsr']:
          if len(key) == 2:
            return Tensor(map(lambda x: x[key],self._slices))
          elif len(key) == 3:
            if isinstance(key[2],slice):
              return Tensor(map(lambda x: x[key[0],key[1]],self._slices[key[2]]))
            else:
              new_slices = self._slices[key[2]][key[0],key[1]]
              if isinstance(new_slices,Number):
                return new_slices
              else:
                if isinstance(key[2],int):
                  return new_slices
                else:
                  return Tensor(new_slices)
          else:
            raise ValueError("invalid amount of indices/slices, {} found, must "
                             "have at\n most 3 indices and/or slices.".\
                             format(len(key)))
        else:
          raise TypeError('{} format does not support slice assignment,\n use '
                          'self.convert_slices to reformat the tensor to dok or '
                          'lil format'.format(self._slice_format))

  def _set_ndarray_as_slices(self, slices,lateral = True, key = None):
    '''
      Helper function for setting the slices when an ndarray is passed in. A
      given shape must be passed in if the ndarray is 2-dimensional to assess
      whether it's a transverse or lateral slice.

      :Input:
        slices - (ndarray)
          the slices of the tensor to set. slices must have order at most 3.
        lateral - (optional tuple)
          a bool indicating whether to set a 2d array as a lateral slice o
          as a transverse slice. Only checked for 2d array case.
        key - (optional tuple)
          The indices or slices to index into the tensor and set the elements
          with length at most 3. If no key is passed in, it's assumed that
          the slices object must be created, and self.shape will be set.
    '''
    if len(slices.shape) == 1: #assumed to be a tubal scalar
      T = slices.shape[0]
      if key is not None:
        if isinstance(key,slice) or isinstance(key,int):
          raise ValueError("key must be at least length 2 to identify which "
                           "tubal scalar to set\n")
        else:
          if isinstance(key[0],slice) or isinstance(key[1],slice):
            raise ValueError("cannot set tubal scalar over 1st or 2nd modes, "
                             "slices can only be set over 3rd mode.\n")
          if len(key) == 3:
            if isinstance(key[2],int):
              slice_length = 1
            else:
              (start,stop,step) = key[2].indices(self.shape[2])
              slice_length = (stop - start)/step
          else:
            slice_length = self.shape[2]

          if slice_length != len(slices):
            raise ValueError('tubal scalar of length {} is incorrect size, '
                             'must be of length {}'.format(len(slices),
                                                           slice_length))
          if len(key) == 2:
            if self._slice_format == "dense":
              self._slices[key[0],key[1],:] = slices
            else:
              for t in xrange(T):
                self._slices[t][key] = slices[t]
          elif len(key) == 3:
            if self._slice_format == 'dense':
              self._slices[key] = slices
            else:
              for t,slice_t in enumerate(self._slices[key[2]]):
                slice_t[key[:2]] = slices[t]
          else:
            raise ValueError('cannot index or slice into more than 3 '
                             'modes.key is of length {}\n'.format(len(key)))
      else:
        self._slices = ndarray((1,1,T),buffer=slices.flatten())
        self.shape = (1,1,T)
        self._slice_format = 'dense'
    elif len(slices.shape) == 2:
      (N,T) = slices.shape
      if key is not None:
        if isinstance(key,int):
          if self._slice_format == 'dense':
            self._slices[:,:,key] = slices
          else:
            self._slices[key] = slices
        elif len(key) == 2:
          if self._slice_format == 'dense':
            self._slices[key[0],key[1],:] = slices
          else:
            if isinstance(key[0],int) and isinstance(key[1],slice):
              for t,t_slice in enumerate(self._slice):
                self._slices[key[:2]] = slices[t,:]
            elif isinstance(key[1],int) and isinstance(key[0],slice):
              for t, t_slice in enumerate(self._slice):
                self._slices[key[1]] = slices[:,t]
            else:
              raise ValueError("cannot assign matrix to a scalar or "
                               "subtensor.\n")
        elif len(key) == 3:
          if self._slice_format == 'dense':
            self._slices[key] = slices
          else:
            (N,M,T) = self.shape
            if isinstance(key[0],int) and isinstance(key[1],slice) and \
               isinstance(key[2],slice): #transverse
              for (i,t) in enumerate(xrange(*key[2].indices(T))):
                self._slices[t][key[0],key[1]] = slices[:,i]
            elif isinstance(key[0],slice) and isinstance(key[1],int) and \
               isinstance(key[2],slice): #lateral
               for (i,t) in enumerate(xrange(*key[2].indices(T))):
                 self._slices[t][key[0],key[1]] \
                   = slices[:,i].reshape(slices[:,i].shape[0],1)
            elif isinstance(key[0], slice) and isinstance(key[1], slice) and \
             isinstance(key[2], int):  # frontal
               self._slices[key[2]][key[0],key[1]] = slices
            else:
              raise ValueError("Cannot assign matrix to indices as passed in,"
                               "\n must have two slices and one integer to "
                               "assign a matrix.\n")
        else:
          raise ValueError("key must be at most length 3, key is length {"
                           "}\n".format(len(key)))
      else:
        if lateral:
          self._slices = ndarray((N,1,T),buffer=slices.flatten())
          self.shape = (N,1,T)
          self._slice_format = 'dense'
        else:
          self._slices = ndarray((1,N,T),buffer=slices.flatten())
          self.shape = (1,N,T)
          self._slice_format = 'dense'
    else:
      if key is not None:
        if isinstance(key,int) or isinstance(key,slice):
          if self._slice_format == 'dense':
            self._slices[:,:,key] = slices
          else:
            if isinstance(key[2],int):
              self._slices[key[2]][key[:2]] = slices
            else:
              for (i,t) in enumerate(xrange(*key[2].indices(self.shape[2]))):
                self._slices[t] = slices[:,:,i]
        elif len(key) == 2:
          if self._slice_format == 'dense':
            self._slices[key[0],key[1],:] = slices
          else:
            for t in xrange(self.shape[2]):
              self._slices[t][key] = slices[:,:,t]
        elif len(key) == 3:
          if self._slice_format == 'dense':
            self._slices[key] = slices
          else:
            for (i,t) in enumerate(xrange(*key[2].indices(self.shape[2]))):
              self._slices[t][key[:2]] = slices[:,:,i]
        else:
          raise ValueError("key must be at most length 3, key is length {"
                           "}\n".format(len(key)))
      else:
        self._slices = slices
        self.shape = slices.shape
        self._slice_format = 'dense'

  def _set_sparse_matrices_as_slices(self,slices,lateral = True, key=None):
    '''
      This is a helper function for parsing input and setting the non-zeros
      of a tensor when the value is either a sparse matrix or list of sparse
      matrices.

      :Input:
        slices - (list of (or) sparse matrix)
          the slices to set in the tensor. tubal scalars must be made with a
          np.array.
        lateral - (optional bool)
          a boolean indicating whether or not to set a matrix slice as a
          lateral slice, only used for sparse matrix case.
        key - (optional tuple)
          a tuple with indices or slices for setting the values of the
          subtensor. values are assumed to be correct, else the errors will
          be raised by the scipy classes.

    '''

    #tubal scalars must be set with dense arrays
    if sp.issparse(slices):
      if key is not None:
        (N,M,T) = self.shape
        if isinstance(key,int):
          if self.shape[0] == slices.shape[0]\
            and self.shape[1] ==  slices.shape[1]:
            if self._slice_format == 'dense':
              self._slices[:,:,key] = 0
              def assign(A,i,j,v):
                A[i,j,key] = v
            else:
              self._slices[key] = sp.random(N,M,density=0,
                                            format=self._slice_format)
              def assign(A,i,j,v):
                A[key][i,j] = v
          else:
            raise ValueError("cannot assign a matrix of shape {} to tensor "
                             "frontal slice of shape {}"\
                             .format(slices.shape,self.shape[:2]))
        elif isinstance(key,slice):
          raise ValueError("cannot assign a matrix to a single slice. Keys "
                          "of length 1 are assumed to refer to frontal "
                          "slices.\n")
        elif len(key) == 2:
          (start,stop,step) = (0,0,0) #initialize for existance after if statement
          if isinstance(key[0], int) and isinstance(key[1], slice): #transverse

            (start,stop,step) = key[1].indices(M)            #zero out elements

            if self._slice_format == 'dense':
              def assign(A, i, j, v):
                A[key[0],start + i*step,j] = v
            else:
              def assign(A, i, j, v):
                A[j][key[0],start + i*step] = v
          elif isinstance(key[1], int) and isinstance(key[0],slice):# lateral
            (start,stop,step) = key[0].indices(N)
            if self._slice_format == 'dense':
              def assign(A, i, j, v):
                A[start + i*step,key[1],j] = v
            else:
              def assign(A,i,j,v):
                A[j][start + i*step,key[1]] = v

          else:
            raise ValueError("cannot assign matrix to key of type {}. "
                             "mut be either (slice,int) or (int,slice) "
                             "to assign lateral or transverse slice "
                             "respectively.\n".format(map(type, key)))
          #zero out the elements
          if self._slice_format == 'dense':
            self._slices[key[0], key[1], :] = 0
          else:
            for t in xrange(T):
              if self._slice_format == 'dok':
                for (i,j) in self._slices[t][key].iterkeys():
                  assign(self._slices[t],i,j,0)
              else:
                slice_t = self._slices[t][key].tocoo()
                for (i,j) in izip(slice_t.row,slice_t.col):
                  assign(self._slice[t],i,j,0)
        elif len(key) == 3:
          if isinstance(key[0], int) and isinstance(key[1], slice)\
              and isinstance(key[2],slice):
            self._matrix_set_helper(key,slices,mode='transverse')
          elif isinstance(key[0], slice) and isinstance(key[1], int)\
              and isinstance(key[2],slice):
            self._matrix_set_helper(key,slices,mode='lateral')
          elif isinstance(key[0], slice) and isinstance(key[1], slice)\
              and isinstance(key[2],int):
            self._matrix_set_helper(key,slices,mode='frontal')
          else:
            raise ValueError("key must contain two slices and 1 index to "
                             "properly assign a matrix to a subsection of the "
                             "tensor.\n")
        else:
          raise ValueError("key must have at most 3 indices and/or "
                           "slices, {} passed in".format(len(key)))
      else:
        (N,T) = slices.shape
        new_slices = []
        if lateral:
          for t in xrange(N):
            new_slices.append(slices[:,t])
            self.shape = (N, 1, T)
        else:
          for t in xrange(T):
            new_slices.append(slices[t,:])
          self.shape = (1, N, T)
        self._slices = new_slices
        self._slice_format = slices.format
    else:
      if key is not None:
        T = self.shape[2]
        if isinstance(key,int):
          self._matrix_set_helper((slice(None),slice(None),key),slices[0])
        elif isinstance(key,slice):
          for (i,t) in enumerate(xrange(*key.indices(T))):
            self._matrix_set_helper((slice(None),slice(None),t),slices[i])
        elif len(key) == 2:
          if isinstance(key[0],int) and isinstance(key[1],int):
            raise ValueError("inefficient to set tubal scalars with list of "
                             "sparse matrices,\n set with array or list of "
                             "elements.\n")
          else:
            if len(slices) >= T:
              for t in xrange(T):
                self._matrix_set_helper((key[0],key[1],t),slices[t])
            else:
              raise ValueError("not enough frontal slices to set elements of "
                               "the tensor. slices are of length {}, need at "
                               "least {} slices.\n".format(len(slice),T))
        elif len(key) == 3:
          if isinstance(key[2],int):
            self._matrix_set_helper(key,slices[0])
          else:
            for (i,t) in enumerate(xrange(*key[2].indices(T))):
              self._matrix_set_helper((key[0],key[1],t),slices[i])
        else:
          raise ValueError("key must have at most 3 indices and/or "
                           "slices, {} passed in".format(len(key)))
      else:
        self._slices = slices
        (N,M) = slices[0].shape
        self.shape = (N,M,len(slices))

  def _matrix_set_helper(self,key,slices,mode='frontal'):
    (N,M,T) = self.shape

    #define assigning functions
    if mode == 'frontal':
      (start1, stop1, step1) = key[0].indices(N)
      if isinstance(key[1],int):
        start2 = key[1]
        step2 = 1
      else:
        (start2, stop2, step2) = key[1].indices(M)

      if self._slice_format == 'dense':
        def assign(A, i, j, v):
          A[start1 + i * step1, start2 + step2 * j, key[2]] = v
      else:
        def assign(A, i, j, v):
          A[key[2]][start1 + i * step1, start2 + j * step2] = v

    elif mode == 'lateral':
      (start1, stop1, step1) = key[0].indices(N)
      (start3, stop3, step3) = key[2].indices(T)
      if self._slice_format == 'dense':
        def assign(A, i, j, v):
          A[start1 + i * step1, key[1], start3 + step3 * j] = v
      else:
        def assign(A, i, j, v):
          A[start3 + step3 * j][start1 + i * step1, key[1]] = v

    else:  # transverse
      (start2, stop2, step2) = key[1].indices(M)
      (start3, stop3, step3) = key[2].indices(T)

      if self._slice_format == 'dense':
        def assign(A, i, j, v):
          A[key[0], start2 + i * step2, start3 + step3 * j] = v
      else:
        def assign(A, i, j, v):
          print start2,step2,start3,step3,i,j,v,len(A)
          A[start3 + step3 * j][key[0], start2 + i * step2] = v

    #zero out the elements not being set

    if self._slice_format == 'dense':
      self._slices[key] = 0
    else:
      if isinstance(key[2], int):
        if self._slice_format == 'dok':
          for (i, j) in self._slices[key[2]][key[0], key[1]].iterkeys():
            assign(self._slices, i, j, 0)
        else:
          slice_t = self._slices[key[2]][key[0], key[1]].tocoo()
          for (i, j) in izip(slice_t.row, slice_t.col):
            assign(self._slices, i, j, 0)
      else:
        if self._slice_format == 'dok':
          for (k,t) in enumerate(xrange(*key[2].indices(T))):
            if isinstance(key[1], int):
              for (i, _) in self._slices[t][key[0], key[1]].iterkeys():
                assign(self._slices, i, k, 0)
            else:#assumed key[0] is int
              for (_, j) in self._slices[t][key[0], key[1]].iterkeys():
                assign(self._slices, j, k, 0)
        else:
          for (k,t) in enumerate(xrange(*key[2].indices(T))):
            slice_t = self._slices[t][key[0], key[1]].tocoo()
            if isinstance(key[1], int):
              for i in slice_t.row:
                assign(self._slices, i, k, 0)
            else:
              for j in slice_t.col:
                assign(self._slices, j, k, 0)

    #set the non-zero values
    if slices.format == 'dok':
      for ((i, j), v) in slices.iteritems():
        assign(self._slices, i, j, v)
    else:
      if slices.format != 'coo':
        slices = slices.tocoo()
      for (i, j, v) in izip(slices.row, slices.col, slices.data):
        assign(self._slices, i, j, v)

  def _check_slices_are_sparse(self,slices):
    if isinstance(slices,list):
      for slice in slices:
        if not sp.issparse(slice):
          return False
      return True
    else:
      return False

  def __setitem__(self, key, value):
    if self._slice_format in ['coo', 'dia', 'bsr']:
      raise TypeError('{} format does not support slice assignment, use '
                      'self.convert_slices to reformat the tensor to dok or '
                      'lil format\n'.format(self._slice_format))

    #cast to array if list of numbers
    if isinstance(value,list) and isinstance(value[0],Number):
      value = array(value)

    if isinstance(value,ndarray):
      if len(value.shape) == 2 and \
        (isinstance(key,int) or isinstance(key[0],int)):
        self._set_ndarray_as_slices(value,lateral=False,key=key)
      else:
        self._set_ndarray_as_slices(value,key=key)
    elif isinstance(value,Number):
      if len(key) == 3 and all(map(lambda x:isinstance(x,int),key)):
        if self._slice_format == 'dense':
          self._slices[key] = value
        else:
          self._slices[key[2]][key[:2]] = value
      else:
        raise ValueError('to assign a scalar, there must be 3 indices to '
                         'assign the value to an entry in the Tensor.\n')
    elif isinstance(value,Tensor):
      #recurse on slices if Tensor passed in
      self.__setitem__(key,value._slices)
    elif sp.issparse(value):
      if len(value.shape) == 2 and \
          (isinstance(key,int) or isinstance(key[0],int)):
        self._set_sparse_matrices_as_slices(value,lateral=False,key=key)
      else:
        self._set_sparse_matrices_as_slices(value,key=key)
    elif self._check_slices_are_sparse(value):
      self._set_sparse_matrices_as_slices(value,key=key)
    else:
      raise TypeError("setting value must be one of the following;\n"
                      "list, ndarray, list of sparse matrices, sparse matrix, "
                      "or a subclass of a Number.")

  def save(self, file_name):
    '''
      This function takes in a file name and uses the pickle module to save \
    the Tensor instance.

    :Input:
      file_name - (string)
        The name of the file to save the tensor.
    '''
    with open(file_name,'w') as handle:
      pickle.dump([self._slices,self._slice_format,self.shape],handle)

  def load(self,file_name, make_new = False):
    '''
      This function takes in a file name and loads in into the current tensor \
    instance, if the make_new flag is true it will return a new instance of a \
    tensor.

    :Input:
      file_name - (string)
        The name of the file to load the tensor from.
      make_new - (bool)
        Optional bool which indicates whether or not to create a new instance of
        a tensor, or just copy it into the current instance.
    '''
    with open(file_name,'r') as handle:
      private_elements = pickle.load(handle)

    if make_new:
      return Tensor(private_elements[0])
    else:
      self._slices = private_elements[0]
      self._slice_format = private_elements[1]
      self.shape = private_elements[2]

  def convert_slices(self,format, return_slices=False):
    '''
      This function will convert all of the slices to a desired sparse matrix \
    format or to a dense ndarray, this derives its functionality from the \
    scipy.sparse._.asformat function. To convert to a dense tensor, use the \
     dense keyword.

    :Input:
      format- (string)
        string that specifies the possible formats, valid formats are the \
        supported formats of scipy sparse matrices. see scipy reference for \
        most up to date supported formats. if format is set to 'dense' it \
        will convert the tensor to an ndarray.
      return_slices - (optional bool)
        a boolean which indicates whether or not to apply the conversion to \
        the tensor the function is called from, or to return the slices \
        generated.
    :References:
      https://docs.scipy.org/doc/scipy/reference/sparse.html
    '''
    if self._slice_format == format:
      if return_slices:
        return deepcopy(self._slices)
      else:
        pass
    else:
      if self._slice_format == "dense":
        if format == 'coo':
          sparse_matrix = lambda x: sp.coo_matrix(x)
        elif format == 'dok':
          sparse_matrix = lambda x: sp.dok_matrix(x)
        elif format == 'lil':
          sparse_matrix = lambda x: sp.lil_matrix(x)
        elif format == 'csc':
          sparse_matrix = lambda x: sp.csc_matrix(x)
        elif format == 'csr':
          sparse_matrix = lambda x: sp.csr_matrix(x)
        elif format == 'dia':
          sparse_matrix = lambda x: sp.dia_matrix(x)
        elif format == 'bsr':
          sparse_matrix = lambda x: sp.bsr_matrix(x)
        else:
          raise(ValueError('sparse matrix format must be one of;\n'
                           'coo, dok, lil, csc, csr, dia,or bsr.\n'))
        new_slices = []
        for t in xrange(self.shape[2]):
          new_slices.append(sparse_matrix(self._slices[:,:,t]))

        if return_slices:
          return new_slices
        else:
          self._slice_format = format
          self._slices = new_slices
      else:
        if format == 'dense':
          new_slices = np_zeros(self.shape)
          if self._slice_format == 'dok':
            for t in xrange(self.shape[2]):
              for ((i,j),v) in self._slices[t].iteritems():
                new_slices[i,j,t] = v
          else:
            for t in xrange(self.shape[2]):
              slice = self._slices[t]
              if self._slice_format != 'coo':
                slice = slice.tocoo()
              for (i,j,v) in izip(slice.row,slice.col,slice.data):
                new_slices[i,j,t] = v
          if return_slices:
            return new_slices
          else:
            self._slices = new_slices
        else:
          if return_slices:
            slices = []

          for t, slice in enumerate(self._slices):
            if return_slices:
              slices.append(slice.asformat(format))
            else:
              self._slices[t] = slice.asformat(format)

        if return_slices:
          return slices
        else:
          self._slice_format = format

  def resize(self,shape,order = 'C'):
    '''
      This function takes in a 3 tuple and resizes the tensor according to \
    the dimension of the values passed into the tuple. The method will \
    default to row major order (C like) but may be done in col major order \
    (Fortran like).

    :Input:
      shape - (tuple or list of postive ints)
        a tuple or list with at most length 3 which has the appropriate shapes.
      order - (optional character)
        a character indicating whether or not to use column or row major \
        formatting for the reshape.
    '''
    if not isinstance(shape,list) and not isinstance(shape,tuple):
      raise TypeError('shape is not a valid list or tuple, shape is of type {'
                      '}'.format(type(shape)))
    if len(shape) > 3:
      raise ValueError('shape must be at most length 3, shape is of length {'
                       '}'.format(len(shape)))
    prod = lambda list: reduce(lambda x,y: x*y,list)
    if prod(shape) != prod(self.shape):
      raise ValueError("cannot reshape Tensor with {} entries into shape {}".
                       format(prod(self.shape),shape))

    if self._slice_format == 'dense':
      self._slices.reshape(shape,order)
      self.shape = shape
    else:
      raise NotImplementedError("resize needs to have the sparse case finished")

  def transpose(self, inPlace = False):
    '''
      Creates a new instance a tensor class such that the frontal slices \
    are transposed, and the 2nd through nth slices are flipped. Has the \
    option of returning a new instance, or in place.

    :Input:
      InPlace - (optional bool)
        A boolean indicating whether or not to alter the current tensor, \
        or produce a new one.
    :Returns:
      Tensor Instance
        if InPlace is false, then this function returns a new tensor \
        instance.

    TODO:
      In the dense case, need to find out when np.reshape will create a \
      copy of the data under the hood.
    '''

    (N, M, T) = self.shape
    if self._slice_format == "dense":
      if inPlace and N == M:
        #transpose the 0th slice first
        for i in xrange(N):
          for j in xrange(i):
            temp_val = self._slices[i,j,0]
            self._slices[i,j,0] = self._slices[j,i,0]
            self._slices[j,i,0] = temp_val


        for t in xrange(1,(T-1)/2 + 1):
          # handle off diagonals
          for i in xrange(N):
            for j in xrange(i):
              temp_val = self._slices[i,j,t]
              self._slices[i,j,t] = self._slices[j,i,-t%T]
              self._slices[j,i,-t%T] = temp_val

              temp_val = self._slices[j,i,t]
              self._slices[j,i, t] = self._slices[i, j, -t % T]
              self._slices[i,j,-t % T] = temp_val

          #handle diagonals
          for i in xrange(N):
            temp_val = self._slices[i,i,t]
            self._slices[i,i,t] = self._slices[i,i,-t%T]
            self._slices[i,i,-t%T] = temp_val

        #handle middle slice if one exists
        if (T-1)% 2 != 0:
          for i in xrange(N):
            for j in xrange(i):
              temp_val = self._slices[i,j,(T - 1)/2 + 1]
              self._slices[i, j, (T - 1) / 2 + 1] = \
                self._slices[j,i,(T - 1)/2 + 1]
              self._slices[j,i,(T-1)/2 + 1] = temp_val
      else:
        new_slices = ndarray((M,N,T))
        #handle the off diagonals
        for t in xrange(T):
          for i in xrange(N):
            for j in xrange(M):
              new_slices[j,i,t] = self._slices[i,j,- t %T]

        if inPlace:
          self._slices = new_slices
          self.shape = (M,N,T)
        else:
          return Tensor(new_slices)
    else:
      if inPlace:
        first_slice = self._slices[0].T
        self._slices = map(lambda x: x.T, self._slices[:0:-1])
        self._slices.insert(0,first_slice)
        self.shape = (self.shape[1],self.shape[0],self.shape[2])
      else:
        new_slices = map(lambda x: x.T, self._slices[:0:-1])
        new_slices.insert(0,self._slices[0].T)
        return Tensor(new_slices)

  def squeeze(self, X = None):
    '''
      This function takes in either an n x m matrix and will return a \
    (n x 1 x m) Tensor. This corresponds to thinking of the matrix as a \
    frontal slice, and having the function return it as a lateral slice. \
    Note that if no matrix is passed in, then this function will apply the \
    squeeze function to each one of the frontal slices of the current \
    instance of the tensor. Note that this function is paired with the \
    twist function as an inverse i.e. \
                        X = twist(squeeze(X)) \
    It should be noted that X will be a dok sparse matrix after the \
    function calls.

    :Input:
      X - (optional n x m sparse matrix or ndarray)
        A sparse matrix or ndarrayto be squeezed. Note if  none is passed in, \
        then each frontal slice in self._slices will be squeezed and the \
        instance of the Tensor calling this function will be altered. \
    :Returns:
      Z - (n x 1 x m Tensor)
        A tensor corresponding to a single lateral slice. Doesn't return \
        anything if no X is passed in.
    '''

    if X is not None:
      if sp.issparse(X):
        n = X.shape[0]
        m = X.shape[1]

        if X.format == 'coo':
          X = X.asformat('dok')
        slices = []
        for i in range(m):
          slices.append(X[:,i])
        return Tensor(slices)
      elif isinstance(X,ndarray):
        if len(X.shape) != 2:
          raise ValueError("X passed in is not an order 2 ndarray, create an "
                           "instance of a tensor and call this method on that "
                           "class instance for a 3rd order ndarray.\n")
        else:
          (n,m) = X.shape
          return Tensor(X.reshape((n,1,m)))
      else:
        raise TypeError("X passed in not a sparse matrix, X is of type {"
                        "}\n".format(type(X)))
    else:

      #check format
      if self._slice_format == 'dense':
        (N,M,T) = self.shape
        new_slices = ndarray((N,T,M))
        for t in xrange(T):
          new_slices[:,t,:] = self._slices[:,:,t]
        self._slices = new_slices
        self.shape = (N,T,M)
      else:
        if self._slice_format == 'coo':
          warn("internally converting from coo to dok format, " \
                              "may degrade performance\n",RuntimeWarning)
          self.convert_slices('dok')

        #build new slices
        new_slices = []
        (n,m,T) = self.shape
        for i in range(m):
          new_slices.append(sp.random(n,T,density=0,format='dok'))

        #populate them
        for t,slice in enumerate(self._slices):
          if self._slice_format == 'dok':
            for ((i,j),val) in slice.iteritems():
              new_slices[j][i,t] = val
          else:
            if self._slice_format != 'coo':
              slice = slice.tocoo()
            for (i,j,val) in izip(slice.row,slice.col,slice.data):
              new_slices[j][i,t] = val

        self._slices = new_slices
        self.shape = (n,T,m)
        self._slice_format = 'dok'

  def twist(self, X = None):
    '''
      This function takes in an optional n x 1 x m tensor X and returns a \
    sparse n x m matrix corresponding to rotating the lateral slice to a \
    frontal slice. If no tensor is passed in, the algorithm is run on each \
    of frontal slices of the tensor this routine is being called on. Note \
    that this is the inverse function of the squeeze function, i.e. \
                        X = squeeze(twist(X))

    :Input:
      X - (optional n x 1 x m Tensor)
        This is a lateral slice to be converted to a matrix. Note that \
        if no tensor is passed in, then the routine is run on each of \
        the frontal slices of the current instance of the Tensor the \
        function is called on.
    :Returns:
       Z - (sparse dok matrix or ndarray)
         a matrix corresponding to the lateral slice. The type is determined \
         by whether the input tensor is dense or sparse.
    '''

    if X is not None:
      if not isinstance(X,Tensor):
        raise TypeError("X is not a member of the Tensor class, X is of type "
                        "{}".format(type(X)))
      elif X.shape[1] != 1:
        raise ValueError("X is not a lateral slice as the mode-2 dimension is {}"
                         " not 1,\n if you wish to twist this tensor, "
                         "call twist() on that instance".format(X.shape[1]))
      else:
        if X._slice_format == 'dense':
          Z = ndarray((X.shape[0],X.shape[2]))
          for i in xrange(X.shape[0]):
            for j in xrange(X.shape[2]):
              Z[i,j] = X._slices[i, 0, j]
          return Z
        else:
          Z = sp.random(X.shape[0],X.shape[2],format='dok',density=0)
          for t in range(X.shape[2]):

            slice = X._slices[t]
            if X._slice_format == 'dok':
              for ((i,_),v) in slice.iteritems():
                Z[i,t] = v
            else:
              if X._slice_format != 'coo':
                slice = slice.tocoo()
              for (i,_,v) in izip(slice.row,slice.col,slice.data):
                Z[i,t] = v
          return Z
    else:
      if self._slice_format == 'dense':
        new_slices = ndarray((self.shape[0],self.shape[2],self.shape[1]))
        for i in xrange(self.shape[0]):
          for j in xrange(self.shape[1]):
            for t in xrange(self.shape[2]):
              new_slices[i,t,j] = self._slices[i,j,t]
      else:
        new_slices = []
        for j in xrange(self.shape[1]):
          new_slices.append(sp.random(self.shape[0],self.shape[2],
                                      density=0, format='dok'))

        for t in xrange(self.shape[2]):
          if self._slice_format == 'dok':
            for ((i,j), v) in self._slices[t].iteritems():
              new_slices[j][i,t] = v
          else:
            slice = self._slices[t]
            if self._slice_format != 'coo':
              slice = slice.tocoo()
            for (i,j,v) in izip(slice.row,slice.col,slice.data):
              new_slices[j][i,t] = v

        #convert slices back to original format
        if self._slice_format != 'dok':
          new_slices = map(lambda x:x.asformat(self._slice_format),new_slices)

      self._slices = new_slices
      self.shape = (self.shape[0],self.shape[2],self.shape[1])

  def t_product(self,B,transpose = False):
    '''
      This function takes in another tensor instance and computes the \
    t-product of the two through the block circulant definition of the \
    operation.

    :Input:
      B - (Tensor Instance)
        the mode-2 and mode -3 dimensions of the current instance of a \
        tensor must equal the mode 1 and mode 3 dimensions of B.
      transpose - (optional bool)
        a boolean indicating whether or not to transpose the tensor being \
        called upon before applying the t-product.
    :Returns:
      (Tensor Instance)
      Returns a new Tensor which represents the t-product of the current \
      Tensor and B.

    TODO:
      Can use a complex multiplication formula which may be a little less \
      numerically stable, but will improve complex matrix matrix \
      multiplication constants.
    '''

    if isinstance(B, Tensor):
      (N, M, T) = self.shape
      (M2, L, T2) = B.shape

      #check dimensions of B
      if transpose:
        if N != M2 or T != T2:
          raise ValueError("input Tensor B invalid shape {},\n mode 1 "
                           "dimension and mode 3 dimension must be equal to {} "
                           "and {} respectively"
                           "".format(B.shape, N, T))
      else:
        if M != M2 or T != T2:
          raise ValueError("input Tensor B invalid shape {},\n mode 1 "
                           "dimension and mode 3 dimension must be equal to {} "
                           "and {} respectively"
                           "".format(B.shape,M, T))

      if self._slice_format == 'dense':
        if self._slices.dtype.name == 'complex128':
          new_slices = fft(self._slices)
          if B._slice_format != 'dense':
            B_slices = fft(B.convert_slices('dense',return_slices=True))
          else:
            B_slices = fft(B._slices)

          for t in range(T):
            new_slices[:,:,t] = dot(new_slices[:,:,t],B_slices[:,:,t])
          ifft(new_slices,overwrite_x=True)
        else:
          if transpose:
            A_slices =rfft(self.transpose()._slices)
            (N,M,T) = A_slices.shape
          else:
            A_slices = rfft(self._slices)

          if B._slice_format != 'dense':
            B_slices = rfft(B.convert_slices('dense',return_slices=True))
          else:
            B_slices = rfft(B._slices)

          new_slices = np_empty((N,L,T),dtype ='float64')

          #handle the first slice
          for i in xrange(N):
            for j in xrange(L):
              new_slices[i,j,0] = A_slices[i,0,0]*B_slices[0,j,0]
              for k in xrange(1,M):
                new_slices[i,j,0] += A_slices[i,k,0]*B_slices[k,j,0]

          if T % 2 == 0:
            #handle the last slice
            for i in xrange(N):
              for j in xrange(L):
                new_slices[i, j, -1] = A_slices[i, 0, -1] * B_slices[0, j, -1]
                for k in xrange(1, M):
                  new_slices[i, j, -1] +=\
                    A_slices[i, k, -1] * B_slices[k, j,-1]

            end_T = (T - 1) / 2 + 1
          else:
            end_T = T / 2 + 1

          for t in xrange(1, end_T):
            for i in xrange(N):
              for j in xrange(L):
                real = A_slices[i,0,2*t-1]*B_slices[0,j, 2*t-1] - \
                       A_slices[i,0,2*t]*B_slices[0,j,2*t]
                imaginary = A_slices[i,0,2*t]*B_slices[0,j,2*t-1] + \
                            A_slices[i,0,2*t-1]*B_slices[0,j,2*t]
                new_slices[i,j,t*2] = imaginary
                new_slices[i,j,t*2 -1] = real
                for k in xrange(1,M):
                  real = A_slices[i, k, 2*t-1] * B_slices[k, j, 2*t-1] - \
                         A_slices[i, k, 2*t] * B_slices[k, j, 2*t]
                  imaginary = A_slices[i, k, 2*t] * B_slices[k, j, 2*t-1] +\
                              A_slices[i, k, 2*t-1] * B_slices[k, j, 2*t]
                  new_slices[i, j, 2*t] += imaginary
                  new_slices[i, j, 2*t-1] += real

        irfft(new_slices,overwrite_x=True)
        return Tensor(new_slices)
      else:
        if B._slice_format != 'csr':
          B_slices = B.convert_slices('csr',return_slices=True)
        else:
          B_slices = B._slices
        new_slices = []
        for i in xrange(T):
          if transpose:
            new_slice = sp.random(M, L, density=0)
          else:
            new_slice = sp.random(N,L,density=0)
          for j in xrange(T):
            if transpose:
              new_slice += self._slices[(j + (T - i))%T].T * B_slices[j]
            else:
              new_slice += self._slices[(i + (T - j))%T] * B_slices[j]
          new_slices.append(new_slice)

      return Tensor(new_slices)
    else:
      raise TypeError("B must be a Tensor instance, input is of type {"
                      "}".format(type(B)))

  def scale_tensor(self,scalar, inPlace = False):
    '''
      This function takes in a scalar value and either returns a Tensor scaled \
    by a scalar in the field or scales the tensor in question in place and \
    returns nothing.

    :Input:
      scalar - (subclass of Number)
        must be a scalar value of a field, will be applied to each of the \
        tensor slices.
      inPlace - (optional bool)
        a bool indicating whether or not the tensor this function is called \
        on should be scaled, or whether it should return a new tensor.
    '''

    if not isinstance(scalar, Number):
      raise TypeError("{} is not a subclass of a Number, value passed in is "
                      "of type {}\n".format(scalar,type(scalar)))
    else:
      if inPlace:
        if self._slice_format == 'dense':
          for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
              for k in xrange(self.shape[2]):
                self._slices[i,j,k] *= scalar
        else:
          self._slices = map(lambda x: scalar *x, self._slices)
      else:
        if self._slice_format == 'dense':
          return Tensor(self._slices * scalar)
        else:
          return Tensor(map(lambda x: scalar *x, self._slices))

  def frobenius_norm(self):
    '''
      Returns the Frobenius norm of the tensor. Computed using scipy's norm \
    function for numerical stability.
    '''
    if self._slice_format == 'dense':
      return sqrt(reduce(lambda x,y: x + y**2,self._slices.flat,0))
    else:
      return np_norm(map(lambda x: sp_norm(x,ord='fro'),self._slices))

  def norm(self):
    '''
      This function returns the norm (defined with the t product) of the \
    tensor called upon. Method is computed in a manner rebust to \
    over/underflow by scaling by the largest element of the tensor.

    :Returns:
      norm - (float)
        a float indicating the size of the tensor.
    '''
    norm = 0.0
    if all(map(lambda x: x.nnz == 0,self._slices)):
      return norm
    else:
      T = self.shape[2]

      #compute frobenius norm of \langle X, X \rangle slice wise
      for i in xrange(T):
        slice = self._slices[(T - i) % T].T * self._slices[0]
        for t in xrange(1,T):
          slice += self._slices[(t + (T - i)) % T].T * self._slices[t]

        norm += sp_norm(slice,ord = 'fro')**2

      return sqrt(norm)/self.frobenius_norm()

  def tubal_angle(self,B):
    '''
      This function returns the tubal angle of the current instance of a \
    tensor with another tensor B passed in. This is defined using the inner \
    product defined by the t-product.

    :Returns:
      cos_distance - (float)
        the cosine distance between the current tensor and the tensor passed in.
    '''
    raise NotImplementedError('finish tubal_angle function')

  def find_max(self):
    '''
      This function returns the largest element of the tensor.
    '''
    if self._slice_format == 'dense':
      return self._slices.max()
    else:
      if self._slice_format == 'dok':
        max_val = NINF
        (N,M,T) = self.shape
        for t in xrange(T):
          for val in self._slices[t].itervalues():
            if val > max_val:
              max_val = val

        return max_val
      else:
        if self._slice_format in ['dia','lil']:
          reducing_func = lambda x,y: max(x,y.tocoo().max())
          initial_val = (self._slices[0].tocoo()).max()
          return reduce(reducing_func, self._slices[1:], initial_val)
        else:
          return max(map(lambda x:x.max(),self._slices))

  def zero_out(self,threshold):
    '''
      This function iterates through the non-zeros of the tensor and zeros \
    them out if their absolute value is below the threshold given.

    :Input:
      threshold - (float)
        the tolerance to delete and entry with.
    '''
    if self._slice_format == "dense":
      (N,M,T) = self.shape
      for i in xrange(N):
        for j in xrange(M):
          for t in xrange(T):
            if abs(self._slices[i,j,k]) < threshold:
              self._slices[i,j,k] = 0
    else:
      pass

  def is_equal_to_tensor(self,other, tol = None):
    '''
      This function takes in a another object and a tolerance value and \
    determines whether or not the input tensor is either elementwise equal to \
    or with a tolerance range of another tensor.

    :Input:
      other - (unspecified)
        object to compare the tensor with, may be any time, but will only \
        return true if the input is a Tensor instance
      tol - (float)
        the tolerance to declare whether or not a tensor is elementwise \
        close enough. uses the absolute value, b - tol < a < b + tol.
    :Returns:
      (bool)
        indicates whether or not the two tensors are equal.
    '''
    if tol:
      comp = lambda x,y: abs(x - y) < tol
    else:
      comp = lambda x,y: x == y

    if isinstance(other, Tensor):
      if self.shape == other.shape:
        if self._is_equal_helper(other,comp):
          return other._is_equal_helper(self,comp)
        else:
          return False
      else:
        return False
    else:
      return False

  def _is_equal_helper(self,other,comp):

    if other._slice_format == 'dense':
      for t in xrange(self.shape[2]):
        if self._slice_format == 'dok':
          for ((i,j),v) in self._slices[t].iteritems():
            if not comp(other._slices[i,j,t],v):
              return False
        elif self._slice_format == 'dense':
          for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
              if not comp(other._slices[i,j,t],self._slices[i,j,t]):
                return False
        else:
          if self._slice_format == 'coo':
            slice = self._slices[t]
          else:  # other matrix forms are faster to convert to coo to check
            slice = self._slices[t].tocoo()
          for (i, j, v) in izip(slice.row, slice.col, slice.data):
            if not comp(other._slices[i,j,t], v):
              return False
    else:
      for t in xrange(self.shape[2]):
        # if other is of type coo, change to something one can index into
        if other._slice_format in ['coo','dia','bsr']:
          other_slice = other._slices[t].todok()
        else:
          other_slice = other._slices[t]

        # iterate over dok differently than others
        if self._slice_format == 'dok':
          for ((i, j), v) in self._slices[t].iteritems():
            if not comp(other_slice[i, j], v):
              return False
        elif self._slice_format == 'dense':
          for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
              if not comp(self._slices[i,j,t],other_slice[i,j]):
                return False
        else:
          if self._slice_format == 'coo':
            slice = self._slices[t]
          else:  # other matrix forms are faster to convert to coo to check
            slice = self._slices[t].tocoo()
          for (i, j, v) in izip(slice.row, slice.col, slice.data):
            if not comp(other_slice[i, j], v):
              return False
    return True

'''-----------------------------------------------------------------------------
                              NON-CLASS FUNCTIONS
-----------------------------------------------------------------------------'''

def zeros(shape, dtype = None,format = 'coo'):
  '''
    This function takes in a tuple indicating the size, and a dtype string \
  compatible with scipy's data types and returns a Tensor instance corresponding \
  to shape passed in filled with all zeros.

  :Input:
    shape - (list or tuple of ints)
      a list or tuple with the dimensions of each of the 3 modes. must be of \
      length 3.
    format - (string)
      the format of the sparse matrices to produce, default is COO. If a \
      dense tensor is desired, pass in 'dense' for the value of format.
  :Returns:
    Zero_Tensor - (Tensor Instance)
      an instance of a Tensor of the appropriate dimensions with all zeros.
  '''

  if isinstance(shape,list) or isinstance(shape, tuple):
    if len(shape) == 3:
      for i in range(3):
        if not isinstance(shape[i],int):
          raise TypeError("mode {} dimension must be an integer,\n dimension "
                           "passed in is of type {}\n".format(i,type(i)))
      if format =='dense':
        slices = np_zeros(shape)
      else:
        slices = []
        for t in xrange(shape[2]):
          slices.append(sp.random(shape[0],shape[1],density=0,format=format))
      return Tensor(slices)
    else:
      raise ValueError("shape must be of length 3.\n")

def empty(shape, format = 'dok'):
  '''
  This function takes in a list or tuple of three elements, and an optional\
  bool and returns a tensor with either no elements, or no initialized \
  elements depending on whether the Tensor is requested to be sparse or not.

  :Input:
    shape - (list or tuple of ints)
      a list or tuple of length 3 which indicates the shape of the tensor to \
      instantiate.
    format - (optional string)
      a string indicating how to format the tensor, all scipy sparse arrays \
      are supported, and format can be set to 'dense' to return a ndarray. \
      Note that when a sparse tensor is chosen, it will be equivalent to a \
      zero tensor. default format is dok to ensure ease of setting elements.
  :Returns:
    (Tensor)
      an instance of a tensor.

  '''
  if isinstance(shape,list) or isinstance(shape,tuple):
    if len(shape) == 3:
      if shape[2] == 1:
        raise ValueError("3rd mode cannot be of length 1.\n")
      else:
        if format == "dense":
          slices = np_empty(shape)
        else:
          slices = []
          for t in xrange(shape[2]):
            slices.append(sp.random(shape[0],shape[1],density=0,format=format))
        return Tensor(slices)
    else:
      raise ValueError("shape must be of length 3 to create an instance of a "
                       "tensor.\n passed in shape is of length {}.\n".\
                       format(len(shape)))
  else:
    raise TypeError("shape must be either a length 3 tuple or list, "
                    "shape passed in is of type {}.\n".format(type(shape)))

def random(shape,density = 0.1,dtype = 'float64' ,format='coo',
           random_state = None):
  '''
    This function takes in a tuple indicating the size, and a dtype string \
  compatible with scipy/numpy data types and returns a Tensor instance \
  corresponding to values which are drawn from a uniform distribution from [ \
  0,1) of the shape passed in. Sparsity is controlled by the density \
  parameter which is ignored if the desired tensor will be dense.

  :Input:
    shape - (list or tuple of ints)
      a list or tuple with the dimensions of each of the 3 modes. must be of \
      length 3.
    density - (optional float)
      the amount of non-zeroes hoped to be introduced into the sparse tensor.
    dtype - (optional dtype)
      a datatype consistent with scipy or numpy datatype standards. default np \
      float64.
    format - (optional string)
      the format of the sparse matrices to produce, default is coo.
    random_state - (optional int)
      an integer which is passed in as a seed for each of the slices. Each \
      slice will increment the seed value by 1, so each slice will have a \
      unique seed.
  :Returns:
    (Tensor Instance)

  '''
  if isinstance(shape,list) or isinstance(shape,tuple):
    if len(shape) == 3:
      if format =='dense':
        if random_state is not None: #seeds Mersenne Twister algorithm
          seed(seed=random_state)
        if dtype != 'float64':
          return Tensor(rand(*shape).astype(dtype))
        else:
          return Tensor(rand(*shape))
      else:
        slices = []
        for t in xrange(shape[2]):
          if random_state is not None:
            random_state += 1 #added to make each slice different
          slices.append(sp.rand(shape[0],shape[1],density=density,
                                format=format,dtype = dtype,
                                random_state=random_state))
        return Tensor(slices)
    else:
      raise ValueError("shape must be of length 3.")
  else:
    raise TypeError("shape must be either a list or tuple of length 3.")

def normalize(X,return_sparse_a = False):
  '''
  This function takes in a tensor slice and returns a transverse slice a and \
  tensor V such that each lateral slice V has Frobenius norm 1 and that
  V[:,j,:] * a[j,:,:] = self._slices[:,j,:]. Note that is not the same
  finding an orthogonal basis for the Tensor, just a more convenient way to
  compute multiple normalizations at once.

  :Input:
    X - (Tensor Instance)
      the lateral slice passed in to be normalized.
    return_sparse_a - (optional bool)
      a boolean indicating whether or not the lateral slices should be returned \
      as sparse tensor or not.

  :Returns:
    a - (Tensor Instance)
      A lateral slice with the tubal scalars in each of the rows
    V - (Tensor Instance)
      the lateral slice with frobenius norm 1
  '''

  if not isinstance(X,Tensor):
    raise(TypeError("Input must be a Tensor instance\n, X is of type {"
                    "}".format(type(X))))

  (n,m,T) = X.shape
  A = np_empty((m,1,T))

  V_slices = []

  #compute the fft of the elements in the lateral slice
  for j in range(m):
    if X._slice_format == 'dense':
      slice_fft = rfft(X._slices[:,j,:])
    else:
      slice_fft = np_zeros((n,T))

      # copy the non-zeros in
      for t in xrange(T):
        if X._slice_format == 'dok':
          for ((i,_),v) in X._slices[t][:,j].iteritems():
             slice_fft[i,t] = v
        else:
          slice = X._slices[t][:,j]
          if X._slice_format != 'coo':
            slice = slice.tocoo()
          for (i,v) in izip(slice.row,slice.data):
            slice_fft[i,t] = v
      rfft(slice_fft,overwrite_x=True)

    #normalize all the columns
    tubal_scalar_non_zeros = []

    A[j,0,0] = np_norm(slice_fft[:,0])
    for i in range(n):

      slice_fft[i,0] = slice_fft[i,0]/A[j,0,0]

    if T % 2 == 0:
      end_T = (T-1)/2+1
    else:
      end_T = T/2+1


    for t in xrange(1,end_T):

      #compute the norm of the complex components
      norm = (slice_fft[0,2*t-1]**2 + slice_fft[0,2*t]**2)
      for i in xrange(1,n):
        norm += slice_fft[i,2*t-1]**2
        norm += slice_fft[i,2*t]**2

      norm = sqrt(norm)

      #scale entries
      for i in xrange(n):
        slice_fft[i,2*t-1] = slice_fft[i,2*t-1]/norm
        slice_fft[i,2*t] = slice_fft[i,2*t]/norm
      A[j,0,2*t-1] = norm
      A[j,0,2*t] = 0

    if T % 2 == 0:
      A[j,0,-1] = np_norm(slice_fft[:, T / 2])
      for i in xrange(n):
        slice_fft[i, -1] = slice_fft[i, -1] / A[j,0,-1]


    irfft(slice_fft,overwrite_x = True)
    V_slices.append(sp.dok_matrix(slice_fft))


  irfft(A,overwrite_x=True)

  V = Tensor(V_slices)
  V.squeeze()

  if return_sparse_a:
    slices = []
    for t in xrange(T):
      slices.append(sp.dok_matrix((m,1)))
      for j in xrange(m):
        if abs(A[j,0,t]) > 1e-15:
          slices[t][j,0] = A[j,0,t]
    a = Tensor(slices)
  else:
    a = Tensor(A)

  return V,a

def identity(N,T, format='csr'):
  '''
  This function returns the identity tensor for the t product. The tensor is\
  comprised of T, N x N frontal slices where the first frontal slice is an nth \
  order frontal slice, and all the other frontal slices are 0 matrices.

  :Inputs:
    N - (int)
      the order of the frontal slices
    T - (int)
      the dimension of the third mode.
    format - (optional string)
      the format of the sparse matrices to return, if dense tensor is desired,\
       pass in 'dense'.
  :Returns:
    Identity - (Tensor Instance)
      a tensor corresponding to the identity tensor over the field and inner \
      product induced by the t_product.

  '''
  if format =='dense':
    slices = np_zeros((N,N,T))
    for i in xrange(N):
      slices[i,i,0] = 1.0
  else:
    slices = [sp.identity(N,format=format)]
    for t in xrange(T-1):
      slices.append(sp.random(N,N,density=0,format=format))

  return Tensor(slices)

def sparse_givens_rotation(A,i,j,i_swap,apply = False):
  '''
    This function takes in a Tensor instance and a row and column and either \
  returns a sparse tensor instance corresponding to the tubal givens \
  rotation corresponding to zeroing out the ith ,jth tubal scalar or it \
  will apply it to the tensor passed in.

  :Input:
    A - (Tensor Instance)
      the tensor to compute the givens rotation for.
    i - (int)
      the row of the tubal scalar to zero out.
    j - (int)
      the column of the tubal scalar to zero out.
    i_swap - (int)
      the second row of the tubal scalar to rotate with respect to.
    apply - (optional boolean)
       a bool which indicates whether or not to the apply the givens rotation to \
      the tensor rather than return a tensor instance corresponding to the \
      givens rotation.
  :Returns:
    Q - (Tensor Instance)
      if apply is False (default), then Q will be the tensor instance to \
      which the

  TODO:
    handle i_swap tests to ensure that i =/= i_swap
  '''
  raise NotImplementedError("needs debugging.")

  #check for valid input
  if A._slice_format == 'coo':
    raise ValueError("cannot index into a coo matrix, convert with "
                     "convert_slices methods of Tensor class \n")

  if abs(i) >= A.shape[0]:
    raise ValueError("i is out of bounds, i must be in (-{},{})".format(
      A.shape[0],A.shape[0]))

  if abs(j) >= A.shape[1]:
    raise ValueError("j is out of bounds, j must be in (-{},{})".format(
      A.shape[1],A.shape[1]))
  if abs(i_swap) >= A.shape[0]:
    raise ValueError("i_swap is out of bounds, i_swap must be in (-{},"
                     "{})".format(A.shape[0],A.shape[0]))
  #TODO: handle test for checking i \neq i_swap


  #compute fft of the tubal scalars
  tubal_scalar1 = fftshift(fft(map(lambda x: x[i,j], A._slices)))
  tubal_scalar2 = fftshift(fft(map(lambda x: x[i_swap, j], A._slices)))


  #compute cos(theta) and sin(theta) in fourier domain
  for t in range(A.shape[2]):
    a_mod = abs(tubal_scalar1[t])
    r = a_mod*sqrt(1 + (abs(tubal_scalar2[t])/a_mod)**2)
    tubal_scalar1[t] =  tubal_scalar1[t] / r
    tubal_scalar2[t] =  tubal_scalar2[t] / r

  tubal_scalar2_conj = map(lambda x: conj(x),tubal_scalar2)


  print tubal_scalar1

  #convert back with inverse fftscipy fftpack
  ifft(tubal_scalar1,overwrite_x=True)
  tubal_scalar1 = ifftshift(tubal_scalar1)
  ifft(tubal_scalar2,overwrite_x=True)
  tubal_scalar2 = ifftshift(tubal_scalar2)
  ifft(tubal_scalar2_conj, overwrite_x=True)
  tubal_scalar4 = ifftshift(tubal_scalar2_conj)
  print tubal_scalar1


  if apply:
    pass
  else:
    for t in range(A.shape[2]):
      if t == 0:
        Q_slices = [sp.identity(A.shape[0],format=A._slice_format)]
      else:
        Q_slices.append(sp.random(A.shape[0],A.shape[0],
                           density=0,format=A._slice_format))
      if i_swap < i:
        Q_slices[t][i, i]     = -tubal_scalar2_conj[t]
        Q_slices[t][i_swap, i_swap] =  tubal_scalar2[t]
        Q_slices[t][i, i_swap]   =  tubal_scalar1[t]
        Q_slices[t][i_swap, i]   =  tubal_scalar1[t]
      else:
        Q_slices[t][i_swap, i_swap] =  tubal_scalar1[t]
        Q_slices[t][i, i]     =  tubal_scalar1[t]
        Q_slices[t][i_swap, i]   =  tubal_scalar2[t]
        Q_slices[t][i, i_swap]   = -tubal_scalar2_conj[t]
    return Tensor(Q_slices)

def MGS(A):
  '''
  This function runs the modified gram schmidt algorithm using the inner
  product induced by the t_product over the tubal scalars field.

  :Inputs:
    A - (Tensor Instance)
      the tensor instance to find an orthogonal basis for.
  :Returns:
    Q - (Tensor Instance)
      the orthogonal basis
    R - (Tensor Instance)
      The upper triangular matrix (over the tubal scalars) which relates the
      elements of the orthogonal tensor to the original tensor.
  '''
  if isinstance(A,Tensor):
    (N, M, T) = A.shape
    V = Tensor(deepcopy(A._slices))
    Q = empty(A.shape)
    R = empty((M,M,T))

    for i in xrange(M):
      Q[:,i,:], R[i,i] = normalize(V[:,i,:])
      for j in xrange(M):
        R[i,j] = Q[:,i,:].t_product(V[:,j,:],transpose=True)
        V[:,j,:] = V[:,j,:] - Q[:,i,:] * R[i,j]

    return Q, R
  else:
    raise(TypeError("this function is defined for a tensor instance,\n"
                    " A passed in is of type {}".format(type(A))))

if __name__ == '__main__':
  slices = []
  for i in range(3):
    slices.append(sp.random(5,5,format='dok'))
  A = Tensor(slices)

  x = sp.random(2,2,format='dok')
  A[0] = x