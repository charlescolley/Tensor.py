'''-----------------------------------------------------------------------------
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
        convert_slices(format)
        set_frontal_slice
        get_front_slice
        set_scalar
        get_scalar
        transpose              DENSE UNTESTED
        squeeze
        twist
        t-product
        scale_tensor
        find_max                     UNTESTED
        is_equal_to_tensor           UNTESTED
        frobenius_norm               UNTESTED
        norm                         UNTESTED
        cos_distance
        to_dense
      Overloaded Methods:
        __add__
        __sub__
        __mul__                      UNTESTED
        __neg__
        __eq__                       UNTESTED

    Class utilization
      zeros

  TODO: -update constructor to take in a file path
        -write random Tensor
        -write indexing functions
        -write print overloading
        -write todense function
        -squeeze needs to handle different kind of sparse matrices
        -add in non-zero count private element?
--------------------------------------------------------------------------------
  Dependencies
-----------------------------------------------------------------------------'''
import os
import scipy.sparse as sp
import pickle
from itertools import izip
from scipy.sparse.linalg import norm as sp_norm
from numpy import ndarray
from numpy.linalg import norm as np_norm
from warnings import warn
from numbers import Number



'''-----------------------------------------------------------------------------
  Core Class
-----------------------------------------------------------------------------'''

class Tensor:

  def __init__(self, slices = None):

    if slices is not None:
      #if string passed in, assumed to be a file path
      if isinstance(slices,str):
        self.load(slices)
      elif isinstance(slices,ndarray):
        #check order of tensor
        if len(slices.shape) != 3:
          raise ValueError("ndarray must be of order 3, slices passed in has "
                           "order {}\n".format(len(slices.shape)))
        else:
          self._slices = slices
          self.shape = slices.shape
          self._slice_format = "dense"
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
          return Tensor(
            map(lambda (x, y): x + y, zip(self._slices, other._slices)))
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
    if  isinstance(key,slice) or isinstance(key,int):
      return Tensor(self._slices[key])
    elif len(key) == 2:
      if isinstance(key[0],slice):
        return Tensor(map(lambda x: x[:,key[1]],self._slices[key[0]]))
      else:
        return Tensor([self._slices[key[0]][:,key[1]]])
    elif len(key) == 3:
      if isinstance(key[0],slice):
        print map(lambda x: x[key[0],key[1]],self._slices[key[2]])
      else:
        return Tensor([self._slices[key[0]][key[2],key[1]]])
    else:
      raise(ValueError("invalid amount of indices/slices, {} found, must add "
                       "at most 3 indices and/or slices.".format(len(key))))

  def __setitem__(self, key, value):

  '''---------------------------------------------------------------------------
    save(file_name)
      This function takes in a file name and uses the pickle module to save 
      the Tensor intance.
    Input:
      file_name - (string)
        The name of the file to save the tensor.
  ---------------------------------------------------------------------------'''
  def save(self, file_name):
    with open(file_name,'w') as handle:
      pickle.dump([self._slices,self._slice_format,self.shape],handle)


  '''---------------------------------------------------------------------------
    load(file_name)
      This function takes in a file name and loads in into the current tensor 
      instance, if the make_new flag is true it will return a new instance of a 
      tensor. 
    Input:
      file_name - (string)
        The name of the file to load the tensor from.
      make_new - (bool)
        Optional bool which indicates whether or not to create a new instance of
        a tensor, or just copy it into the current instance. 
  ---------------------------------------------------------------------------'''
  def load(self,file_name, make_new = False):
    with open(file_name,'r') as handle:
      private_elements = pickle.load(handle)

    if make_new:
      return Tensor(private_elements[0])
    else:
      self._slices = private_elements[0]
      self._slice_format = private_elements[1]
      self.shape = private_elements[2]

  '''---------------------------------------------------------------------------
      convert_slices(format)
        This function will convert all of the slices to a desired sparse matrix 
        format, this derives its functionality from the scipy.sparse._.asformat 
        function. To convert to a dense tensor, use todense(). 
      Input:
        format- (string)
          string that specifies the possible formats, valid formats are the 
          supported formats of scipy sparse matrices. see scipy reference for
          most up to date supported formats
      References:
        https://docs.scipy.org/doc/scipy/reference/sparse.html
  ---------------------------------------------------------------------------'''
  def convert_slices(self,format):
    if self._slice_format == "dense":
      raise AttributeError("this function is for sparse tensors\n")
    else:
      for t, slice in enumerate(self._slices):
        self._slices[t] = slice.asformat(format)
      self._slice_format = format

  '''---------------------------------------------------------------------------
      get_frontal_slice(t)
        returns the t-th frontal slice. Paired with set_slice().
      Input: 
        t - (int)
          index of the slice to return
      Returns:
        slice - (sparse scipy matrix)
          the t-th slice
  ---------------------------------------------------------------------------'''
  def get_frontal_slice(self,t):
    return self._slices[t]

  '''---------------------------------------------------------------------------
      get_frontal_slice(t)
          replaces the t-th frontal slice. Paired with get_slice().
        Input: 
          t - (int)
            index of the slice to replace. if t is larger than the third-mode 
            dimension, sparse matrices of all zeros are added. 
          slice - (sparse scipy matrix)
            the new t-th slice
  ---------------------------------------------------------------------------'''
  def set_frontal_slice(self, t, slice):

    #check for correct type
    n = self.shape[0]
    m = self.shape[1]
    if not sp.issparse(slice):
      raise TypeError("slice is not a scipy sparse matrix, slice passed in "
                       "is of type {}\n".format(type(slice)))
    if slice.shape != (n,m):
      raise ValueError("slice shape is invalid, slice must be of "
                               "shape ({},"
                       "{}), slice passed in is of shape {}\n",n,m,slice.shape)

    if slice.getformat() != self._slice_format:
      warn("converting frontal slice to format {}\n".
                      format(self._slice_format),
                    UserWarning)

    #insert slice in
    if t > self.shape[2]:
      for i in range(t - self.shape[2]-1):
        self._slices.append(sp.random(n,m,density=0,format=self._slice_format))
      self._slices.append(slice)
      self.shape = (n,m,t)
    else:
      self._slices[t] = slice

  '''---------------------------------------------------------------------------
      set_scalar(i,j,k,scalar)
          This function sets i,j,k element of the tensor to be the value 
         passed as the scalar variable.Note that because COO matrices don't 
         support assignment, the tensor must be converted. paired with the 
         get_scalar function.
      Input:
        i - (integer)
          The mode 1 index to insert the scalar
        j - (integer)
          The mode 2 index to insert the scalar
        k - (integer)
         The mode 3 index to insert the scalar
        scalar - (scalar type)
          The value to be inserted into the tensor, will be cast to the type 
          of whatever type of matrix the slices are comprised of. 
    TODO:
     -expand the tensor when the use passes an index out of range of the 
      current  
  ---------------------------------------------------------------------------'''
  def set_scalar(self,k,j,i,scalar):

    if not isinstance(scalar,Number):
      raise TypeError("scalar must be a subclass of Number, scalar passed "
                      "in is of type{}\n".format(type(scalar)))
    #check for bounds
    if abs(i) > self.shape[0]:
      raise ValueError("i index out of bounds, must be in the domain [-{},"
                       "{}]".format(self.shape[0],self.shape[0]))
    if abs(j) > self.shape[1]:
      raise ValueError("j index out of bounds, must be in the domain [-{},"
                       "{}]".format(self.shape[1],self.shape[1]))
    if abs(k) > self.shape[2]:
      raise ValueError("k index out of bounds, must be in the domain [-{},"
                       "{}]".format(self.shape[2],self.shape[2]))

    #can't assign elements to a coo matrix
    if self._slices[0].format == 'coo':
      warn("Tensor slices are of type coo, which don't support index "
                    "assignment, Tensor slices are being converted to dok.\n",
                    RuntimeWarning)
      self.convert_slices('dok')

    self._slices[k][i,j] = scalar

  '''---------------------------------------------------------------------------
      get_scalar(k,j,i)
        This function gets the i,j,k element of the tensor. paired with the 
        set_scalar function. 
      Input:
        i - (integer)
          The mode 1 index to insert the scalar
        j - (integer)
          The mode 2 index to insert the scalar
        k - (integer)
         The mode 3 index to insert the scalar
      Returns:
        A[i,j,k] - (scalar number)
          returns the value at the i,j element of the kth frontal slice. 
  ---------------------------------------------------------------------------'''
  def get_scalar(self,k,j,i):
    #check bounds of i,j,k
    if abs(i) > self.shape[0]:
      raise ValueError("i index out of bounds, must be in the domain [-{},"
                       "{}]".format(self.shape[0],self.shape[0]))
    if abs(j) > self.shape[1]:
      raise ValueError("j index out of bounds, must be in the domain [-{},"
                       "{}]".format(self.shape[1],self.shape[1]))
    if abs(k) > self.shape[2]:
      raise ValueError("k index out of bounds, must be in the domain [-{},"
                       "{}]".format(self.shape[2],self.shape[2]))

    if self._slices[0].format == 'coo':
      warn("{}th slice is COO format, converting to dok to read value, "
           "please consider converting slices if multiple reads are needed."
           "\n".format(k),RuntimeWarning)
      return self._slices[k].asformat('dok')[i,j]
    else:
      return self._slices[k][i,j]

  '''---------------------------------------------------------------------------
      transpose(inPlace)
        creates a new instance a tensor class such that the frontal slices 
        are transposed, and the 2nd through nth slices are flipped. Has the 
        option of returning a new instance, or in place. 
      Input:
        InPlace - (optional bool)
          A boolean indicating whether or not to alter the current tensor, 
          or produce a new one. 
      Return:
        Tensor Instance
          if InPlace is false, then this function returns a new tensor 
          instance. 
      Note: 
        In the dense case, need to find out when np.reshape will create a 
        copy of the data undert the hood. 
  ---------------------------------------------------------------------------'''
  def transpose(self, inPlace = False):
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
              self._slices[i,j,t] = self.self._slices[j,i,-t%T]
              self._slices[j,i,-t%T] = temp_val

              temp_val = self._slices[j,i,t]
              self._slices[j,i, t] = self.self._slices[i, j, -t % T]
              self._slices[i, j, -t % T] = temp_val

          #handle diagonals
          for i in xrange(N):
            temp_val = self._slices[i,i,t]
            self._slices[i,i,T] = self._slices[i,i,-t%T]
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

  '''---------------------------------------------------------------------------
     squeeze()
       This function takes in either an n x m matrix and will return a 
       (n x 1 x m) Tensor. This corresponds to thinking of the matrix as a 
       frontal slice, and having the function return it as a lateral slice. 
       Note that if no matrix is passed in, then this function will apply the
       squeeze function to each one of the frontal slices of the current 
       instance of the tensor. Note that this function is paired with the 
       twist function as an inverse i.e.   
                            X = twist(squeeze(X))
       It should be noted that X will be a dok sparse matrix after the 
       functio calls. 
     Input:
       X - (optional n x m sparse matrix)
         A sparse matrix to be squeezed. Note if none is passed in, then each 
         frontal slice in self._slices will be squeezed and the instance of 
         the Tensor calling this function will be altered. 
     Returns:
       Z - (n x 1 x m Tensor)
         A tensor corresponding to a single lateral slice. Doesn't return 
         anything if no X is passed in. 
  ---------------------------------------------------------------------------'''
  def squeeze(self, X = None):
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
      else:
        raise TypeError("X passed in not a sparse matrix, X is of type {"
                        "}\n".format(type(X)))
    else:
      #check format
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
      for (t,slice) in enumerate(self._slices):
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

  '''---------------------------------------------------------------------------
     twist(X)
       This function takes in an optional n x 1 x m tensor X and returns a 
       sparse n x m matrix corresponding to rotating the lateral slice to a 
       frontal slice. If no tensor is passed in, the algorithm is run on each of
       frontal slices of the tensor this routine is being called on. Note 
       that this is the inverse function of the squeeze function, i.e. 
                            X = squeeze(twist(X))
     Input:
       X - (optional n x 1 x m Tensor)
         This is a lateral slice to be converted to a sparse matrix. Note 
         that if no tensor is passed in, then the routine is run on each of 
         the frontal slices of the current instance of the Tensor the 
         function is called on. 
     Returns:
       Z - (sparse dok matrix)
         a sparse matrix corresponding to the lateral slice 
  ---------------------------------------------------------------------------'''
  def twist(self, X = None):
    if X is not None:
      if not isinstance(X,Tensor):
        raise TypeError("X is not a member of the Tensor class, X is of type "
                        "{}".format(type(X)))
      elif X.shape[1] != 1:
        raise ValueError("X is not a lateral slice as the mode-2 dimension is {}"
                         " not 1,\n if you wish to twist this tensor, "
                         "call twist() on that instance".format(X.shape[1]))
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
      self._shape = (self.shape[0],self.shape[2],self.shape[1])

  '''---------------------------------------------------------------------------
     t_product(B)
         This function takes in another tensor instance and computes the 
       t-product of the two through the block circulant definition of the 
       operation. 
     Input:
       B - (Tensor Instance)
         the mode-2 and mode -3 dimensions of the current instance of a tensor 
         must equal the mode 1 and mode 3 dimensions of B. 
       transpose - (optional bool)
         a boolean indicating whether or not to transpose the tensor being 
         called upon before applying the t-product.
     Returns: 
       Tensor Instance
         Returns a new Tensor which represents the t-product of the current 
         Tensor and B. 
     Notes:
       Develop future support for  
  ---------------------------------------------------------------------------'''
  def t_product(self,B,transpose = False):

    #TODO: check for tensor object
    if isinstance(B, Tensor):
      #check dimensions of B
      if transpose:
        if self.shape[0] != B.shape[0] or self.shape[2] != B.shape[2]:
          raise ValueError("input Tensor B invalid shape {},\n mode 1 "
                           "dimension and mode 3 dimension must be equal to {} "
                           "and {} respectively"
                           "".format(B.shape, self.shape[0], self.shape[2]))
      else:
        if self.shape[1] != B.shape[0] or self.shape[2] != B.shape[2]:
          raise ValueError("input Tensor B invalid shape {},\n mode 1 "
                           "dimension and mode 3 dimension must be equal to {} "
                           "and {} respectively"
                           "".format(B.shape,self.shape[1], self.shape[2]))
      T = self.shape[2]

      new_slices = []
      for i in xrange(T):
        if transpose:
          new_slice = sp.random(self.shape[1], B.shape[1], density=0)
        else:
          new_slice = sp.random(self.shape[0],B.shape[1],density=0)
        for j in xrange(T):
          if transpose:
            new_slice += self._slices[(j + (T - i))%T].T * B._slices[j]
          else:
            new_slice += self._slices[(i + (T - j))%T] * B._slices[j]
        new_slices.append(new_slice)

      return Tensor(new_slices)
    else:
      raise TypeError("B must be a Tensor instance, input is of type {"
                      "}".format(type(B)))

  '''---------------------------------------------------------------------------
     scale_tensor(scalar, inPlace)
       This function takes in a scalar value and either returns a Tensor 
       scaled by a scalar in the field or scales the tensor in question in 
       place and returns nothing. 
     Input: 
       scalar - (subclass of Number)
         must be a scalar value of a field, will be applied to each of the 
         tensor slices. 
       inPlace - (optional bool)
         a bool indicating whether or not the tensor this function is called 
         on should be scaled, or whether it should return a new tensor. 
  ---------------------------------------------------------------------------'''
  def scale_tensor(self,scalar, inPlace = False):
    if not isinstance(scalar, Number):
      raise TypeError("{} is not a subclass of a Number, value passed in is "
                      "of type {}\n".format(scalar,type(scalar)))
    else:
      if inPlace:
        self._slices = map(lambda x: scalar *x, self._slices)
      else:
        return Tensor(map(lambda x: scalar *x, self._slices))

  '''---------------------------------------------------------------------------
     frobenius_norm()
         Returns the Frobenius norm of the tensor. Computed using scipy's norm 
       function for numerical stability. 
  ---------------------------------------------------------------------------'''
  def frobenius_norm(self):
    return np_norm(map(lambda x: sp_norm(x,ord='fro'),self._slices))

  '''---------------------------------------------------------------------------
     norm()
      This function returns the norm (defined with the t product) of the 
      tensor called upon. Method is computed in a manner rebust to 
      over/underflow by scaling by the largest element of the tensor. 
    Returns:
      norm - (float)
        a float indicating the size of the tensor.
  ---------------------------------------------------------------------------'''
  def norm(self):
    norm = 0.0
    if (map(lambda x: x.nnz == 0),self._slices).all():
      return norm
    else:
      T = self.shape[2]

      #compute frobenius norm of \langle X, X \rangle slice wise
      for i in xrange(T):
        slice = self._slices[(T - i) % T].T * self._slices[0]
        for t in xrange(1,T):
          slice += self._slices[(j + (T - i)) % T].T * self._slices[j]

        norm += sp_norm(slice,ord = 'fro')**2

      return norm/self.frobenius_norm()

  '''---------------------------------------------------------------------------
     tubal_angle(B)
        This function returns the tubal angle of the current instance of a 
      tensor with another tensor B passed in. This is defined using the inner 
      product defined by the t-product. 
    Returns:
      cos_distance - (float)
        the cosine distance between the current tensor and the tensor passed in.
  ---------------------------------------------------------------------------'''

  '''---------------------------------------------------------------------------
     find_max()
       This function returns the largest element of the tensor.
  ---------------------------------------------------------------------------'''
  def find_max(self):
    return max(map(lambda x: x.max(),self._slices))

  '''---------------------------------------------------------------------------
     is_equal_to_tensor(other,tol)
       This function takes in a another object and a tolerance value and 
       determines whether or not the input tensor is either elementwise equal to
       or with a tolerance range of another tensor.
     Input: 
       other - (unspecified)
         object to compare the tensor with, may be any time, but will only 
         return true if the input is a Tensor instance
       tol - (float)
         the tolerance to declare whether or not a tensor is elementwise 
         close enough. uses the absolute value, b - tol < a < b + tol.
     Returns:
       (bool)
         indicates whether or not the two tensors are equal.
  ---------------------------------------------------------------------------'''
  def is_equal_to_tensor(self,other, tol = None):
    if tol:
      comp = lambda x,y: abs(x - y) < tol
    else:
      comp = lambda x,y: x == y

    if isinstance(other, Tensor):
      for t in xrange(self.shape[2]):
        # if other is of type coo, change to something one can index into
        if other._slice_format == 'coo':
          other_slice = other._slices[t].todok()
        else:
          other_slice = other._slices[t]

        # iterate over dok differently than others
        if self._slice_format == 'dok':
          for ((i, j), v) in self._slices[t].iteritems():
            if not comp(other_slice[i, j], v):
              return False
        else:
          if self._slice_format == 'coo':
            slice = self._slices[t]
          else:  # other matrix forms are faster to convert to coo to check
            slice = self._slices[t].tocoo()
          for (i, j, v) in izip(slice.row, slice.col, slice.data):
            if not comp(other_slice[i, j],v):
              return False
      return True
    else:
      return False

  '''---------------------------------------------------------------------------
     todense(make_new)
         this function will convert the current tensor instance into a dense 
       tensor, or if make_new is true, will return a dense tensor instance.  
  ---------------------------------------------------------------------------'''
  def todense(self):
    print "to do"
'''-----------------------------------------------------------------------------
                              NON-CLASS FUNCTIONS
-----------------------------------------------------------------------------'''

'''-----------------------------------------------------------------------------
  zeros(shape)
      This function takes in a tuple indicating the size, and a dtype 
    string compatible with scipy's data types and returns a Tensor instance 
    corresponding to shape passed in filled with all zeros.
  Input:
    shape - (list or tuple of ints)
      a list or tuple with the dimensions of each of the 3 modes. must be of 
      length 3. 
    format - (string)
      the format of the sparse matrices to produce, default is COO. 
  Returns:
    Zero_Tensor - (Tensor Instance)
      an instance of a Tensor of the appropriate dimensions with all zeros. 
-----------------------------------------------------------------------------'''
def zeros(shape, dtype = None,format = 'coo'):
  if isinstance(shape,list) or isinstance(shape, tuple):
    if len(shape) == 3:
      for i in range(3):
        if not isinstance(shape[i],int):
          raise TypeError("mode {} dimension must be an integer,\n dimension "
                           "passed in is of type {}\n".format(i,type(i)))
      slices = []
      for t in range(shape[2]):
        slices.append(sp.random(shape[0],shape[1],density=0,format=format))
      return Tensor(slices)
    else:
      raise ValueError("shape must be of length 3.\n")

'''-----------------------------------------------------------------------------
  random(shape)
      This function takes in a tuple indicating the size, and a dtype 
    string compatible with scipy's data types and returns a Tensor instance 
    corresponding to shape passed of a given density, .
  Input:
    shape - (list or tuple of ints)
      a list or tuple with the dimensions of each of the 3 modes. must be of 
      length 3. 
    dtype - (dtype)
      a datatype consistent with scipy sparse datatype standards
    format - (string)
      the format of the sparse matrices to produce, default is COO.
    random_state - (int)
      an integer which is passed in as a seed for each of the slices. Each 
      slice will increment the seed value by 1, so each slice will have a 
      unique seed. 
  Returns:
    random_Tensor - (Tensor Instance)
      an instance of a Tensor of the appropriate dimensions with all zeros. 
-----------------------------------------------------------------------------'''
