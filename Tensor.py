'''-----------------------------------------------------------------------------
    This is a class of 3rd order sparse tensors which is designed to support
    the t-product.

    Tensor Class
      Private:
        _slices - (list of sparse matrices)
         a list of sparse scipy matrices which represents the frontal slices of
         the tensor, i.e. the t-th element of the list will be the matrix
         A[:,:,t]. All slices will be the same type of sparse matrix.
        _slice_format - (string)
         a string indicating the sparse matrix format of each of the frontal
         slices in the tensor.
        shape - (tuple of ints)
         a tuple with the shape of the tensor. The ith element of shape
         corresponds to the dimension of the ith mode.
      Public Methods:
        save(folder_name, overwrite) UNTESTED
        load(
        convert_slices(format)
        set_frontal_slice            UNTESTED
        get_front_slice              UNTESTED
        set_scalar
        get_scalar
        transpose
        squeeze                      UNTESTED
        twist                        UNTESTED
        t-product                    UNTESTED
        scale_tensor                 UNTESTED
      Overloaded Methods:
        __add__                      UNTESTED
        __mul__                      UNTESTED
        __neg__                      UNTESTED


  TODO: -update constructor to take in a file path
        -change save function make one big flattened matrix and save that
          scipy file
        -write random Tensor
        -write load to handle the flattened tensor save
        -write indexing functions
        -write print overloading
        -write add overloading
        -write todense function
--------------------------------------------------------------------------------
  Dependencies
-----------------------------------------------------------------------------'''
import os
import scipy.sparse as sp
from warnings import warn
from numbers import Number

'''--------------------------------------------------------------------------'''


class Tensor:

  def __init__(self, slices = None):

    if slices:

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

  def __add__(self, other):
    if isinstance(other,Tensor):
      #check dimensions
      if self.shape != other.shape:
        raise ValueError("invalid shape, input tensor must be of shape {}, \n"
                         "input tensor is of shape {}.\n".format(self.shape,
                                                                 other.shape))
      else:
        return Tensor(map(lambda (x,y): x + y, zip(self._slices,other._slices)))
    else:
      raise TypeError("input {} passed in is not an instance of a Tensor, "
                      "parameter passed in is of type {}".
                      format(other, type(other)))

    def __neg__(self):
      return self.scale_tensor(-1)

  '''---------------------------------------------------------------------------
    save(folder_name, overwrite)
      This function takes in a folder name and saves all the slices to the 
      folder. If the folder already exists, a new name will be created by 
      appending a number to the folder name, unless the overwrite flag is 
      True, in which case, the function will delete the contents of the 
      folder and replace them with the tensor slices. Note that since the 
      save function depends on the scipy save_npz function, the slices must 
      be of the appropriate format, if they're not, then they will be 
      converted to the COO format for fast conversion upon loading.
    Input:
      folder_name - (string)
        The name of the folder to save the tensor slices
      overwrite - (optional bool)
        a boolean indicating whether or not to overwrite the contents of the 
        folder if the folder_name given coincides with a folder already in 
        the directory. 
  ---------------------------------------------------------------------------'''
  def save(self, folder_name, overwrite = False):
    if os.path.exists(folder_name):
      if overwrite:
        for file in os.listdir(folder_name):
          os.remove(os.path.join(folder_name,file))
      else:
        # find a valid folder name
        new_folder_index = 1
        while True:
          new_folder_name = folder_name +'_'+ str(new_folder_index)
          if not os.path.exists(new_folder_name):
            folder_name = new_folder_name
            break
          new_folder_index += 1
        os.makedirs(folder_name)
    else:
      os.makedirs(folder_name)

    #save the first slice and check if type is valid for saving.
    try:
      sp.save_npz(folder_name + "slice_0",self._slices[0])
    except AttributeError:
      self.convert_slices('coo')
      sp.save_npz(folder_name + "slice_0",self._slices[0])

    for t,slice in enumerate(self._slices[1:],1):
      file_name = folder_name+"/slice_{}".format(t)
      sp.save_npz(file_name,self._slices[t])


  '''---------------------------------------------------------------------------
      convert_slices(format)
        This function will convert all of the slices to a desired format, 
        this derives its functionality from the scipy.sparse._.asformat 
        function. 
      Input:
        format- (string)
          string that specifies the possible formats, valid formats are the 
          supported formats of scipy sparse matrices. see scipy reference for
          most up to date supported formats
      References:
        https://docs.scipy.org/doc/scipy/reference/sparse.html
  ---------------------------------------------------------------------------'''
  def convert_slices(self,format):
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
      raise ValueError("slice is not a scipy sparse matrix, slice passed in "
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
    if t > self.shape[3]:
      for i in range(t - self.shape[3]-1):
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
  def set_scalar(self,i,j,k,scalar):
    #can't assign elements to a coo matrix
    if self._slices[0].format == 'coo':
      warn("Tensor slices are of type coo, which don't support index "
                    "assignment, Tensor slices are being converted to dok.\n",
                    RuntimeWarning)
      self.convert_slices('dok')

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

    self._slices[k][i,j] = scalar

  '''---------------------------------------------------------------------------
      get_scalar(i,j,k)
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
  def get_scalar(self,i,j,k):
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
  ---------------------------------------------------------------------------'''
  def transpose(self, inPlace = False):
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
  def squeeze(self, X= None):
    if X:
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
      #build new slices
      new_slices = []
      (n,m,T) = self.shape
      for i in range(m):
        new_slices.append(sp.random(n,T,density=0,format='dok'))

      #populate them
      for (t,slice) in enumerate(self._slices):
        for ((i,j),val) in slice.iteritems():
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
    if X:
      if not isinstance(X,Tensor):
        raise TypeError("X is not a member of the Tensor class, X is of type "
                        "{}".format(type(X)))
      elif X.shape[1] != 1:
        raise ValueError("X is not a lateral slice as the mode-2 dimension is {}"
                         " not 1,\n if you wish to twist this tensor, "
                         "call twist() on that instance".format(X.shape[1]))
      else:
        Z = sp.random(X.shape[0],X.shape[2],format='dok',density=0)

  '''---------------------------------------------------------------------------
     t_product(B)
         This function takes in another tensor instance and computes the 
       t-product of the two through the block circulant definition of the 
       operation. 
     Input:
       B - (Tensor Instance)
         the mode-2 and mode -3 dimensions of the current instance of a tensor 
         must equal the mode 1 and mode 3 dimensions of B. 
     Returns: 
       Tensor Instance
         Returns a new Tensor which represents the t-product of the current 
         Tensor and B. 
     Notes:
       Develop future support for  
  ---------------------------------------------------------------------------'''
  def t_product(self,B):

    #TODO: check for tensor object

    #check dimensions of B
    if self.shape[1] != B.shape[0] or self.shape[2] != B.shape[2]:
      raise ValueError("input Tensor B invalid shape {}, mode 1 "
                       "dimension and mode 3 dimension must be equal to {} "
                       "and {} respectively"
                       "".format(B.shape,self.shape[1], self.shape[2]))
    T = self.shape[2]

    new_slices = []
    for i in xrange(T):
      new_slice = sp.random(self.shape[0],B.shape[1],density=0)
      for j in xrange(T):
        new_slice += self._slices[(i+(T - j))%T] * B._slices[j]
      new_slices.append(new_slice)

    return Tensor(new_slices)

  '''---------------------------------------------------------------------------
     zeros(shape)
         This function takes in a tuple indicating the size, and a dtype 
       string compatible with scipy's data types and returns a Tensor instance 
       corresponding to shape passed in filled with all zeros.
  ---------------------------------------------------------------------------'''
  def zeros(self,shape, dtype = None):
    print "balh"



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




