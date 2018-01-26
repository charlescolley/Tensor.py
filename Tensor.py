'''-----------------------------------------------------------------------------
    This is a class of 3rd order sparse tensors which is designed to support
    the t-product.

    Tensor Class
      Private:
        _slices - (list of sparse matrices)
         a list of sparse scipy matrices which represents the frontal slices of
         the tensor, i.e. the t-th element of the list will be the matrix
         A[:,:,t]. All slices will be the same type of sparse matrix.
        _shape - (tuple of ints)
         a tuple with the shape of the tensor. The ith element of shape
         corresponds to the dimension of the ith mode.
      Public Methods:
        save(folder_name)
--------------------------------------------------------------------------------
  Dependencies
-----------------------------------------------------------------------------'''
import os
import scipy.sparse as sp


'''--------------------------------------------------------------------------'''


class Tensor:

  def __init__(self, slices = None):

    if slices:

      #check for valid slice array
      slice_shape = slices[0].shape
      for t,slice in enumerate(slice):
        if slice.shape != slice_shape:
          raise ValueError("slices must all have the same shape, slice {} "
                           "has shape {}, but slice 0 has shape {}\n".
                           format(t,slice.shape,slice_shape))


      self._slices = slices
      self._shape = (slice_shape[0],slice_shape[1],len(slices))
    else:
      self._slices = []
      self._shape = (0,0,0)

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
