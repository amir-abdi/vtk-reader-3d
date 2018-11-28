from scipy.ndimage import interpolation
import numpy as np
from vtk import vtkStructuredPointsReader
import matplotlib.pyplot as plt


def get_slice(data, angle_x, angle_y, angle_z, displacement, slice_dim, reshape=False):
    '''
    :param data: data to rotate and get a slice from
    :param displacement: value between -1 and +1 to indicate the slice shift in the slice_dim
    :param angle_x: rotation around x axis
    :param angle_y: rotation around y axis
    :param angle_z: rotation around z axis
    :param slice_dim: the dimension to use for slicing
    :return: a single slice along the slice_dim, after the volume was rotated by angle_x, angle_y, angle_z
    '''
    rot_vol = interpolation.rotate(data, angle_x, reshape=reshape, prefilter=False, axes=(1, 2))
    rot_vol = interpolation.rotate(rot_vol, angle_y, reshape=reshape, prefilter=False, axes=(0, 2))
    rot_vol = interpolation.rotate(rot_vol, angle_z, reshape=reshape, prefilter=False, axes=(0, 1))

    dim = slice_dim
    dist = max(0,
               min(rot_vol.shape[dim] - 1,
                   round(rot_vol.shape[dim] / 2 + displacement / 2 * rot_vol.shape[dim])
                   ))

    if dim == 0:
        slice = rot_vol[dist, :, :]
    elif dim == 1:
        slice = rot_vol[:, dim, :]
    elif dim == 2:
        slice = rot_vol[:, :, dim]

    return slice


def read_vtk(filename):
    reader = vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()
    dims = data.GetDimensions()
    points = np.array(data.GetPointData().GetScalars())
    points = points.reshape(dims)
    print('vtk file read, points.shape = ', points.shape)

    return points


if __name__ == "__main__":
    im = read_vtk('VTK04.vtk')
    slice = get_slice(im, angle_x=0, angle_y=90, angle_z=0, slice_dim=0, displacement=0)

    plt.imshow(slice)
    plt.show()

    np.save('test_file.npy', slice)
