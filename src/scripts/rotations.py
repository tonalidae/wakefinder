import numpy as np
from scipy.spatial.transform import Rotation as R


#cartessian to spherical coordinates
def cartesian_to_spherical(n_vector):
    x, y, z = n_vector
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    # print('r:',r,'Theta',theta,'phi',phi)
    n_sphe = np.array([r,theta,phi])
    return n_sphe
# print(normal_vector_spherical)

#Hay algun problema aqui al recibir el vec en esfericas
def rotate(n_sphe, vector):
    rot3 = R.from_euler('zyx', np.array([-n_sphe[2],-n_sphe[1], 0]))
    # print('rot3',rot3.as_matrix())
    rot_vector = rot3.apply(vector)
    return rot_vector

# def rotate(vec):
#     rot3 = R.from_euler('zyx', np.array([-3.0286791,-1.6850372308367008, 0]))
#     rot_vec = rot3.apply(vec)
#     return rot_vec


#This function verifies if the vectors are perpendicular and rotation is correct
def test_rot(pos_orb, n_vector):
    tangent = np.gradient(pos_orb, axis=0)
    tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]
    dot_product = np.dot(n_vector, tangent.mean(axis=0))
    print(dot_product)
    if (dot_product < 0.0001):
        print("The vectors are perpendicular")
    elif (dot_product < 0):
        print("The vectors are oriented in opposite directions")
    else:
        print("The vectors are oriented in the same direction")


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew