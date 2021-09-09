import os
import trimesh
import binvox_rw
import numpy as np
from utils import libmcubes
from utils.libsimplify import simplify_mesh

def extract_mesh(occ_hat, threshold=0.2, n_face_simp=None):
    ''' Extracts the mesh from the predicted occupancy grid.

    Args:
        occ_hat (tensor): value grid of occupancies
    '''

    # Make sure that mesh is watertight
    occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
    vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, threshold)

    # Directly return if mesh is empty
    if vertices.shape[0] == 0:
        return None

    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1

    # calculate mesh information
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.
    scale = np.linalg.norm(bbox_max-bbox_center-(1.-threshold))/np.linalg.norm(bbox_max-bbox_center)
    vertices = (vertices-bbox_center)*scale + bbox_center

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    if n_face_simp is not None:
        mesh = simplify_mesh(mesh, n_face_simp, 5.)

    return mesh


if __name__ == '__main__':
    vox_path = '000000.binvox'

    name = vox_path.split('/')[-1][:-7]
    fp = open(vox_path, 'rb')
    vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
    vox_data = np.array(vox_data, dtype=np.float)
    fp.close()

    mesh = extract_mesh(vox_data, n_face_simp=148000)
    if mesh is not None:
        mesh.export(name + '.off', 'off')
    print('Processed done!')


