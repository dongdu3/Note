import os
import numpy as np
import binvox_rw
from tqdm import tqdm
from voxel2layer_torch import *


data_root = '/media/administrator/Code/don/4DReconstruction/code/toy_experiment/data/'

def write_binvox_file(data, filename, voxel_size=128, axis_order='xyz'):     # xyz or xzy
    with open(filename, 'wb') as f:
        voxel = binvox_rw.Voxels(data, [voxel_size, voxel_size, voxel_size], [0, 0, 0], 1, axis_order)
        binvox_rw.write(voxel, f)
    f.close()

with torch.no_grad():
    for folder_name in os.listdir(data_root):
        vox_dir = os.path.join(data_root, folder_name, 'voxel')

        print('processing %s ...' % folder_name)
        for name in tqdm(sorted(os.listdir(vox_dir))):
            vox_path = os.path.join(vox_dir, name)
            fp = open(vox_path, 'rb')
            vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
            vox_data = np.array(vox_data, dtype='uint8')
            vox_size = vox_data.shape[0]

            vox_data = torch.from_numpy(vox_data)
            fp.close()

            idx1, idx2, idx3 = generate_indices(vox_size, device=torch.device('cpu'))
            shape_layer = encode_shape(vox_data, num_layers=1, id1=idx1, id2=idx2, id3=idx3)
            vox_hole_filled = decode_shape(shape_layer, id1=idx1, id2=idx2, id3=idx3)
            vox_hole_filled = vox_hole_filled.numpy()
            write_binvox_file(vox_hole_filled, vox_path, voxel_size=vox_size)
    print('Done!')