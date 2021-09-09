import os
import numpy as np
import binvox_rw
from voxel2layer_torch import *

def write_binvox_file(data, filename, voxel_size=256, axis_order='xyz'):     # xyz or xzy
    with open(filename, 'wb') as f:
        voxel = binvox_rw.Voxels(data, [voxel_size, voxel_size, voxel_size], [0, 0, 0], 1, axis_order)
        binvox_rw.write(voxel, f)
    f.close()

vox_size = 256
voxel_root = '/media/administrator/Dong1/ShapeNetCore.v1/vox' + str(vox_size) + '_surface'
output_root = '/media/administrator/Dong1/ShapeNetCore.v1/vox' + str(vox_size)
if not os.path.exists(output_root):
    os.mkdir(output_root)

cat_id_set = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459',
              '04090263', '04256520', '04379243', '04401088', '04530566']

with torch.no_grad():
    for cat_id in cat_id_set:
        vox_path = os.path.join(voxel_root, cat_id)
        out_path = os.path.join(output_root, cat_id)
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        num = 0
        with open(os.path.join('../', cat_id + '.txt'), 'r') as f:
            name_list = f.readlines()
            for name in name_list:
                name = name.strip()
                num += 1
                print('processing', cat_id, name, num, '...')

                fp = open(os.path.join(vox_path, name + '.binvox'), 'rb')
                vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
                vox_data = np.array(vox_data, dtype='uint8')
                vox_data = torch.from_numpy(vox_data)
                fp.close()

                idx1, idx2, idx3 = generate_indices(vox_size, device=torch.device('cpu'))
                shape_layer = encode_shape(vox_data, num_layers=1, id1=idx1, id2=idx2, id3=idx3)
                vox_hole_filled = decode_shape(shape_layer, id1=idx1, id2=idx2, id3=idx3)
                vox_hole_filled = vox_hole_filled.numpy()
                print(vox_hole_filled.shape)
                write_binvox_file(vox_hole_filled, os.path.join(out_path, name + '.binvox'), voxel_size=vox_size)
            f.close()
    print('Done!')



