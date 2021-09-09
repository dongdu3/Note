import os
import numpy as np
import os.path
import subprocess
import trimesh
import shutil
from pathlib import Path
from tqdm import tqdm

voxel_size = 512

mesh_root = '/media/administrator/Code/don/4DReconstruction/code/NeuralGraph/out/meshes/'
voxel_root = '/media/administrator/Code/don/4DReconstruction/code/toy_experiment/data/'
if not os.path.exists(voxel_root):
    os.makedirs(voxel_root)

temp_off_path = Path('./temp%d.off' % voxel_size)
temp_vox_path = Path('./temp%d.binvox' % voxel_size)
binvox_exe_list = []
binvox_exe_list.append(['binvox', '-e', '-cb', '-d', str(voxel_size), str(temp_off_path)])

for folder_name in os.listdir(mesh_root):
    mesh_dir = os.path.join(mesh_root, folder_name)
    voxel_dir = os.path.join(voxel_root, folder_name, 'voxel')
    if not os.path.exists(voxel_dir):
            os.makedirs(voxel_dir)
    
    out_vox_path_str_list = []
    out_vox_path_str_list.append(voxel_dir)

    print('processing %s ...' % folder_name)
    for name in tqdm(os.listdir(mesh_dir)):
        if temp_off_path.is_file():
            os.remove(str(temp_off_path))
        mesh = trimesh.load(os.path.join(mesh_dir, name))
        mesh.export(str(temp_off_path))
        
        name = name.split('.')[0]
        out_path = os.path.join(voxel_dir, name + '.binvox')
        for idx, binvox_exe in enumerate(binvox_exe_list):
            if temp_vox_path.is_file():
                os.remove(str(temp_vox_path))
            status = subprocess.run(binvox_exe, stdout=subprocess.PIPE)

            if temp_vox_path.is_file():
                shutil.move(str(temp_vox_path), out_path)

print('Done!')
