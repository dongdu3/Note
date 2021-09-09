import os
import tqdm
import argparse
import trimesh
import numpy as np
from joblib import Parallel, delayed

def sample_cloud_points_with_occupancy(mesh_path, out_path, n_sample=400000, bbmax=np.array([0.85, 1.8, 0.88], dtype=np.float32), bbmin=np.array([-0.75, -0.2, -0.88], dtype=np.float32)):
    mesh = trimesh.load(mesh_path)

    if mesh.is_watertight:
        n_sample_surf = n_sample // 2
        n_sample_random = n_sample - n_sample_surf

        surface_points, _ = trimesh.sample.sample_surface(mesh, n_sample_surf)
        sample_points = surface_points + np.random.normal(scale=0.05, size=surface_points.shape)

        length = bbmax-bbmin
        random_points = np.random.rand(n_sample_random, 3) * length + bbmin

        sample_points = np.concatenate((surface_points, random_points), axis=0)
        sample_occs = mesh.contains(sample_points)
        
        np.savez(out_path, points=sample_points, labels=sample_occs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/media/administrator/Code/don/4DReconstruction/code/toy_experiment/data',
                        help="file path to source data")
    parser.add_argument('-P', '--process', type=int, default=8, help="number of threads to parallel")
    args = parser.parse_args()

    process_paths = []
    folder_names = sorted(os.listdir(args.data_root))
    for folder_name in folder_names:
        mesh_dir = os.path.join(args.data_root, folder_name, 'mesh')
        occ_dir = os.path.join(args.data_root, folder_name, 'pc_occ')
        if not os.path.exists(occ_dir):
            os.makedirs(occ_dir)
        for name in sorted(os.listdir(mesh_dir)):
            mesh_path = os.path.join(mesh_dir, name)
            if os.path.exists(mesh_path):
                process_paths.append((mesh_path, os.path.join(occ_dir, name.split('.')[0]+'.npz')))
    
    # process data
    Parallel(n_jobs=args.process, verbose=2)(delayed(sample_cloud_points_with_occupancy)(mesh_path, out_path) for mesh_path, out_path in process_paths)

    print('Process done!')