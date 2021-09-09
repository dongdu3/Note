import numpy as np
import os
import h5py
import binvox_rw
import argparse
from joblib import Parallel, delayed

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def safe_minmax(x, limit):
    if len(x) == 0:
        return 0, limit
    xmin, xmax = np.min(x), np.max(x)
    xmin = min(xmin, max(xmax - 1, 0))
    xmax = max(xmax, min(xmin + 1, limit))
    return xmin, xmax


def find_bounding_box(voxel_model):
    x, y, z = np.where(voxel_model > 0.5)
    xmin, xmax = safe_minmax(x, voxel_model.shape[0])
    ymin, ymax = safe_minmax(y, voxel_model.shape[1])
    zmin, zmax = safe_minmax(z, voxel_model.shape[2])
    return [xmin, xmax, ymin, ymax, zmin, zmax]


def sample_points_from_vox3d(fname, voxel_model_src, batch_size, d=3, sigma=0.5):
    """sample points from voxel surface
    :param fanme: fname to save hdf5
    :param voxel_model_src: voxel model at n^3 resolution
    :param batch_size: number of points to be sampled
    :param d: size of neighbor window
    :param sigma: sigma for normal distribution
    :return:
    """
    dim_voxel = voxel_model_src.shape[0]

    hdf5_file = h5py.File(fname, 'w')
    hdf5_file.create_dataset('points', [batch_size, 3], np.float, compression=9)
    hdf5_file.create_dataset('values', [batch_size], np.uint8, compression=9)

    # translate voxel model
    voxel_model = np.zeros((dim_voxel + d * 2, dim_voxel + d * 2, dim_voxel + d * 2), dtype=np.uint8)
    voxel_model[d: d+dim_voxel, d: d+dim_voxel, d: d+dim_voxel] = voxel_model_src

    # bounding box
    bbox = find_bounding_box(voxel_model)

    # statistics
    exceed = 0

    # sample points near surface
    sample_points = np.zeros([batch_size, 3], np.float)
    sample_values = np.zeros([batch_size], np.uint8)
    batch_size_counter = 0
    voxel_model_flag = np.zeros_like(voxel_model, dtype=np.uint8)
    for i in range(max(bbox[0] - d, d), min(bbox[1] + d, dim_voxel + d)):
        for j in range(max(bbox[2] - d, d), min(bbox[3] + d, dim_voxel + d)):
            for k in range(max(bbox[4] - d, d), min(bbox[5] + d, dim_voxel + d)):
                if batch_size_counter >= batch_size:
                    break
                neighbor_cube = voxel_model[i - d:i + d + 1, j - d:j + d + 1, k - d:k + d + 1]
                if np.max(neighbor_cube) != np.min(neighbor_cube):
                    sample_points[batch_size_counter, 0] = i
                    sample_points[batch_size_counter, 1] = j
                    sample_points[batch_size_counter, 2] = k
                    sample_values[batch_size_counter] = voxel_model[i, j, k]
                    voxel_model_flag[i, j, k] = 1
                    batch_size_counter += 1

    if batch_size_counter >= batch_size:
        # print("Batch_size exceeded! Desired {}, but got {}.".format(batch_size, batch_size_counter))
        exceed += 1
        batch_size_counter = 0
        voxel_model_flag = np.zeros_like(voxel_model, dtype=np.uint8)
        for i in range(max(bbox[0] - d, d), min(bbox[1] + d, dim_voxel + d), 2):
            for j in range(max(bbox[2] - d, d), min(bbox[3] + d, dim_voxel + d), 2):
                for k in range(max(bbox[4] - d, d), min(bbox[5] + d, dim_voxel + d), 2):
                    if batch_size_counter >= batch_size:
                        break
                    neighbor_cube = voxel_model[i - d:i + d + 1, j - d:j + d + 1, k - d:k + d + 1]
                    if np.max(neighbor_cube) != np.min(neighbor_cube):
                        sample_points[batch_size_counter, 0] = i
                        sample_points[batch_size_counter, 1] = j
                        sample_points[batch_size_counter, 2] = k
                        sample_values[batch_size_counter] = voxel_model[i, j, k]
                        voxel_model_flag[i, j, k] = 1
                        batch_size_counter += 1
    if batch_size_counter == 0:
        raise RuntimeError("no occupied! {}".format(np.sum(voxel_model)))

    # fill remaining slots with random points
    if batch_size_counter < batch_size:
        unpick_pos = np.where(voxel_model_flag < 0.5)
        unpick_pos = np.array(unpick_pos).transpose()
        np.random.shuffle(unpick_pos)
        if batch_size <= batch_size_counter+unpick_pos.shape[0]:
            for it in range(batch_size_counter, batch_size):
                i = unpick_pos[it, 0]
                j = unpick_pos[it, 1]
                k = unpick_pos[it, 2]
                sample_points[batch_size_counter, 0] = i
                sample_points[batch_size_counter, 1] = j
                sample_points[batch_size_counter, 2] = k
                sample_values[batch_size_counter] = voxel_model[i, j, k]
                voxel_model_flag[i, j, k] = 1
                batch_size_counter += 1
        else:
            for it in range(batch_size_counter, batch_size_counter+unpick_pos.shape[0]):
                i = unpick_pos[it, 0]
                j = unpick_pos[it, 1]
                k = unpick_pos[it, 2]
                sample_points[batch_size_counter, 0] = i
                sample_points[batch_size_counter, 1] = j
                sample_points[batch_size_counter, 2] = k
                sample_values[batch_size_counter] = voxel_model[i, j, k]
                voxel_model_flag[i, j, k] = 1
                batch_size_counter += 1

            for it in range(batch_size_counter, batch_size):
                i = np.random.randint(2, dim_voxel - 3) + np.random.uniform(low=-sigma, high=sigma)
                j = np.random.randint(2, dim_voxel - 3) + np.random.uniform(low=-sigma, high=sigma)
                k = np.random.randint(2, dim_voxel - 3) + np.random.uniform(low=-sigma, high=sigma)
                sample_points[batch_size_counter, 0] = i
                sample_points[batch_size_counter, 1] = j
                sample_points[batch_size_counter, 2] = k
                i = int(i)
                j = int(j)
                k = int(k)
                sample_values[batch_size_counter] = voxel_model[i, j, k]
                voxel_model_flag[i, j, k] = 1
                batch_size_counter += 1

    # translate coordinates back
    hdf5_file['points'][:] = (sample_points - d)/float(dim_voxel)-0.5     # belong to [-0.5, 0.5)
    hdf5_file['values'][:] = sample_values
    hdf5_file.close()

def process_one(src_vox_path, tar_occ_path, batch_size):
    # read source voxel data
    if os.path.exists(src_vox_path):
        fp = open(os.path.join(src_vox_path), 'rb')
        vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
        vox_data = np.array(vox_data, dtype='uint8')
        fp.close()

        sample_points_from_vox3d(tar_occ_path, vox_data, batch_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/media/administrator/Code/don/4DReconstruction/code/toy_experiment/data',
                        help="file path to source data")
    parser.add_argument('-P', '--process', type=int, default=10, help="number of threads to parallel")
    args = parser.parse_args()


    batch_size = 250000

    vox_paths = []
    folder_names = sorted(os.listdir(args.data_root))
    for folder_name in folder_names:
        vox_dir = os.path.join(args.data_root, folder_name, 'voxel')
        occ_dir = os.path.join(args.data_root, folder_name, 'pc_occ')
        if not os.path.exists(occ_dir):
            os.makedirs(occ_dir)
        for name in sorted(os.listdir(vox_dir)):
            vox_path = os.path.join(vox_dir, name)
            if os.path.exists(vox_path):
                vox_paths.append((vox_path, os.path.join(occ_dir, name.split('.')[0] + '.h5')))

    Parallel(n_jobs=args.process, verbose=2)(delayed(process_one)(vox_path, occ_path, batch_size) for vox_path, occ_path in vox_paths)


if __name__ == '__main__':
    main()
