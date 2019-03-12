import cv2
import trimesh
import numpy as np

def render_model(mesh, sgrid):
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))

    grid_hits = sgrid[index_ray]
    dist = np.linalg.norm(grid_hits - loc, axis=-1)
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    im = dist_im
    print 'img_val:', im.min(), im.max()
    return im


def make_sgrid(b, alpha, beta, gamma):
    res = b * 2
    pi = np.pi
    phi = np.linspace(0, 180, res * 2 + 1)[1::2]
    theta = np.linspace(0, 360, res + 1)[:-1]
    grid = np.zeros([res, res, 3])
    for idp, p in enumerate(phi):
        for idt, t in enumerate(theta):
            grid[idp, idt, 2] = np.cos((p * pi / 180))
            proj = np.sin((p * pi / 180))
            grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
            grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
    grid = np.reshape(grid, (res * res, 3))
    print 'grid_val:', grid.min(), grid.max()
    return grid


def render_spherical(obj_path=None):
    b = 64
    mesh = trimesh.load(obj_path)
    mesh.vertices = mesh.vertices/1.5    # scale the mesh(smaller)
    print mesh.vertices.min(), mesh.vertices.max()
    sgrid = make_sgrid(b, 0, 0, 0)
    im_depth = render_model(mesh, sgrid)
    im_depth = im_depth.reshape(2 * b, 2 * b)
    im_depth = np.where(im_depth > 1, 1, im_depth)
    return im_depth

if __name__ == '__main__':
    mesh_path = '3.ply'
    save_path = mesh_path.split('.')[0] + '.png'
    im_depth = render_spherical(mesh_path) * 255
    cv2.imwrite(save_path, im_depth)
    print 'Done!'