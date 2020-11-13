import numpy as np

def colormap(t):
    '''
    :param t: a number from [0, 1]
    :return: a rgb color from blue (0, 0, 255) to red (255, 0, 0)
    '''
    l = 2 + np.sqrt(2)
    t0 = 1 / l
    t1 = (1 + np.sqrt(2)) / l
    r = np.array([255, 0, 0])
    y = np.array([255, 255, 0])
    q = np.array([0, 255, 255])
    b = np.array([0, 0, 255])

    rt = np.zeros_like(r, dtype=np.uint8)
    if (t <= t0):
        s = 1 - t / t0
        rt = s * r + (1 - s) * y
    elif (t <= t1):
        s = 1 - (t - t0) / (t1 - t0)
        rt = s * y + (1 - s) * q
    else:
        s = 1 - (t - t1) / (1 - t1)
        rt = s * q + (1 - s) * b

    return rt

def colormap_depthmap(depmap, backgroud=255):
    '''
    :param depmap: shape [M, N]; value range [0, 255)
    :return: colorful depth map
    '''

    depmap /= 255.

    l = 2 + np.sqrt(2)
    t0 = 1 / l
    t1 = (1 + np.sqrt(2)) / l

    r = np.array([255, 0, 0])
    y = np.array([255, 255, 0])
    q = np.array([0, 255, 255])
    b = np.array([0, 0, 255])

    img_h, img_w = depmap.shape
    depmap_color = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # for t <= t0
    vid1, uid1 = np.where(depmap <= t0)
    s = np.tile(np.array(1 - depmap[vid1, uid1] / t0).reshape((-1, 1)), (1, 3))
    depmap_color[vid1, uid1] = s * r + (1 - s) * y

    # for t > t0 and t <= t1
    vid2, uid2 = np.where(np.logical_and(depmap > t0, depmap <= t1))
    s = np.tile(np.array(1 - (depmap[vid2, uid2] - t0) / (t1 - t0)).reshape((-1, 1)), (1, 3))
    depmap_color[vid2, uid2] = s * y + (1 - s) * q

    # for t > t1 and t < 1
    vid3, uid3 = np.where(np.logical_and(depmap > t1, depmap < 1. - 1e-8))
    s = np.tile(np.array(1 - (depmap[vid3, uid3] - t1) / (1 - t1)).reshape((-1, 1)), (1, 3))
    depmap_color[vid3, uid3] = s * q + (1 - s) * b

    return depmap_color