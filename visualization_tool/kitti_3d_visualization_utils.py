import numpy as np
import mayavi.mlab as mlab
import kitti_utils

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


def normalize(vec):
    """normalizes an Nd list of vectors or a single vector
    to unit length.
    The vector is **not** changed in place.
    For zero-length vectors, the result will be np.nan.
    :param numpy.array vec: an Nd array with the final dimension
        being vectors
        ::
            numpy.array([ x, y, z ])
        Or an NxM array::
            numpy.array([
                [x1, y1, z1],
                [x2, y2, z2]
            ]).
    :rtype: A numpy.array the normalized value
    """
    # calculate the length
    # this is a duplicate of length(vec) because we
    # always want an array, even a 0-d array.
    return (vec.T / np.sqrt(np.sum(vec**2, axis=-1))).T


def rotation_matrix_numpy0(axis, theta, dtype=None):
    #dtype = dtype or axis.dtype
    # make sure the vector is normalized
    if not np.isclose(np.linalg.norm(axis), 1.):
        axis = normalize(axis)

    thetaOver2 = theta * 0.5
    sinThetaOver2 = np.sin(thetaOver2)

    return np.array([
        sinThetaOver2 * axis[0], sinThetaOver2 * axis[1],
        sinThetaOver2 * axis[2],
        np.cos(thetaOver2)
    ])


def rotation_matrix_numpy(axis, theta):
    mat = np.eye(3, 3)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)

    return np.array([[
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)
    ], [
        2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)
    ], [
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c
    ]])


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None,
                      bgcolor=(0, 0, 0),
                      fgcolor=None,
                      engine=None,
                      size=(1600, 1000))
    if color is None: color = pc[:, 2]
    #draw points
    mlab.points3d(pc[:, 0],
                  pc[:, 1],
                  pc[:, 2],
                  color,
                  color=None,
                  mode='point',
                  colormap='gnuplot',
                  scale_factor=1,
                  figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    #draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ],
                    dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0),
                tube_radius=None,
                figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0),
                tube_radius=None,
                figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1),
                tube_radius=None,
                figure=fig)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    # mlab.view(azimuth=180, elevation=40, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)

    return fig


#pts_mode='sphere'
def draw_lidar(pc,
               color=None,
               fig=None,
               bgcolor=(0, 0, 0),
               pts_scale=0.3,
               pts_mode='sphere',
               pts_color=None,
               color_by_intensity=False,
               pc_label=False,
               foreground=False,
               box=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    #ind = (pc[:,2]< -1.65)
    #pc = pc[ind]
    pts_mode = 'point'
    # print("====================",pc.shape)
    if fig is None:
        fig = mlab.figure(figure=None,
                          bgcolor=bgcolor,
                          fgcolor=None,
                          engine=None,
                          size=(1600, 1000))
    if color is None: color = pc[:, 0]
    # if color is None: color = (np.linalg.norm(pc[:,0:3], axis=1, ord=2))
    if pc_label:
        print(np.count_nonzero(pc[:, 4]))
        fore_index = pc[:, 4] == 1
        back_index = pc[:, 4] == 0
        mlab.points3d(pc[fore_index, 0],
                      pc[fore_index, 1],
                      pc[fore_index, 2],
                      color=(1, 0, 1),
                      mode='sphere',
                      colormap='gnuplot',
                      scale_factor=0.4,
                      figure=fig)
        # mlab.points3d(pc[back_index, 0], pc[back_index, 1], pc[back_index, 2], pc[back_index, 0], color=pts_color, mode=pts_mode,
        #               colormap='gnuplot',
        #               scale_factor=pts_scale, figure=fig)
        # mlab.points3d(pc[back_index, 0], pc[back_index, 1], pc[back_index, 2], color=(1, 1, 1),
        #               mode='sphere',
        #               colormap='gnuplot',
        #               scale_factor=0.07, figure=fig)

    if color_by_intensity: color = pc[:, 2]

    if pc_label == False:
        mlab.points3d(pc[:, 0],
                      pc[:, 1],
                      pc[:, 2],
                      color=pts_color,
                      mode='sphere',
                      colormap='gnuplot',
                      scale_factor=0.07,
                      figure=fig)

    #draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.5)
    mlab.text3d(0,
                2.5,
                -0.1,
                'LiDAR',
                scale=(0.4, 0.4, 0.4),
                color=(1, 1, 1),
                figure=fig)
    # car = np.load('D:/Dataset/ShapeNet/cadd.npy')
    # col = (0.25, 0.4, 0.9)
    # sf = 0.08

    # mlab.points3d(car[:, 0], car[:, 1], car[:, 2], color=col, mode='sphere',
    #               colormap='gnuplot',
    #               scale_factor=sf, figure=fig)
    # car = np.load('D:/Dataset/ShapeNet/caddd.npy')
    # mlab.points3d(car[:, 0], car[:, 1], car[:, 2], color=col, mode='sphere',
    #               colormap='gnuplot',
    #               scale_factor=sf, figure=fig)
    #draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ],
                    dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0),
                tube_radius=None,
                figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0),
                tube_radius=None,
                figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1),
                tube_radius=None,
                figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov = np.array(
        [  # 45 degree
            [20., 20., 0., 0.],
            [20., -20., 0., 0.],
        ],
        dtype=np.float64)
    #
    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]],
                color=(0, 0.6, 0),
                tube_radius=None,
                line_width=1,
                figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]],
                color=(0, 0.6, 0),
                tube_radius=None,
                line_width=1,
                figure=fig)
    #
    # # draw square region
    # TOP_Y_MIN=-20
    # TOP_Y_MAX=20
    # TOP_X_MIN=0
    # TOP_X_MAX=40
    # TOP_Z_MIN=-2.0
    # TOP_Z_MAX=0.4
    #
    # x1 = TOP_X_MIN
    # x2 = TOP_X_MAX
    # y1 = TOP_Y_MIN
    # y2 = TOP_Y_MAX
    # mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    # mlab.view(
    #     azimuth=180,
    #     elevation=40, # default = 70
    #     # focalpoint='auto',
    #     focalpoint=[12.0909996, -1.04700089, -2.03249991],
    #     distance=62.0, # default = 62.0
    #     figure=fig,
    # )

    # mlab.move()
    return fig


def fg_seg(box, pc, fig, color):
    p1, p2, p3, p4 = box[0][0], box[0][1], box[0][3], box[0][4]
    i = p2 - p1
    j = p3 - p1
    k = p4 - p1
    fg = np.zeros_like(pc[:, 0])
    for n in range(pc.shape[0]):
        v = pc[n, 0:3] - p1
        if 0 < v.dot(i) < i.dot(i) and 0 < v.dot(j) < j.dot(j) and 0 < v.dot(
                k) < k.dot(k):
            fg[n] = 1
    fore_index = fg == 1
    if color != (0, 0, 0):
        scale = 0.13
    else:
        scale = 0.1
    if pc[fore_index][:, 0].shape[0] != 0:
        dist = np.abs(np.max(pc[fore_index][:, 0]))
    else:
        dist = 1000
    mlab.points3d(pc[fore_index, 0],
                  pc[fore_index, 1],
                  pc[fore_index, 2],
                  color=color,
                  mode='sphere',
                  colormap='gnuplot',
                  scale_factor=scale,
                  figure=fig)

    # mlab.view(azimuth=180, elevation=40, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return dist


def draw_gt_boxes3d(gt_boxes3d,
                    fig,
                    color=(1, 1, 1),
                    line_width=3,
                    draw_text=False,
                    text_scale=(1.2, 1.2, 1.2),
                    color_list=None,
                    score=None,
                    obj_id=None,
                    state=None,
                    det_class='Car'):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    if det_class == 'Car':
        obj_text_color = (0, 1, 1)
    elif det_class == 'Pedestrian':
        obj_text_color = (0, 1, 0)
    elif det_class == 'Cyclist':
        obj_text_color = (1, 1, 0)
    else:
        obj_text_color = (1, 1, 1)

    num = len(gt_boxes3d)
    import torch
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            # mlab.text3d(b[4,0], b[4,1], b[4,2], '%.2f'% torch.sigmoid(torch.tensor(score)), scale=text_scale, color=color, figure=fig)
            if obj_id is not None:
                if state is not None:
                    mlab.text3d(b[4, 0],
                                b[4, 1],
                                b[4, 2],
                                '%s, Class: %s, ID: %s' %
                                (state, det_class, obj_id),
                                scale=text_scale,
                                color=obj_text_color,
                                figure=fig)
        for k in range(0, 4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)
    #mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    # mlab.view(azimuth=180, elevation=40, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def xyzwhl2eight(xyzwhl):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1
    '''
    x, y, z, w, h, l = xyzwhl[:6]
    box8 = np.array([[
        x + w / 2, x + w / 2, x - w / 2, x - w / 2, x + w / 2, x + w / 2,
        x - w / 2, x - w / 2
    ],
                     [
                         y - h / 2, y + h / 2, y + h / 2, y - h / 2, y - h / 2,
                         y + h / 2, y + h / 2, y - h / 2
                     ],
                     [
                         z - l / 2, z - l / 2, z - l / 2, z - l / 2, z + l / 2,
                         z + l / 2, z + l / 2, z + l / 2
                     ]])
    return box8.T


def draw_xyzwhl(gt_boxes3d,
                fig,
                color=(1, 1, 1),
                line_width=1,
                draw_text=True,
                text_scale=(1, 1, 1),
                color_list=None,
                rot=False):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        # print(gt_boxes3d[n])
        box6 = gt_boxes3d[n]
        b = xyzwhl2eight(box6)
        if rot:
            b = b.dot(rotz(box6[7]))
            #b = b.dot(rotx(box6[6]))
            #print(rotz(box6[6]))
            #b = b.dot( rotz(box6[6]).dot(rotz(box6[7])) )
            vec = np.array([-1, 1, 0])
            b = b.dot(rotation_matrix_numpy(vec, box6[6]))
            #b = b.dot(roty(box6[7]))

        # print(b.shape,b)
        if color_list is not None:
            color = color_list[n]
        #if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                        [b[i, 2], b[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    mlab.view(azimuth=180,
              elevation=40,
              focalpoint=[12.0909996, -1.04700089, -2.03249991],
              distance=62.0,
              figure=fig)
    return fig


if __name__ == '__main__':
    pc = np.loadtxt('mayavi/kitti_sample_scan.txt')
    # mlab.savefig('pc_view.png', figure=fig)
    fig = draw_lidar(pc)
    mlab.savefig('./pc_view.jpg', figure=fig)
    raw_input()
