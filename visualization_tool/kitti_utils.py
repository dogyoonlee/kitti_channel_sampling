import numpy as np, cv2, os
import math


class Tracklet3d(object):
    ''' Tracklet label '''
    def __init__(self, label_file_line):
        # data = label_file_line.split(' ') # original
        data = label_file_line
        data[3:] = [float(x) for x in data[3:]]
        # extract label, truncation, occlusion
        self.frame_id = int(data[0])
        self.tracklet_id = int(data[1])
        self.type = data[2]  # 'Car', 'Pedestrian', ...
        self.truncation = data[3]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[4]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[5]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[6]  # left
        self.ymin = data[7]  # top
        self.xmax = data[8]  # right
        self.ymax = data[9]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[10]  # box height
        self.w = data[11]  # box width
        self.l = data[12]  # box length (in meters)
        self.t = (data[13], data[14], data[15]
                  )  # location (x,y,z) in camera coord.
        self.ry = data[
            16]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))


class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        if len(data) > 17:
            data[1:17] = [float(x) for x in data[1:17]]
        else:
            data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13]
                  )  # location (x,y,z) in camera coord.
        self.ry = data[
            14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        if len(data) > 15: self.score = float(data[15])
        if len(data) > 16: self.id = int(data[16])
        if len(data) > 17:
            self.state = data[17]
            self.moving_vec = data[18]
            self.oxt_move = data[19]
            self.oxt_x = data[20]
            self.oxt_y = data[21]
            self.obj_x = data[22]
            self.obj_y = data[23]
            self.obj_z = data[24]
            self.move_threshold = data[25]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' %
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' %
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' %
              (self.t[0], self.t[1], self.t[2], self.ry))

    def convert_to_str(self):
        return '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % \
            (self.type, self.truncation, self.occlusion, self.alpha, self.xmin, self.ymin, self.xmax, self.ymax,
                self.h, self.w, self.l, self.t[0], self.t[1], self.t[2], self.ry)


def read_calib_file(filepath, extend_matrix=True):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''

    calibration = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(info)
                   for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
    P1 = np.array([float(info)
                   for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
    P2 = np.array([float(info)
                   for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
    P3 = np.array([float(info)
                   for info in lines[3].split(' ')[1:13]]).reshape([3, 4])

    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    calibration['P0'] = P0
    calibration['P1'] = P1
    calibration['P2'] = P2
    calibration['P3'] = P3
    R0_rect = np.array([float(info)
                        for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect
    calibration['R0_rect'] = rect_4x4
    Tr_velo_to_cam = np.array(
        [float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
    Tr_imu_to_velo = np.array(
        [float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    calibration['Tr_velo_to_cam'] = Tr_velo_to_cam
    calibration['Tr_imu_to_velo'] = Tr_imu_to_velo
    return calibration


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = read_calib_file(calib_filepath, extend_matrix=False)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        if calibs.__contains__(('Tr_velo_to_cam')):
            self.V2C = calibs['Tr_velo_to_cam']
        else:
            self.V2C = calibs['Tr_velo_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        if calibs.__contains__('R0_rect'):
            self.R0 = calibs['R0_rect']
        else:
            self.R0 = calibs['R_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

        self.I2V = calibs['Tr_imu_to_velo']
        self.V2I = inverse_rigid_trans(self.I2V)

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        # TODO: Need modified since read_calib_file function has been changed
        data = {}
        cam2cam = read_calib_file(
            os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(
            os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(
            np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        #x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        #y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        '''
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = (
            (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = (
            (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def read_label_seq(label_filename):  # read the label of the data in sequence
    label_data = list()
    for line in open(label_filename):
        label_data.append(line.rstrip().split())
    label_len = len(label_data)
    seq_label_data = [[] for i in range(int(label_data[label_len - 1][0]) + 1)]
    seq_label_len = len(seq_label_data)
    for i in range(len(label_data)):
        seq_label_data[int(label_data[i][0])].append(
            label_data[i])  # divide data by frame
    return seq_label_data, seq_label_len


def read_label_idx_frame(label_filename,
                         idx):  # read the label of the data in sequence
    label_data = list()
    for line in open(label_filename):
        label_data_tmp = line.rstrip().split()
        if int(label_data_tmp[0]) == int(idx):
            label_data.append(label_data_tmp)
        elif int(label_data_tmp[0]) > int(idx):
            break
        else:
            continue
    objects = [Tracklet3d(line) for line in label_data]
    return objects


def objects_from_label(label_seq_filename):
    label_seq_list = label_seq_filename
    objects = [Tracklet3d(line) for line in label_seq_list]
    return objects


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # if mask: objects = [Object3d_Mask(line) for line in lines]
    objects = [Object3d(line) for line in lines]
    return objects


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    #print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print('cornsers_3d: ', corners_3d)

    # TODO: bugs when the point is behind the camera, the 2D coordinate is wrong

    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=4):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    if qs is not None:
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]),
                             color, thickness)  # use LINE_AA for opencv3

            i, j = k + 4, (k + 1) % 4 + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]),
                             color, thickness)

            i, j = k, k + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]),
                             color, thickness)
    return image


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# oxt file format :
# The GPS/IMU information is given in a single small text file which is
# written for each synchronized frame. Each text file contains 30 values
# which are:


#   - lat:     latitude of the oxts-unit (deg)
#   - lon:     longitude of the oxts-unit (deg)
#   - alt:     altitude of the oxts-unit (m)
#   - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
#   - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
#   - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
#   - vn:      velocity towards north (m/s)
#   - ve:      velocity towards east (m/s)
#   - vf:      forward velocity, i.e. parallel to earth-surface (m/s)
#   - vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
#   - vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
#   - ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
#   - ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
#   - az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
#   - af:      forward acceleration (m/s^2)
#   - al:      leftward acceleration (m/s^2)
#   - au:      upward acceleration (m/s^2)
#   - wx:      angular rate around x (rad/s)
#   - wy:      angular rate around y (rad/s)
#   - wz:      angular rate around z (rad/s)
#   - wf:      angular rate around forward axis (rad/s)
#   - wl:      angular rate around leftward axis (rad/s)
#   - wu:      angular rate around upward axis (rad/s)
#   - posacc:  velocity accuracy (north/east in m)
#   - velacc:  velocity accuracy (north/east in m/s)
#   - navstat: navigation status
#   - numsats: number of satellites tracked by primary GPS receiver
#   - posmode: position mode of primary GPS receiver
#   - velmode: velocity mode of primary GPS receiver
#   - orimode: orientation mode of primary GPS receiver
def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def load_points(foreground_filename):
    points = np.load(foreground_filename)
    return points


def load_image(img_filename):
    return cv2.imread(img_filename)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)