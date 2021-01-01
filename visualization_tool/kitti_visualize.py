''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
수정할 필요가 있다
'''
from __future__ import print_function
import os
import sys
import numpy as np
import cv2
import psutil
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
# import kitti_util as utils
import kitti_utils as utils
import argparse
import glob

if 'mlab' not in sys.modules: import mayavi.mlab as mlab
from kitti_3d_visualization_utils import draw_lidar_simple, draw_lidar, draw_gt_boxes3d, fg_seg

import pdb

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])


class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    def __init__(self, root_dir, args=None):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir  # args.dir = /SSD/Dataset/kitti_t_o
        self.split = args.split  #
        print(root_dir, args.split)
        self.split_dir = os.path.join(
            root_dir, self.split)  # args.dir/split('training'/'testing')

        if self.split == 'training':
            self.num_samples = 7481
        elif self.split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (self.split))
            exit(-1)

        lidar_dir = 'velodyne'
        depth_dir = 'depth'
        # pred_dir  = 'pred'
        # occ_dir   = 'occ_pred'
        pred_dir = None
        occ_dir = None
        if args is not None:
            lidar_dir = args.lidar  #
            depth_dir = args.depthdir  # no use
            pred_dir = args.preddir  # ./results
            occ_dir = args.occdir  # no use
            occpoint_dir = args.occpointdir  # no use
            seq_id = args.seq
            thres_car = args.thrs_car
            thres_ped = args.thrs_ped
            thres_cyc = args.thrs_cyc
        thres_car_default = str(0.66)
        thres_ped_default = str(0.15)
        thres_cyc_default = str(0.2)

        self.image_dir = os.path.join(self.split_dir, 'image_02')
        self.image_dir_seq = os.path.join(
            os.path.join(self.split_dir, 'image_02'), seq_id)
        self.label_dir = os.path.join(self.split_dir, 'label_02')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.calib_dir_seq = os.path.join(
            os.path.join(self.split_dir, 'calib'), '%s.txt' % seq_id)

        self.depthpc_dir = os.path.join(self.split_dir, 'depth_pc')
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.lidar_dir_seq = os.path.join(
            os.path.join(self.split_dir, lidar_dir), seq_id)
        self.depth_dir = os.path.join(os.path.join(self.split_dir, depth_dir),
                                      seq_id)

        # self.pred_dir  = os.path.join(self.split_dir, pred_dir)
        self.pred_dir_list = [
            os.path.join(os.path.join(args.preddir, 'car_3d_det_val_thres'),
                         ('trk_id_included_dy_st_term_1_thres_') + thres_car),
            os.path.join(os.path.join(args.preddir, 'ped_3d_det_val_thres'),
                         ('trk_id_included_dy_st_term_1_thres_') + thres_ped),
            os.path.join(os.path.join(args.preddir, 'cyc_3d_det_val_thres'),
                         ('trk_id_included_dy_st_term_1_thres_') + thres_cyc)
        ]

        if os.path.exists(self.pred_dir_list[0]) is False:
            self.pred_dir_list[0] = os.path.join(
                os.path.join(args.preddir, 'car_3d_det_val_thres'),
                ('trk_id_included_dy_st_term_1_thres_') + thres_car_default)
        if os.path.exists(self.pred_dir_list[1]) is False:
            self.pred_dir_list[1] = os.path.join(
                os.path.join(args.preddir, 'car_3d_det_val_thres'),
                ('trk_id_included_dy_st_term_1_thres_') + thres_ped_default)
        if os.path.exists(self.pred_dir_list[2]) is False:
            self.pred_dir_list[2] = os.path.join(
                os.path.join(args.preddir, 'car_3d_det_val_thres'),
                ('trk_id_included_dy_st_term_1_thres_') + thres_cyc_default)

        self.pred_dir_list_seq = [
            os.path.join(self.pred_dir_list[0], seq_id),
            os.path.join(self.pred_dir_list[1], seq_id),
            os.path.join(self.pred_dir_list[2], seq_id)
        ]
        self.occ_dir = os.path.join(self.split_dir, occ_dir)
        # self.foreground_dir = os.path.join(self.split_dir, 'PR_OUT/foreground_points')
        self.occluded_dir = os.path.join(self.split_dir, occpoint_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        # img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        img_filename = os.path.join(self.image_dir_seq, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float64, n_vec=4):
        assert (idx < self.num_samples)
        # lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        lidar_filename = os.path.join(self.lidar_dir_seq, '%06d.bin' % (idx))
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        # calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        calib_filename = self.calib_dir_seq
        return utils.Calibration(calib_filename)

    # def get_label_objects(self, idx):
    #     assert(idx<self.num_samples and self.split=='training')
    #     label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
    #     return utils.read_label(label_filename)

    def get_label_objects(self, idx, seq_id):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%s.txt' % (seq_id))
        return utils.read_label_idx_frame(label_filename, idx)

    def get_foreground_points(self, idx):
        assert (idx < self.num_samples)
        foreground_filename = os.path.join(self.foreground_dir,
                                           '%06d.npy' % (idx))
        return utils.load_points(foreground_filename)

    def get_occluded_points(self, idx):
        assert (idx < self.num_samples)
        occluded_filename = os.path.join(self.occluded_dir, '%06d.npy' % (idx))
        return utils.load_points(occluded_filename)

    # def get_pred_objects(self, idx):
    #     assert(idx<self.num_samples and self.split=='training')
    #     pred_filename = os.path.join(self.pred_dir, '%06d.txt'%(idx))
    #     is_exist = os.path.exists(pred_filename)
    #     if is_exist:
    #         return utils.read_label(pred_filename)
    #     else:
    #         return None

    def get_pred_objects(self, idx, seq_id, det_type=0):
        assert (idx < self.num_samples and self.split == 'training')
        pred_filename = os.path.join(self.pred_dir_list_seq[det_type],
                                     '%06d.txt' % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_occ_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        occ_filename = os.path.join(self.occ_dir, '%06d.txt' % (idx))
        is_exist = os.path.exists(occ_filename)
        if is_exist:
            return utils.read_label(occ_filename)
        else:
            return None

    def get_depth(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.depth_dir, '%06d.png' % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.depth_dir, '%06d.png' % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.depthpc_dir, '%06d.bin' % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        #print(lidar_filename, is_exist)
        #return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        pred_filename = os.path.join(self.pred_dir, '%06d.txt' % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        depth_filename = os.path.join(self.depth_dir, '%06d.txt' % (idx))
        return os.path.exists(depth_filename)


class kitti_object_video(object):
    ''' Load data for KITTI videos '''
    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted([os.path.join(img_dir, filename) \
            for filename in os.listdir(img_dir)])
        self.lidar_filenames = sorted([os.path.join(lidar_dir, filename) \
            for filename in os.listdir(lidar_dir)])
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        #assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, 'dataset/2011_09_26/')
    dataset = kitti_object_video(\
        os.path.join(video_path, '2011_09_26_drive_0023_sync/image_02/data'),
        os.path.join(video_path, '2011_09_26_drive_0023_sync/velodyne_points/data'),
        video_path)
    print(len(dataset))
    for i in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        Image.fromarray(img).show()
        draw_lidar(pc)
        raw_input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        raw_input()
    return


def show_image_with_boxes(img,
                          objects,
                          calib,
                          objects_pred,
                          show3d=True,
                          depth=None):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    img3 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'DontCare': continue
        if obj.type == 'Misc': continue
        if obj.type == 'Cyclist': continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(img2,
                                          box3d_pts_2d,
                                          color=(0, 255, 0))
    for obj in objects_pred:
        if obj.type == 'DontCare': continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), (255, 0, 0), 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(img2,
                                          box3d_pts_2d,
                                          color=(255, 0, 0))

        # project
        #box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        #box3d_pts_32d = utils.box3d_to_rgb_box00(box3d_pts_3d_velo)
        #box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        #img3 = utils.draw_projected_box3d(img3, box3d_pts_32d)
    #print("img1:", img1.shape)
    # Image.fromarray(img1).show()
    #print("img3:",img3.shape)
    #Image.fromarray(img3).show()
    if show3d:
        #print("img2:",img2.shape)
        Image.fromarray(img2).show()
    if depth is not None:
        Image.fromarray(depth).show()


def show_image_with_3D_boxes(img,
                             objects,
                             objects_pred,
                             objects_occ,
                             calib,
                             show3d=True,
                             depth=None):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    img3 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'DontCare': continue
        if obj.type == 'Misc': continue
        if obj.type == 'Cyclist': continue
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)

    if objects_pred is not None:
        for obj_pred in objects_pred:
            if obj_pred.type == 'DontCare': continue
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(
                obj_pred, calib.P)
            img2 = utils.draw_projected_box3d(img2,
                                              box3d_pts_2d,
                                              color=(0, 255, 255))

        for obj_occ in objects_occ:
            if obj_occ.type == 'DontCare': continue
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj_occ, calib.P)
            img2 = utils.draw_projected_box3d(img2,
                                              box3d_pts_2d,
                                              color=(255, 0, 0))

    if show3d:
        #print("img2:",img2.shape)
        Image.fromarray(img2).show()
    if depth is not None:
        Image.fromarray(depth).show()


def show_image_with_boxes_3type(img, objects, calib, objects2d, name,
                                objects_pred):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img)  # for 2d bbox
    type_list = ['Pedestrian', 'Car', 'Cyclist']
    # draw Label
    color = (0, 255, 0)
    for obj in objects:
        if obj.type not in type_list: continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), color, 3)
    startx = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [obj.type for obj in objects if obj.type in type_list]
    text_lables.insert(0, "Label:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0,
                    cv2.LINE_AA)
    # draw 2D Pred
    color = (0, 0, 255)
    for obj in objects2d:
        cv2.rectangle(img1, (int(obj.box2d[0]), int(obj.box2d[1])),
                      (int(obj.box2d[2]), int(obj.box2d[3])), color, 2)
    startx = 85
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [type_list[obj.typeid - 1] for obj in objects2d]
    text_lables.insert(0, "2D Pred:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0,
                    cv2.LINE_AA)
    # draw 3D Pred
    if objects_pred is not None:
        color = (255, 0, 0)
        for obj in objects_pred:
            if obj.type not in type_list: continue
            cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                          (int(obj.xmax), int(obj.ymax)), color, 1)
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [
            obj.type for obj in objects_pred if obj.type in type_list
        ]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0,
                        cv2.LINE_AA)

    Image.fromarray(img1).show()
    cv2.imwrite("imgs/" + str(name) + ".png", img1)


def get_lidar_in_image_fov(pc_velo,
                           calib,
                           xmin,
                           ymin,
                           xmax,
                           ymax,
                           return_more=False,
                           clip_distance=2):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_lidar_index_in_image_fov(pc_velo,
                                 calib,
                                 xmin,
                                 ymin,
                                 xmax,
                                 ymax,
                                 return_more=False,
                                 clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def depth_region_pt3d(depth, obj):
    b = obj.box2d
    #depth_region = depth[b[0]:b[2],b[2]:b[3],0]
    pt3d = []
    #import pdb; pdb.set_trace()
    for i in range(int(b[0]), int(b[2])):
        for j in range(int(b[1]), int(b[3])):
            pt3d.append([j, i, depth[j, i]])
    return np.array(pt3d)


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def show_lidar_with_depth(pc_velo,
                          objects,
                          calib,
                          img_fov=False,
                          img_width=None,
                          img_height=None,
                          objects_pred_car=None,
                          objects_pred_ped=None,
                          objects_pred_cyc=None,
                          depth=None,
                          cam_img=None,
                          constraint_box=False,
                          pc_label=False,
                          save=False,
                          gt=False,
                          save_velo_dir=None,
                          idx=None,
                          fig=None,
                          view_=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''

    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(pc_velo[:, :3], calib, 0,
                                                     0, img_width, img_height)
        pc_velo = pc_velo[pc_velo_index, :]

    draw_lidar(pc_velo, fig=fig, pc_label=pc_label, pts_color=(1, 1, 1))
    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "D:/Dataset/KITTI/3D/training/depth_pc"
            save_filename = os.path.join(vely_dir, '%06d.bin' % (data_idx))
            print(save_filename)
            #np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    color = (0, 1, 0)
    if gt == True:
        for obj in objects:
            if obj.type == 'DontCare': continue
            if obj.type == 'Misc': continue
            if obj.type == 'Cyclist': continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            fg_seg([box3d_pts_3d_velo], pc_velo, fig, color)

    if objects_pred_car is not None:
        # color=(0,1,0)
        for obj in objects_pred_car:
            if obj.type == 'DontCare': continue
            if obj.type == 'Misc': continue
            if obj.type == 'Cyclist': continue
            if obj.type == 'Pedestrian': continue
            obj_state = obj.state
            obj_id = obj.id
            if obj_state == 'static':
                color = (1, 0, 0)
            elif obj_state == 'dynamic':
                color = (0, 0, 1)
            else:
                color = (1, 1, 0)
            # Draw 3d bounding box for car
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            box3d_pts_3d_velo += [-0.2, 0.2, 0]

            draw_gt_boxes3d([box3d_pts_3d_velo],
                            fig=fig,
                            color=color,
                            draw_text=True,
                            score=obj.score,
                            obj_id=obj_id,
                            state=obj_state,
                            det_class=obj.type)
            fg_seg([box3d_pts_3d_velo], pc_velo, fig, color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(
                obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                        color=color,
                        tube_radius=None,
                        line_width=1,
                        figure=fig)

    if objects_pred_ped is not None:
        # color=(0,1,0)
        for obj in objects_pred_ped:
            if obj.type == 'DontCare': continue
            if obj.type == 'Misc': continue
            if obj.type == 'Cyclist': continue
            if obj.type == 'Car': continue
            obj_state = obj.state
            obj_id = obj.id
            if obj_state == 'static':
                color = (1, 0, 0)
            elif obj_state == 'dynamic':
                color = (0, 0, 1)
            else:
                color = (1, 1, 0)
            # Draw 3d bounding box for ped
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            box3d_pts_3d_velo += [-0.2, 0.2, 0]

            draw_gt_boxes3d([box3d_pts_3d_velo],
                            fig=fig,
                            color=color,
                            draw_text=True,
                            score=obj.score,
                            obj_id=obj_id,
                            state=obj_state,
                            det_class=obj.type)
            fg_seg([box3d_pts_3d_velo], pc_velo, fig, color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(
                obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                        color=color,
                        tube_radius=None,
                        line_width=1,
                        figure=fig)

    if objects_pred_cyc is not None:
        # color=(0,1,0)
        for obj in objects_pred_cyc:
            if obj.type == 'DontCare': continue
            if obj.type == 'Misc': continue
            if obj.type == 'Car': continue
            if obj.type == 'Pedestrian': continue
            obj_state = obj.state
            obj_id = obj.id
            if obj_state == 'static':
                color = (1, 0, 0)
            elif obj_state == 'dynamic':
                color = (0, 0, 1)
            else:
                color = (1, 1, 0)
            # Draw 3d bounding box for cyc
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            box3d_pts_3d_velo += [-0.2, 0.2, 0]

            draw_gt_boxes3d([box3d_pts_3d_velo],
                            fig=fig,
                            color=color,
                            draw_text=True,
                            score=obj.score,
                            obj_id=obj_id,
                            state=obj_state,
                            det_class=obj.type)
            fg_seg([box3d_pts_3d_velo], pc_velo, fig, color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(
                obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                        color=color,
                        tube_radius=None,
                        line_width=1,
                        figure=fig)

    azimuth, elevation, focalpoint, distance = view_
    mlab.view(azimuth=azimuth,
              elevation=elevation,
              focalpoint=focalpoint,
              distance=distance)

    if save_velo_dir is not None:
        save_filename = os.path.join(save_velo_dir, '%06d.jpg' % (idx))

        mlab.savefig(save_filename, figure=fig)
        print("==> save filename : ", save_filename)
        mlab.clf()


def show_segmented_lidar_with_depth(pc_velo,
                                    pc_seg,
                                    objects,
                                    calib,
                                    img_fov=False,
                                    img_width=None,
                                    img_height=None,
                                    objects_pred=None,
                                    depth=None,
                                    cam_img=None,
                                    constraint_box=False,
                                    pc_label=False,
                                    save=False):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None,
                      bgcolor=(0, 0, 0),
                      fgcolor=None,
                      engine=None,
                      size=(1000, 500))
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(pc_velo[:, :3], calib, 0,
                                                     0, img_width, img_height)
        pc_velo = pc_velo[pc_velo_index, :]
        print(('FOV point num: ', pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    # draw_lidar(pc_velo, fig=fig, pc_label=False)

    pc_seg[:, 0:3] = calib.project_rect_to_velo(pc_seg[:, 0:3])
    draw_lidar(pc_seg, fig=fig, pc_label=True)

    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "D:/Dataset/KITTI/3D/training/depth_pc"
            save_filename = os.path.join(vely_dir, '%06d.bin' % (data_idx))
            print(save_filename)
            #np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == 'DontCare': continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == 'DontCare': continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo],
                            fig=fig,
                            color=color,
                            draw_text=True,
                            score=obj.score)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(
                obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                        color=color,
                        tube_radius=None,
                        line_width=1,
                        figure=fig)
    mlab.show(1)


def show_occluded_lidar_with_depth(pc_velo,
                                   pc_occ,
                                   objects,
                                   calib,
                                   img_fov=False,
                                   img_width=None,
                                   img_height=None,
                                   objects_pred=None,
                                   objects_occ=None,
                                   depth=None,
                                   cam_img=None,
                                   constraint_box=False,
                                   pc_label=False,
                                   save=False):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from kitti_3d_visualization_utils import draw_lidar_simple, draw_lidar, draw_gt_boxes3d, fg_seg

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None,
                      bgcolor=(0.1, 0.1, 0.1),
                      fgcolor=None,
                      engine=None,
                      size=(1000, 500))
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(pc_velo[:, :3], calib, 0,
                                                     0, img_width, img_height)
        pc_velo = pc_velo[pc_velo_index, :]
        print(('FOV point num: ', pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    # draw_lidar(pc_velo, fig=fig, pc_label=False)

    pc_occ[:, 0:3] = calib.project_rect_to_velo(pc_occ[:, 0:3])
    draw_lidar(pc_occ, fig=fig, pc_label=True)

    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "D:/Dataset/KITTI/3D/training/depth_pc"
            save_filename = os.path.join(vely_dir, '%06d.bin' % (data_idx))
            print(save_filename)
            #np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == 'DontCare': continue
        # Draw 3d bounding box
        if obj.type == 'Misc': continue
        if obj.type == 'Cyclist': continue
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # print("box3d_pts_3d_velo:")
        # print(box3d_pts_3d_velo)
        fg_seg([box3d_pts_3d_velo], pc_occ, fig, color)
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
        # fg_seg([box3d_pts_3d_velo], pc_occ, fig, color)

    if objects_pred is not None:
        color = (0, 0.8, 1)
        for obj in objects_pred:
            if obj.type == 'DontCare': continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            # print("box3d_pts_3d_velo:")
            # print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo],
                            fig=fig,
                            color=color,
                            draw_text=False,
                            score=obj.score)
            fg_seg([box3d_pts_3d_velo], pc_occ, fig, color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(
                obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                        color=color,
                        tube_radius=None,
                        line_width=1,
                        figure=fig)
        color = (1, 0, 0)
        import torch
        for obj in objects_occ:
            if obj.type == 'DontCare': continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            # print("box3d_pts_3d_velo:")
            # print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo],
                            fig=fig,
                            color=color,
                            draw_text=False,
                            score=obj.score)
            fg_seg([box3d_pts_3d_velo], pc_occ, fig, color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(
                obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                        color=color,
                        tube_radius=None,
                        line_width=1,
                        figure=fig)
    mlab.show(1)

def save_depth0(data_idx, pc_velo, calib, img_fov, img_width, img_height, \
               depth, constraint_box=False):

    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(pc_velo[:, :3], calib, 0,
                                                     0, img_width, img_height)
        pc_velo = pc_velo[pc_velo_index, :]
        type = np.zeros((pc_velo.shape[0], 1))
        pc_velo = np.hstack((pc_velo, type))
        print(('FOV point num: ', pc_velo.shape))
    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))

        type = np.ones((depth_pc_velo.shape[0], 1))
        depth_pc_velo = np.hstack((depth_pc_velo, type))
        print("depth_pc_velo:", depth_pc_velo.shape)

        depth_pc = np.concatenate((pc_velo, depth_pc_velo), axis=0)
        print("depth_pc:", depth_pc.shape)

    vely_dir = "D:/Dataset/KITTI/3D/training/depth_pc"
    save_filename = os.path.join(vely_dir, '%06d.bin' % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)

def save_depth(data_idx, pc_velo, calib, img_fov, img_width, img_height, \
               depth, constraint_box=False):

    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc = np.hstack((depth_pc_velo, indensity))

        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/obj/training/depth_pc"
    save_filename = os.path.join(vely_dir, '%06d.bin' % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def show_lidar_with_boxes(pc_velo,
                          objects,
                          calib,
                          img_fov=False,
                          img_width=None,
                          img_height=None,
                          objects_pred=None,
                          depth=None,
                          cam_img=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from kitti_3d_visualization_utils import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None,
                      bgcolor=(0, 0, 0),
                      fgcolor=None,
                      engine=None,
                      size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo[:, 0:3], calib, 0, 0,
                                         img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    #pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == 'DontCare': continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # print("box3d_pts_3d_velo:")
        # print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

        # Draw depth
        if depth is not None:
            #import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))
        #

        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                    color=color,
                    tube_radius=None,
                    line_width=1,
                    figure=fig)
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == 'DontCare': continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(
                obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2],
                        color=color,
                        tube_radius=None,
                        line_width=1,
                        figure=fig)
    mlab.show(1)


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where((pc[:,0]>=bmin[0]) & (pc[:,0]<=bmax[0]) \
                 & (pc[:,1]>=bmin[1]) & (pc[:,1]<=bmax[1]) \
                 & (pc[:,2]>=bmin[2]) & (pc[:,2]<=bmax[2]))[0]
    #print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''

    #print(('All point num: ', pc_velo.shape[0]))

    #draw_lidar(pc_velo, fig=fig)
    #color=(0,1,0)
    for obj in objects:
        if obj.type == 'DontCare': continue
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(
            imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
                   1,
                   color=tuple(color),
                   thickness=-1)
    Image.fromarray(img).show()
    return img


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    ''' top_view image'''
    #print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print('top_image:', top_image.shape)

    # gt

    def bbox3d(obj):
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != 'DontCare']
    gt = np.array(boxes3d)
    #print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != 'DontCare']
    top_image = utils.draw_box3d_on_top(top_image,
                                        gt,
                                        text_lables=lines,
                                        scores=None,
                                        thickness=1,
                                        is_gt=True)
    # pred
    if objects_pred is not None:
        boxes3d = [
            bbox3d(obj) for obj in objects_pred if obj.type != 'DontCare'
        ]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != 'DontCare']
        top_image = utils.draw_box3d_on_top(top_image,
                                            gt,
                                            text_lables=lines,
                                            scores=None,
                                            thickness=1,
                                            is_gt=False)

    Image.fromarray(top_image).show()


def show_lidar_on_image_with_box(pc_velo,
                                 img,
                                 calib,
                                 img_width,
                                 img_height,
                                 objects,
                                 show3d=True,
                                 depth=None,
                                 objects_pred=None):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = np.abs(imgfov_pc_rect[i, 2])
        # depth = np.linalg.norm(imgfov_pc_rect[i, 0:3], axis=0, ord=2)
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(
            imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
                   1,
                   color=tuple(color),
                   thickness=-1)
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img)  # for 2d bbox
    for obj in objects:
        if obj.type == 'DontCare': continue
        cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                      (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2)

    if objects_pred is not None:
        for obj in objects_pred:
            if obj.type == 'DontCare': continue
            cv2.rectangle(img1, (int(obj.xmin), int(obj.ymin)),
                          (int(obj.xmax), int(obj.ymax)), (255, 0, 0), 2)
            closest = 1000000000
            furthest = 0
            for i in range(imgfov_pts_2d.shape[0]):
                if (int(obj.xmin) < np.round(imgfov_pts_2d[i, 0]) < int(
                        obj.xmax)) and (int(obj.ymin) < np.round(
                            imgfov_pts_2d[i, 1]) < int(obj.ymax)):
                    depth = np.abs(imgfov_pc_rect[i, 2])
                    if depth < closest:
                        closest = depth
                    if depth > furthest:
                        furthest = depth

            print("Disparity : %d" % (furthest - closest))

        # project
        # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # box3d_pts_32d = utils.box3d_to_rgb_box00(box3d_pts_3d_velo)
        # box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        # img3 = utils.draw_projected_box3d(img3, box3d_pts_32d)
    # print("img1:", img1.shape)
    Image.fromarray(img1).show()
    # print("img3:",img3.shape)
    # Image.fromarray(img3).show()


def dataset_viz(root_dir, args):  # args.dir, args
    dataset = kitti_object(root_dir, args=args)

    ## load 2d detection results
    # objects2ds = read_det_file('box2d.list')
    cnt = 0
    for data_idx in range(args.ii, len(dataset)):
        if args.ind > 0:
            data_idx = args.ind
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx, args.seq)
        # objects2d = objects2ds[data_idx]
        if data_idx == len(dataset) - 1:
            print(cnt)
            quit()

        objects_pred = None
        if args.pred:
            objects_pred = dataset.get_pred_objects(data_idx, args.seq)
            if objects_pred == None:
                continue
        if objects_pred == None:
            print("no pred file")
            #objects_pred[0].print_object()

        objects_occ = None
        if args.show_occluded_lidar_with_depth or args.show_image_with_3D_boxes:
            objects_occ = dataset.get_occ_objects(data_idx)
            if objects_occ == None:
                continue

            if len(objects_pred) == len(objects_occ):
                continue

        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        # lidar_filename = os.path.join('D:/000001.bin')
        # pc_velo = np.fromfile(lidar_filename, dtype=np.float32)
        # pc_velo = pc_velo.reshape((-1, 4))
        # print(pc_velo)
        # pc_seg  = dataset.get_foreground_points(data_idx)
        # pc_occ  = dataset.get_occluded_points(data_idx).reshape(-1, 5)
        # if np.count_nonzero(pc_occ[:, 4]) < 50:
        #     continue
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        # print(data_idx, 'image shape: ', img.shape)
        # print(data_idx, 'velo  shape: ', pc_velo.shape)
        if args.depth:
            depth, is_exist = dataset.get_depth(data_idx)
            print(data_idx, 'depth shape: ', depth.shape)
        else:
            depth = None

        #depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        #depth_height, depth_width, depth_channel = img.shape

        #print(('Image shape: ', img.shape))

        if args.stat:
            stat_lidar_with_boxes(pc_velo, objects, calib)
            continue
        # print("======== Objects in Ground Truth ========")
        # n_obj = 0
        # for obj in objects:
        #     if obj.type != 'DontCare':
        #         print("=== {} object ===".format(n_obj+1))
        #         obj.print_object()
        #         n_obj += 1

        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            show_lidar_topview_with_boxes(pc_velo, objects, calib,
                                          objects_pred)

        #show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
        if args.show_image_with_boxes:
            # Draw 2d and 3d boxes on image
            show_image_with_boxes(img, objects, calib, objects_pred, True,
                                  depth)
        if args.show_image_with_3D_boxes:
            show_image_with_3D_boxes(img, objects, objects_pred, objects_occ,
                                     calib, True, depth)
        '''
        we only revised show_lidar_with_depth since we used the function.
        '''
        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            show_lidar_with_depth(pc_velo, objects, calib, args.img_fov, img_width, img_height, \
                 objects_pred, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label, gt=args.gt)
            #show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
            #    objects_pred, depth, img)
        '''
        '''
        if args.show_lidar_on_image:
            # Show LiDAR points on image.
            show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width,
                                img_height)
        if args.show_lidar_on_image_with_box:
            show_lidar_on_image_with_box(pc_velo[:, 0:3], img, calib,
                                         img_width, img_height, objects, True,
                                         depth, objects_pred)
        # if args.show_segmented_lidar_with_depth:
        #     show_segmented_lidar_with_depth(pc_velo, pc_seg, objects, calib, args.img_fov, img_width, img_height, \
        #          objects_pred, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label)
        if args.show_occluded_lidar_with_depth:
            # if np.count_nonzero(pc_occ[:, 4])==0:
            #     continue
            show_occluded_lidar_with_depth(pc_velo, pc_occ, objects, calib, args.img_fov, img_width, img_height, \
                 objects_pred, objects_occ, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label, save_velo_dir=save_path)
        # input_str=raw_input()

        mlab.close(all=True)
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
        # if input_str == "killall":
        #     break


def dataset_viz_save(root_dir, args):  # args.dir, args
    dataset = kitti_object(root_dir, args=args)
    if args.img_fov:
        velodyne_demo_dir = os.path.join(args.preddir, "velodyne_demo_img_fov")
    else:
        velodyne_demo_dir = os.path.join(args.preddir, "velodyne_demo")
    save_path = os.path.join(velodyne_demo_dir, args.seq)
    if not os.path.exists(velodyne_demo_dir):
        os.makedirs(velodyne_demo_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # MM_PER_VIEW = 180, 40, [12.0909996, -1.04700089, -2.03249991], 100.0 # azimuth, elevation, focalpoint, distance
    MM_PER_VIEW = 180, 40, [
        24, -1.04700089, -2.03249991
    ], 120.0  # azimuth, elevation, focalpoint, distance

    ## load 2d detection results
    # objects2ds = read_det_file('box2d.list')
    cnt = 0
    figure_bg = mlab.figure(figure=None,
                            bgcolor=(0, 0, 0),
                            fgcolor=None,
                            engine=None,
                            size=(1600, 1000))

    for data_idx in range(args.ii, len(dataset)):
        if args.ind > 0:
            if data_idx is not args.ind:
                continue
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx, args.seq)
        # objects2d = objects2ds[data_idx]
        if data_idx == len(dataset) - 1:  # what does the dataset length mean?
            print(cnt)
            quit()

        objects_pred_car = None
        objects_pred_ped = None
        objects_pred_cyc = None
        if args.pred:
            #if not dataset.isexist_pred_objects(data_idx):
            #    continue
            objects_pred_car = dataset.get_pred_objects(data_idx,
                                                        args.seq,
                                                        det_type=0)
            objects_pred_ped = dataset.get_pred_objects(data_idx,
                                                        args.seq,
                                                        det_type=1)
            objects_pred_cyc = dataset.get_pred_objects(data_idx,
                                                        args.seq,
                                                        det_type=2)
            if objects_pred_car == None:
                if objects_pred_ped == None:
                    if objects_pred_cyc == None:
                        continue

        if objects_pred_car == None:
            if objects_pred_ped == None:
                if objects_pred_cyc == None:
                    print("no pred file")
                    #objects_pred[0].print_object()

        objects_occ = None
        if args.show_occluded_lidar_with_depth or args.show_image_with_3D_boxes:
            objects_occ = dataset.get_occ_objects(data_idx)
            if objects_occ == None:
                continue

            if len(objects_pred) == len(objects_occ):
                continue

        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64

        lidar_filename_tmp = os.path.join(
            os.path.join(
                os.path.join(os.path.join(root_dir, args.split), args.lidar),
                args.seq), '%06d.bin' % (data_idx))
        # print("lidar_filename_tmp : ",lidar_filename_tmp)
        # exit()
        if os.path.exists(lidar_filename_tmp) == False:
            print("No velodyne data")
            continue
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        # lidar_filename = os.path.join('D:/000001.bin')
        # pc_velo = np.fromfile(lidar_filename, dtype=np.float32)
        # pc_velo = pc_velo.reshape((-1, 4))
        # print(pc_velo)
        # pc_seg  = dataset.get_foreground_points(data_idx)
        # pc_occ  = dataset.get_occluded_points(data_idx).reshape(-1, 5)
        # if np.count_nonzero(pc_occ[:, 4]) < 50:
        #     continue
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        # print(data_idx, 'image shape: ', img.shape)
        # print(data_idx, 'velo  shape: ', pc_velo.shape)
        if args.depth:
            depth, is_exist = dataset.get_depth(data_idx)
            print(data_idx, 'depth shape: ', depth.shape)
        else:
            depth = None

        #depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        #depth_height, depth_width, depth_channel = img.shape

        #print(('Image shape: ', img.shape))

        if args.stat:
            stat_lidar_with_boxes(pc_velo, objects, calib)
            continue
        # print("======== Objects in Ground Truth ========")
        # n_obj = 0
        # for obj in objects:
        #     if obj.type != 'DontCare':
        #         print("=== {} object ===".format(n_obj+1))
        #         obj.print_object()
        #         n_obj += 1

        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            show_lidar_topview_with_boxes(pc_velo, objects, calib,
                                          objects_pred)

        #show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
        if args.show_image_with_boxes:
            # Draw 2d and 3d boxes on image
            show_image_with_boxes(img, objects, calib, objects_pred, True,
                                  depth)
        if args.show_image_with_3D_boxes:
            show_image_with_3D_boxes(img, objects, objects_pred, objects_occ,
                                     calib, True, depth)
        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            show_lidar_with_depth(pc_velo, objects, calib, args.img_fov, img_width, img_height, \
                 objects_pred_car, objects_pred_ped, objects_pred_cyc, depth, img, constraint_box=args.const_box, save=args.save_depth,
                 pc_label=args.pc_label, gt=args.gt, save_velo_dir=save_path, idx=data_idx, fig=figure_bg, view_=MM_PER_VIEW)
            #  ref_fig=figure_bg, tar_fig=figure_tar, tar_ref_fig=figure_tar_ref)
            #show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
            #    objects_pred, depth, img)
        if args.show_lidar_on_image:
            # Show LiDAR points on image.
            show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width,
                                img_height)
        if args.show_lidar_on_image_with_box:
            show_lidar_on_image_with_box(pc_velo[:, 0:3], img, calib,
                                         img_width, img_height, objects, True,
                                         depth, objects_pred)
        # if args.show_segmented_lidar_with_depth:
        #     show_segmented_lidar_with_depth(pc_velo, pc_seg, objects, calib, args.img_fov, img_width, img_height, \
        #          objects_pred, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label)
        if args.show_occluded_lidar_with_depth:
            # if np.count_nonzero(pc_occ[:, 4])==0:
            #     continue
            show_occluded_lidar_with_depth(pc_velo, pc_occ, objects, calib, args.img_fov, img_width, img_height, \
                 objects_pred, objects_occ, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label, save_velo_dir=save_path)
        # input_str=raw_input()

        # mlab.close(all=True)
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
        # if input_str == "killall":
        #     break
    mlab.close(all=True)


def depth_to_lidar_format(root_dir, args):
    dataset = kitti_object(root_dir, args=args)
    for data_idx in range(len(dataset)):
        # Load data from dataset

        pc_velo = dataset.get_lidar(data_idx)[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        depth, is_exist = dataset.get_depth(data_idx)
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        print(data_idx, 'image shape: ', img.shape)
        print(data_idx, 'velo  shape: ', pc_velo.shape)
        print(data_idx, 'depth shape: ', depth.shape)
        #depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        #depth_height, depth_width, depth_channel = img.shape

        #print(('Image shape: ', img.shape))
        save_depth(data_idx, pc_velo, calib, args.img_fov, img_width, img_height, \
                       depth, constraint_box=args.const_box)
        input_str = raw_input()


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    objects = {}
    with open(det_filename, 'r') as f:
        for line in f.readlines():
            obj = utils.Object2d(line.rstrip())
            if obj.img_name not in objects.keys():
                objects[obj.img_name] = []
            objects[obj.img_name].append(obj)
        #objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]

    return objects


# def dataset_viz_video(root_dir, args):
#     self.root_dir = root_dir # args.dir = /SSD/Dataset/kitti_t_o
#     self.split = split #
#     print(root_dir, split)
#     self.split_dir = os.path.join(root_dir, split) # args.dir/split('training'/'testing')
#     # if split == 'training':
#     #     self.num_samples = 7481
#     # elif split == 'testing':
#     #     self.num_samples = 7518
#     # else:
#     #     print('Unknown split: %s' % (split))
#     #     exit(-1)
#     lidar_dir = 'velodyne'
#     depth_dir = 'depth'
#     pred_dir = None
#     occ_dir = None
#     if args is not None:
#         lidar_dir = args.lidar #
#         depth_dir = args.depthdir # no use
#         pred_dir  = args.preddir
#         occ_dir   = args.occdir # no use
#         occpoint_dir = args.occpointdir # no use
#         seq_id = args.seq

#     self.image_dir = os.path.join(self.split_dir, 'image_02')
#     self.image_dir_seq = os.path.join(os.path.join(self.split_dir, 'image_02'), seq_id)
#     self.label_dir =os.path.join(self.split_dir, 'label_02')
#     self.calib_dir = os.path.join(self.split_dir, 'calib')
#     self.calib_dir_seq = os.path.join(os.path.join(self.split_dir, 'calib'),'%s.txt'%seq_id)
#     self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
#     self.lidar_dir_seq = os.path.join(os.path.join(self.split_dir, lidar_dir), seq_id)
#     self.pred_dir  = pred_dir
#     self.pred_dir_seq  = os.path.join(pred_dir, seq_id)

#     dataset = kitti_object_video(self.image_dir_seq, self.lidar_dir_seq, self.calib_dir_seq)
#     print(len(dataset))
#     # for i in range(len(dataset)):
#     #     img = dataset.get_image(0)
#     #     pc = dataset.get_lidar(0)
#     #     Image.fromarray(img).show()
#     #     draw_lidar(pc)
#     #     raw_input()
#     #     pc[:,0:3] = dataset.get_calibration().project_velo_to_rect(pc[:,0:3])
#     #     draw_lidar(pc)
#     #     raw_input()
#     cnt = 0
#     for data_idx in range(args.ii, len(dataset)):
#         if args.ind>0:
#            data_idx=args.ind
#         # Load data from dataset
#         objects = dataset.get_label_objects(data_idx, args.seq)
#         # objects2d = objects2ds[data_idx]
#         if data_idx == len(dataset)-1:   # what does the dataset length mean?
#             print(cnt)
#             quit()

#         objects_pred = None
#         if args.pred:
#             #if not dataset.isexist_pred_objects(data_idx):
#             #    continue
#             objects_pred = dataset.get_pred_objects(data_idx, args.seq)
#             if objects_pred == None:
#                 continue
#             # print("__"*20)
#             # print("objects_pred : ",objects_pred)
#         if objects_pred == None:
#             print("no pred file")
#             #objects_pred[0].print_object()

#         n_vec = 4
#         if args.pc_label:
#             n_vec = 5

#         dtype=np.float32
#         if args.dtype64:
#             dtype=np.float64
#         pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:,0:n_vec]
#         # lidar_filename = os.path.join('D:/000001.bin')
#         # pc_velo = np.fromfile(lidar_filename, dtype=np.float32)
#         # pc_velo = pc_velo.reshape((-1, 4))
#         print(pc_velo)
#         # pc_seg  = dataset.get_foreground_points(data_idx)
#         # pc_occ  = dataset.get_occluded_points(data_idx).reshape(-1, 5)
#         # if np.count_nonzero(pc_occ[:, 4]) < 50:
#         #     continue
#         calib   = dataset.get_calibration(data_idx)
#         img     = dataset.get_image(data_idx)
#         img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_height, img_width, img_channel = img.shape
#         print(data_idx, 'image shape: ', img.shape)
#         print(data_idx, 'velo  shape: ', pc_velo.shape)
#         if args.depth:
#             depth, is_exist = dataset.get_depth(data_idx)
#             print(data_idx, 'depth shape: ', depth.shape)
#         else:
#             depth=None

#         #depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
#         #depth_height, depth_width, depth_channel = img.shape

#         #print(('Image shape: ', img.shape))

#         if args.stat:
#             stat_lidar_with_boxes(pc_velo, objects, calib)
#             continue
#         # print("======== Objects in Ground Truth ========")
#         # n_obj = 0
#         # for obj in objects:
#         #     if obj.type != 'DontCare':
#         #         print("=== {} object ===".format(n_obj+1))
#         #         obj.print_object()
#         #         n_obj += 1

#         # Draw 3d box in LiDAR point cloud
#         if args.show_lidar_topview_with_boxes:
#            # Draw lidar top view
#            show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred)

#         #show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
#         if args.show_image_with_boxes:
#             # Draw 2d and 3d boxes on image
#             show_image_with_boxes(img, objects, calib, objects_pred, True, depth)
#         if args.show_image_with_3D_boxes:
#             show_image_with_3D_boxes(img, objects, objects_pred, objects_occ, calib, True, depth)
#         if args.show_lidar_with_depth:
#             # Draw 3d box in LiDAR point cloud
#             show_lidar_with_depth(pc_velo, objects, calib, args.img_fov, img_width, img_height, \
#                  objects_pred, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label, gt=args.gt)
#             #show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
#             #    objects_pred, depth, img)
#         if args.show_lidar_on_image:
#             # Show LiDAR points on image.
#             show_lidar_on_image(pc_velo[:,0:3], img, calib, img_width, img_height)
#         if args.show_lidar_on_image_with_box:
#             show_lidar_on_image_with_box(pc_velo[:,0:3], img, calib, img_width, img_height, objects, True, depth, objects_pred)
#         # if args.show_segmented_lidar_with_depth:
#         #     show_segmented_lidar_with_depth(pc_velo, pc_seg, objects, calib, args.img_fov, img_width, img_height, \
#         #          objects_pred, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label)
#         if args.show_occluded_lidar_with_depth:
#             # if np.count_nonzero(pc_occ[:, 4])==0:
#             #     continue
#             show_occluded_lidar_with_depth(pc_velo, pc_occ, objects, calib, args.img_fov, img_width, img_height, \
#                  objects_pred, objects_occ, depth, img, constraint_box=args.const_box, save=args.save_depth, pc_label=args.pc_label)
#         input_str=raw_input()

#         mlab.close(all=True)
#         for proc in psutil.process_iter():
#             if proc.name() == "display":
#                 proc.kill()
#         if input_str == "killall":
#             break
#     return

if __name__ == '__main__':
    import mayavi.mlab as mlab
    from kitti_3d_visualization_utils import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    parser = argparse.ArgumentParser(description='PyTorch Training RPN')
    parser.add_argument('-d',
                        '--dir',
                        type=str,
                        default="/SSD/Dataset/kitti_t_o",
                        metavar='N',
                        help='input  (default: data/object)')
    parser.add_argument('-i',
                        '--ind',
                        type=int,
                        default=0,
                        metavar='N',
                        help='input index')
    parser.add_argument('-ii',
                        type=int,
                        default=0,
                        metavar='N',
                        help='starting input index')
    parser.add_argument('-g', '--gt', action='store_true', help='show gt')
    parser.add_argument('-p',
                        '--pred',
                        action='store_true',
                        help='show predict results')
    parser.add_argument('-sf',
                        '--save_fig',
                        action='store_true',
                        help='show predict results')
    parser.add_argument('-s',
                        '--stat',
                        action='store_true',
                        help=' stat the w/h/l of point cloud in gt bbox')
    parser.add_argument('-l',
                        '--lidar',
                        type=str,
                        default="velodyne",
                        metavar='N',
                        help='velodyne dir  (default: velodyne)')
    parser.add_argument('-sp',
                        '--split',
                        type=str,
                        default="training",
                        metavar='N',
                        help='training or testing  (default: training)')
    parser.add_argument('-e',
                        '--depthdir',
                        type=str,
                        default="depth",
                        metavar='N',
                        help='depth dir  (default: depth)')
    parser.add_argument('-r',
                        '--preddir',
                        type=str,
                        default="./results",
                        metavar='N',
                        help='predicted boxes  (default: pred)')
    parser.add_argument('-sq',
                        '--seq',
                        type=str,
                        default="0000",
                        metavar='N',
                        help='seq  (default: 0000)')
    parser.add_argument('-th_c',
                        '--thrs_car',
                        type=str,
                        default="0.66",
                        metavar='N',
                        help='predicted boxes  (default: threshold)')
    parser.add_argument('-th_p',
                        '--thrs_ped',
                        type=str,
                        default="0.2",
                        metavar='N',
                        help='thrs_ped  (default: threshold)')
    parser.add_argument('-th_cl',
                        '--thrs_cyc',
                        type=str,
                        default="0.15",
                        metavar='N',
                        help='thrs_cyc  (default: threshold)')
    parser.add_argument('--occdir',
                        type=str,
                        default="PR_OUT/0.82_98",
                        metavar='N',
                        help='occ_dir  (default: occ_pred)')
    parser.add_argument('--occpointdir',
                        type=str,
                        default="PR_OUT/occluded_points",
                        metavar='N',
                        help='occ_predicted points')
    parser.add_argument('--gen_depth',
                        action='store_true',
                        help='generate depth')
    parser.add_argument('--vis', action='store_true', help='show images')
    parser.add_argument('--depth', action='store_true', help='load depth')
    parser.add_argument('--img_fov',
                        action='store_true',
                        help='front view mapping')
    parser.add_argument('--const_box',
                        action='store_true',
                        help='constraint box')
    parser.add_argument('--save_depth',
                        action='store_true',
                        help='save depth into file')
    parser.add_argument('--pc_label',
                        action='store_true',
                        help='5-verctor lidar, pc with label')
    parser.add_argument('--dtype64',
                        action='store_true',
                        help='for float64 datatype, default float64')
    parser.add_argument('--show_lidar_on_image',
                        action='store_true',
                        help='project lidar on image')
    parser.add_argument('--show_lidar_with_depth',
                        action='store_true',
                        help='--show_lidar, depth is supported')
    parser.add_argument('--show_image_with_boxes',
                        action='store_true',
                        help='show lidar')
    parser.add_argument('--show_image_with_3D_boxes',
                        action='store_true',
                        help='show lidar')
    parser.add_argument('--show_lidar_topview_with_boxes',
                        action='store_true',
                        help='show lidar topview')
    parser.add_argument('--show_lidar_on_image_with_box',
                        action='store_true',
                        help='show lidar on image with box')
    parser.add_argument('--show_segmented_lidar_with_depth',
                        action='store_true',
                        help='show segmented lidar')
    parser.add_argument('--show_occluded_lidar_with_depth',
                        action='store_true',
                        help='show occluded lidar')

    args = parser.parse_args()
    # print("args.preddir : \n", args.preddir)
    if args.pred:
        # assert os.path.exists(args.dir+"/training/%s" % args.preddir)
        assert os.path.exists(args.preddir)
        # "/SSD/Github/AB3DMOT/results/car_3d_det_val_thres/trk_id_included_dy_st_term_1_thres_0.66

    # pdb.set_trace()

    if args.vis:
        if args.save_fig == False:
            dataset_viz(args.dir, args)
        else:
            dataset_viz_save(args.dir, args)
    if args.gen_depth:
        depth_to_lidar_format(args.dir, args)
