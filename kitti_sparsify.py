"""
sample usage of original code from Pseudo-Lidar++

python ./src/preprocess/kitti_sparsify.py --pl_path  ./results/ssdn_kitti_train_set/pseudo_lidar_trainval/  \
    --sparse_pl_path  ./results/sdn_kitti_train_set/pseudo_lidar_trainval_sparse/
    
"""

import argparse
import os

import numpy as np
import tqdm


def pto_rec_map(velo_points, H=64, W=512, D=800):
    # depth, width, height
    valid_inds = (
        (velo_points[:, 0] < 80)
        & (velo_points[:, 0] >= 0)
        & (velo_points[:, 1] < 50)
        & (velo_points[:, 1] >= -50)
        & (velo_points[:, 2] < 1)
        & (velo_points[:, 2] >= -2.5)
    )
    velo_points = velo_points[valid_inds]

    x, y, z, i = (
        velo_points[:, 0],
        velo_points[:, 1],
        velo_points[:, 2],
        velo_points[:, 3],
    )
    x_grid = (x * D / 80.0).astype(int)
    x_grid[x_grid < 0] = 0
    x_grid[x_grid >= D] = D - 1

    y_grid = ((y + 50) * W / 100.0).astype(int)
    y_grid[y_grid < 0] = 0
    y_grid[y_grid >= W] = W - 1

    z_grid = ((z + 2.5) * H / 3.5).astype(int)
    z_grid[z_grid < 0] = 0
    z_grid[z_grid >= H] = H - 1

    depth_map = -np.ones((D, W, H, 4))
    depth_map[x_grid, y_grid, z_grid, 0] = x
    depth_map[x_grid, y_grid, z_grid, 1] = y
    depth_map[x_grid, y_grid, z_grid, 2] = z
    depth_map[x_grid, y_grid, z_grid, 3] = i
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def pto_ang_map(
    velo_points,
    H=64,
    W=512,
    slice=1,
    slice_height=False,
    slice_except_top=0,
    slice_except_bottom=0,
    multi_ratio=4,
):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """
    # x: front, y: left, z: up
    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)
    x, y, z, i = (
        velo_points[:, 0],
        velo_points[:, 1],
        velo_points[:, 2],
        velo_points[:, 3],
    )
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.0) - np.arcsin(y / r)  # 45 - angle from x axis to y axis
    phi_ = (phi / dphi).astype(int)  # y axis pixel value
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1  # handling exception
    theta = np.radians(2.0) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = -np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i

    if args.sampling_num3:
        slice_except_top = int((64 - int(64 / int(slice / 2))) / 2)
        slice_except_bottom = slice_except_top
        slice_height = True

    if args.multi_lidar:
        if slice_height is True:
            depth_map_tmp = depth_map[slice_except_top : (H - slice_except_bottom)]
            depth_map_tmp = depth_map_tmp[0 :: int(slice / 2), :, :]
        else:
            depth_map_tmp = depth_map[0 :: int(slice / 2), :, :]
        for i in range(W):
            # if i > (np.radians(67.5) / dphi).astype(int) and i < (
            # np.radians(112.5) / dphi).astype(int):
            # if i > W / 4 and i < (3 * W) / 4:
            if i < W / multi_ratio:
                depth_map_tmp[1::2, i, :] = 99999
            elif i > ((multi_ratio - 1) * W) / multi_ratio:
                depth_map_tmp[0::2, i, :] = 99999

            depth_map = depth_map_tmp
    else:
        if slice_height is True:
            depth_map_tmp = depth_map[slice_except_top : (H - slice_except_bottom)]
            depth_map_tmp = depth_map_tmp[0 :: int(slice), :, :]
        else:
            depth_map_tmp = depth_map[0 :: int(slice), :, :]
        depth_map = depth_map_tmp
    depth_map = depth_map.reshape((-1, 4))

    if args.multi_lidar:
        remove_idx = list()
        for i in range(len(depth_map)):
            if depth_map[i][0] == 99999:
                remove_idx.append(i)
        remove_idx = np.array(remove_idx)
        depth_map = np.delete(depth_map, remove_idx, axis=0)

    depth_map = depth_map[depth_map[:, 0] != -1.0]
    # print('depth_map: ', depth_map)
    return depth_map


def gen_sparse_points(lidar_data_path, args):
    pc_velo = np.fromfile(lidar_data_path, dtype=np.float32).reshape((-1, 4))

    # depth, width, height
    valid_inds = (
        (pc_velo[:, 0] < 120)
        & (pc_velo[:, 0] >= 0)
        & (pc_velo[:, 1] < 50)
        & (pc_velo[:, 1] >= -50)
        & (pc_velo[:, 2] < 1.5)
        & (pc_velo[:, 2] >= -2.5)
    )
    pc_velo = pc_velo[valid_inds]
    # print('pc_velo: ', pc_velo)
    # print('pc_velo shape: ', pc_velo.shape)
    return pto_ang_map(
        pc_velo,
        H=args.H,
        W=args.W,
        slice=args.slice,
        slice_height=args.slice_height,
        slice_except_bottom=args.slice_except_bottom,
        slice_except_top=args.slice_except_top,
        multi_ratio=args.multi_ratio,
    )


def gen_sparse_points_seq(lidar_path, outputfolder, seq):
    os.makedirs(outputfolder, exist_ok=True)
    data_idx_list = sorted(
        [x.strip() for x in os.listdir(lidar_path) if x[-3:] == "bin"]
    )

    for data_idx in tqdm.tqdm(data_idx_list, desc="sequence " + seq):
        sparse_points = gen_sparse_points(os.path.join(lidar_path, data_idx), args)
        # print('processed pc_velo: ', sparse_points)
        # print('processed pc_velo shape: ', sparse_points.shape)
        # input('Time')
        sparse_points = sparse_points.astype(np.float32)
        sparse_points.tofile(f"{outputfolder}/{data_idx}")


def gen_sparse_points_all(args):
    if args.data_mode == "training":
        lidar_path = os.path.join(
            os.path.join(args.data_dir, args.data_mode), "velodyne"
        )
    else:
        lidar_path = os.path.join(
            os.path.join(os.path.join(args.data_dir, args.data_mode), "velodyne"),
            "velodyne",
        )
    if args.multi_lidar:
        outputfolder = os.path.join(
            os.path.join(
                os.path.join(
                    args.sparse_lidar_save_path,
                    str(int(64 / int(args.slice))) + "channel_X2",
                ),
                args.data_mode,
            ),
            "velodyne",
        )
    else:
        outputfolder = os.path.join(
            os.path.join(
                os.path.join(
                    args.sparse_lidar_save_path,
                    str(int(64 / int(args.slice))) + "channel",
                ),
                args.data_mode,
            ),
            "velodyne",
        )
    # outputfolder = args.sparse_lidar_save_path
    for seq in sorted(os.listdir(lidar_path)):
        lidar_path_seq = os.path.join(lidar_path, seq)
        outputfolder_seq = os.path.join(outputfolder, seq)
        os.makedirs(outputfolder_seq, exist_ok=True)
        gen_sparse_points_seq(lidar_path_seq, outputfolder_seq, seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate sparse LiDAR points")
    # parser.add_argument('--pl_path',
    #                     default='./scratch/datasets',
    #                     help='pseudo-lidar path')
    parser.add_argument("--data_dir", default="./data/kitti_t_o", help="lidar path")
    parser.add_argument(
        "--data_mode", type=str, default="training", choices=["training", "testing"]
    )
    parser.add_argument(
        "--sparse_lidar_save_path",
        default="./sparse_lidar_results",
        help="sparsed lidar path",
    )
    parser.add_argument("--slice", default=1, type=int)
    parser.add_argument("--H", default=64, type=int)
    parser.add_argument("--W", default=512, type=int)
    parser.add_argument("--D", default=700, type=int)
    parser.add_argument(
        "--multi_lidar", action="store_true", help="mimic the multi lidar environment"
    )
    parser.add_argument(
        "--slice_height", action="store_true", help="slice along height"
    )
    parser.add_argument("--slice_except_top", default=0, type=int)
    parser.add_argument("--slice_except_bottom", default=0, type=int)
    parser.add_argument("--multi_ratio", default=4, type=int)
    parser.add_argument(
        "--sampling_num3", action="store_true", help="sampling number 3"
    )
    # parser.add_argument('--channel', default=4, type=int)
    args = parser.parse_args()

    print("Generate sparse LiDAR points")
    print("Data mode: ", args.data_mode)
    if args.multi_lidar:
        print("Channel: 64 -> ", int(64 / int(args.slice)), "X 2 ")
    else:
        print("Channel: 64 -> ", int(64 / int(args.slice)))
    gen_sparse_points_all(args)
