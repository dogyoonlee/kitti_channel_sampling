# python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --slice 2
# python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --slice 4
# python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --slice 8
# python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --slice 16
# python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --slice 32

python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --sparse_lidar_save_path ./spar_lidar_res_det/sparse_lidar_det_height_cut_top_3_bottom_3 --slice 2 --multi_lidar --slice_height --slice_except_top 3 --slice_except_bottom 3
python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --sparse_lidar_save_path ./spar_lidar_res_det/sparse_lidar_det_height_cut_top_3_bottom_3 --slice 4 --multi_lidar --slice_height --slice_except_top 3 --slice_except_bottom 3
python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --sparse_lidar_save_path ./spar_lidar_res_det/sparse_lidar_det_height_cut_top_3_bottom_3 --slice 8 --multi_lidar --slice_height --slice_except_top 3 --slice_except_bottom 3
python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --sparse_lidar_save_path ./spar_lidar_res_det/sparse_lidar_det_height_cut_top_3_bottom_3 --slice 16 --multi_lidar --slice_height --slice_except_top 3 --slice_except_bottom 3
python kitti_sparsify_detection.py --data_dir ./data_det --multi_lidar --det --sparse_lidar_save_path ./spar_lidar_res_det/sparse_lidar_det_height_cut_top_3_bottom_3 --slice 32 --multi_lidar --slice_height --slice_except_top 3 --slice_except_bottom 3
