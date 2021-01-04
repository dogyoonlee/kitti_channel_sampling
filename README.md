# kitti_channel_sampling

Channel sampling code for **kitti tracking benchmark dataset** which is composed of point clouds.

## Usage

### Generate Sparse Single Lidar

```
$ python kitti_sparsify.py --slice $slice_num --data_mode training
$ python kitti_sparsify.py --slice $slice_num --data_mode testing
```

### Generate Sparse Multi Lidar

```
$ python kitti_sparsify.py --slice $slice_num --data_mode training --multi_lidar
$ python kitti_sparsify.py --slice $slice_num --data_mode testing --multi_lidar
```

### Visualize Original Kitti Data

```
$ python kitti_visualize_raw.py --seq $seq_num --vis --show_lidar_with_depth --img_fov
```

### Visualize Single Downsampled Kitti Data

```
$ python kitti_visualize_raw.py --sparse --seq 0000 --vis --show_lidar_with_depth --sparse_dir $sparse_lidar_directory
```

### Visualize Multi Downsampled Kitti Data

```
$ python kitti_visualize_raw.py --sparse --seq 0000 --vis --show_lidar_with_depth --sparse_dir $sparse_lidar_directory
```


<!-- [[Paper]](paper_address) -->

<!-- ## Overview
`RSMix` generates the virtual sample from each part of the two point cloud samples by mixing them without shape distortion. It effectively generalize the deep neural network model and achieve remarkable performance for shape classification. -->

<!-- <img src='./rsmix_pipeline.png' width=800> -->
<!-- 
## Implementation

### RSMix on PointNet++

* [RSMix-PointNet++(TensorFlow)](./pointnet2_rsmix)

### RSMix on DGCNN

* [RSMix-DGCNN(PyTorch)](./dgcnn_rsmix)


## License

MIT License


## Acknowledgement
The structure of this codebase is borrowed from 
[PointNet++](https://github.com/charlesq34/pointnet2/) and [DGCNN-PyTorch](https://github.com/WangYueFt/dgcnn/tree/master/pytorch).


### Citation
If you find our work useful in your research, please consider citing:

        Our_citation -->
