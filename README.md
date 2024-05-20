# 🌟 [RAL 2023] Enhancing Generalizable 6D Pose Tracking of an In-Hand Object with Tactile Sensing 🌟

[RAL 2023] Official repository of "Enhancing Generalizable 6D Pose Tracking of an In-Hand Object with Tactile Sensing".

### 📄[Paper](https://arxiv.org/pdf/2210.04026.pdf) | 🎥[Dataset](https://1drv.ms/f/s!Ap-t7dLl7BFUaQ794lX1srGnwlQ?e=JgohXw)

#### Authors

Yun Liu*, Xiaomeng Xu*, Weihang Chen, Haocheng Yuan, He Wang, Jing Xu, Rui Chen, Li Yi

## Dataset

### Data Download

Please download the whole dataset (184GB) or a preview version (5GB) from [One Drive](https://1drv.ms/f/s!Ap-t7dLl7BFUaQ794lX1srGnwlQ?e=JgohXw).

To merge the files and get the zip file of the whole dataset, please use: ```cat Overall_Dataset_Split* >Overall_Dataset.zip```.

### Data Usage

Please refer to ```docs/dataset_documentation.md``` for details of our dataset.

If you have any questions about the dataset, please contact ```yun-liu22@mails.tsinghua.edu.cn```.

## Code

We provide the implementation of TEG-Track(ShapeAlign) and the kinematics-only method.

### Installation

TODO

### Kinematic-only Method

Please use the following commands:

```x
cd code/tracking
python kinematics_only.py --dataset_dir <the path of the dataset> --tracking_save_sir <the path to save the tracking results> --kinematics_save_dir <the path to save the computed object kinematic states>
```

### TEG-Track(ShapeAlign)

Please use the following commands:

```x
cd code/tracking
python TEG_Track_ShapeAlign.py --dataset_dir <the path of the dataset> --completed_point_cloud_dir <the path of completed object point clouds of the first frame> --tracking_save_sir <the path to save the tracking results> --kinematics_save_dir <the path to load the computed object kinematic states (if provided)>
```

### Training the Learning-based Velocity Predictor

Please use the following commands:

```x
cd code/tactile_learning
python main.py --dataset_dir <the path of the dataset> --save_dir <the path to save the checkpoints>
```

## Citation

If you find our work useful in your research, please consider citing:

```
@ARTICLE{10333330,
  author={Liu, Yun and Xu, Xiaomeng and Chen, Weihang and Yuan, Haocheng and Wang, He and Xu, Jing and Chen, Rui and Yi, Li},
  journal={IEEE Robotics and Automation Letters}, 
  title={Enhancing Generalizable 6D Pose Tracking of an In-Hand Object With Tactile Sensing}, 
  year={2024},
  volume={9},
  number={2},
  pages={1106-1113},
  keywords={Visualization;Kinematics;Sensors;Robots;Tracking;Three-dimensional displays;Tactile sensors;Force and tactile sensing;sensor fusion;visual tracking},
  doi={10.1109/LRA.2023.3337690}}
```

## Email

If you have any questions, please contact ```yun-liu22@mails.tsinghua.edu.cn```.
