# 🌟 [RA-L 2023] Enhancing Generalizable 6D Pose Tracking of an In-Hand Object with Tactile Sensing 🌟

[RA-L 2023] Official repository of "Enhancing Generalizable 6D Pose Tracking of an In-Hand Object with Tactile Sensing".

### 📄[Paper](https://arxiv.org/pdf/2210.04026.pdf) | 🎥[Dataset](https://1drv.ms/f/s!Ap-t7dLl7BFUaQ794lX1srGnwlQ?e=JgohXw)

#### Authors

Yun Liu*, Xiaomeng Xu*, Weihang Chen, Haocheng Yuan, He Wang, Jing Xu, Rui Chen, Li Yi

## Dataset

### Data Download

Please download the whole dataset (184GB) or a preview version (5GB) from [One Drive](https://1drv.ms/f/s!Ap-t7dLl7BFUaQ794lX1srGnwlQ?e=JgohXw).

To merge the files and get the zip file of the whole dataset, please use: ```cat Overall_Dataset_Split* >Overall_Dataset.zip```. After unzipping the dataset, merge it with ```object_tactile_point_clouds.zip``` to integrate pre-processed object points contacting the sensor surfaces. Finally, the folder of each video sequence (e.g. ```0907_bottle1_1_003```) should include 17 sub-folders.

### Data Usage

Please refer to ```docs/dataset_documentation.md``` for details of our dataset.

If you have any questions about the dataset, please contact ```yun-liu22@mails.tsinghua.edu.cn```.

## Code

We provide the implementation of TEG-Track(ShapeAlign) and the kinematics-only method.

### Installation

Our code is tested on Ubuntu 20.04 with a single NVIDIA GeForce RTX 3090. The Driver version is 535.171.04. The CUDA version is 12.2.

To set up the envoronment, please refer to the following commands:

```x
conda create -n tegtrack python=3.7
conda activate tegtrack
<install PyTorch, we use PyTorch v1.10.1>
pip install -r requirements.txt
```

### Kinematics-only Method

Please use the following commands:

```x
cd code/tracking
python kinematics_only.py --dataset_dir <the path of the dataset> --tracking_save_sir <the path to save the tracking results> --kinematics_save_dir <the path to save the computed object kinematic states>
```

### TEG-Track(ShapeAlign)

ShapeAlign is a simple vision-based tracking method designed by ourselves, which first completes the object point cloud observed in the first frame to an integral object model, and then track the object pose by fitting the visual observations to the object model. The completion is done by pre-trained PoinTr networks. We provide the point cloud completion results in our dataset (```preprocessed_completed_point_cloud_data.zip```).

To run TEG-Track(ShapeAlign), please first download ```preprocessed_completed_point_cloud_data.zip``` and unzip it.

Then, please use the following commands:

```x
cd code/tracking
python TEG_Track_ShapeAlign.py --dataset_dir <the path of the dataset> --completed_point_cloud_dir <the path of the completed point cloud data> --tracking_save_sir <the path to save the tracking results> --kinematics_save_dir <the path to load the computed object kinematic states (if provided)>
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
