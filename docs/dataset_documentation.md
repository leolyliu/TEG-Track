# Dataset for "Enhancing Generalizable 6D Pose Tracking of an In-Hand Object with Tactile Sensing"

Please download our dataset via the following links:

* TODO

### Overview

Our dataset is a large-scale visual-tactile in-hand object pose tracking dataset covering 5 rigid object categories. Key information and statistics for this dataset are listed as follows:

* Frame number for each video sequence: 401
* Sequence number for each object category:
    + Camera: 44
    + Can: 44
    + Bottle: 43
    + Mug: 36
    + Bowl: 32
* Visual sensor image resolution: 1280x720
* Tactile sensor image resolution: 640x480

Please check our paper for more information.

### Data Organization

The dataset is organized as:

```
./
|--{object_category}
|--metadata.pkl
|--training_sequence_names.json
|--test_sequence_names.json

./{object_category}/{sequence_name}/
|--camera_params
    |--visual_camera_intrinsic.txt
    |--visual_camera_extrinsic.txt
|--rgb
    |--*.png
|--depth
    |--*.png
|--visual_point_cloud
    |--*.ply
|--object_mask
    |--*.png
|--left_rgb
    |--*.png
|--left_depth
    |--*.png
|--left_pad_pose
    |--*.txt
|--left_marker_point_positions
    |--*.txt
|--right_rgb
    |--*.png
|--right_depth
    |--*.png
|--right_pad_pose
    |--*.txt
|--right_marker_point_positions
    |--*.txt
|--object_pose
    |--*.npy
|--precomputed_object_contact_information
    |--*.pkl
```

### File Definitions

#### (1) General Information

* ```./metadata.pkl```: A pickle file indicating the frame number, the object name, and the object scale for each video sequence.
    + The unit of the ```object_scale``` parameter is meter (m). The scale for each object is measured under its canonicalized frame.
* ```./training_sequence_names.json```: Sequences to train the learning-based velocity predictor.
* ```./test_sequence_names.json```: Sequences to evaluate performances of object pose tracking methods.

#### (2) Visual Sensor Signals

* ```./{object_category}/{sequence_name}/rgb```: Color images captured from the visual sensor.
* ```./{object_category}/{sequence_name}/depth```: Depth images captured from the visual sensor.
* ```./{object_category}/{sequence_name}/visual_point_cloud```: A downsampled object point cloud under the world coordinate system for each frame.
* ```./{object_category}/{sequence_name}/object_mask```: Object 2D mask annotation for each frame.
    + The object pixels are shown as the red (RGB : (255, 0, 0) ) color.
    + Other pixels are shown as the black (RGB : (0, 0, 0) ) color.

#### (3) Left Tactile Sencor Signals

Our robot gripper is equipped with two tactile sensors at the positions of its left and right pads. This subsection defines signals from the tactile sensor at the left pad.

* ```./{object_category}/{sequence_name}/left_rgb```: Color images captured from the left tactile sensor.
* ```./{object_category}/{sequence_name}/left_depth```: Depth images captured from the left tactile sensor.
* ```./{object_category}/{sequence_name}/left_pad_pose```: The pose of the left pad for each frame.
* ```./{object_category}/{sequence_name}/left_marker_point_positions```: Precomputed marker point pixels of the left tactile sensor for each frame.
    + Note that we first crop a 320x320 image patch centered at (320, 240) of the original tactile image, and then compute marker positions at the image patch. The pixels are defined on the image patch.

#### (4) Right Tactile Sensor Signals

Definitions for signals captured from the tactile sensor at the right pad are very similar to the previous subsection.

* ```./{object_category}/{sequence_name}/right_rgb```: Color images captured from the right tactile sensor.
* ```./{object_category}/{sequence_name}/right_depth```: Depth images captured from the right tactile sensor.
* ```./{object_category}/{sequence_name}/right_pad_pose```: The pose of the right pad for each frame.
* ```./{object_category}/{sequence_name}/right_marker_point_positions```: Precomputed marker point pixels of the right tactile sensor for each frame.

#### (5) Object Pose Annotations

* ```./{object_category}/{sequence_name}/object_pose```: The object pose under the world coordinate system for each frame.

#### (6) Precomputed Object Contact Information

* ```./{object_category}/{sequence_name}/precomputed_object_contact_information```: Precomputed contact information between the in-hand object and two tactile sensors for each frame.
    + ```left_contact_points``` and ```right_contact_points```: 3D positions of contact points in the world coordinate system.
    + ```left_mean_contact_velocities``` and ```right_mean_contact_velocities```: 3D velocities of contact points in the world coordinate system.
    + ```dt```: The time difference between adjacent frames. The unit is second (s).
