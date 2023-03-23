# CC_AF
This repo only showcases the perception block of autonomous driving pipeline - **Visual Computing** - dedicated to competition 
of Carolo Cup according to the confidentiality agreement.

## Visual Computing
This block of visual computing is dedicated to lane detection which involves detecting and tracking the land markers which 
are specific to the Carolo Cup.

### Dependencies
#### ROS 2 Foxy
Please refer to the [official guide for installation](https://docs.ros.org/en/foxy/Installation.html).
#### rviz
```shell
sudo apt-get install libzbar-dev libzbar0 ros-foxy-rviz2
```
#### ROS2 Wrapper for Intel Realsense D400
It's recommended to always use the newest stable version. Please refer to [this link](https://github.com/IntelRealSense/realsense-ros/wiki/Build-with-local-librealsense2)
as installation guide. After the package has been installed and built, it only needs to be added to the `.bashrc` configuration file.


### Build and Install
```shell
# Create a workspace
mkdir -p vc-ws/src
cd vc-ws/src
git clone git@github.com:lenardxu/CC_AF.git
# Or via https 
git clone https://github.com/lenardxu/CC_AF.git
cd ../
# Build 
colcon build --symlink-install
# Install
. install/local_setup.bash
```

### Run 
Launch the single ROS node `sliding_window` for lane detection and simplistic reference trajectory generation by
```shell
ros2 launch cc_visual_computing sliding_window
```
Or the single ROS node `probabilistic_tracking` for lane detection and tracking and simplistic reference trajectory generation by
```shell
ros2 launch cc_visual_computing probabilistic_tracking
```

### Comments
Currently, the process of computing homography is not included in this repo. However, there are several resources you may refer to:
* [OpenCV guide](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) and 
[related discussion](https://stackoverflow.com/questions/48576087/birds-eye-view-perspective-transformation-from-camera-calibration-opencv-python)
* Implementation adopted whose foundations are summarized as shown [here](./docs/IPM.pdf) (german).