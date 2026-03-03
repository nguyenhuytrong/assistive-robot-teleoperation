FROM osrf/ros:humble-desktop-full


RUN apt-get update && apt-get install -y \
    ros-humble-teleop-twist-keyboard \
    ros-humble-ign-ros2-control \
    ros-humble-urdf \
    ros-humble-urdf-tutorial \
    ros-humble-xacro \
    python3-colcon-common-extensions \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    python3-rosdep \
    python3-pip \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libhidapi-hidraw0 \
    libhidapi-dev \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install pydualsense

RUN rosdep update --include-eol-distros || true


WORKDIR /ros2_ws


RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "if [ -f /ros2_ws/install/local_setup.bash ]; then source /ros2_ws/install/local_setup.bash; fi" >> /root/.bashrc

CMD ["bash"]