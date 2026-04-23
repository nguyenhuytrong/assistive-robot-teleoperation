# Dockerfile chuẩn ROS2 Humble Desktop Full
FROM osrf/ros:humble-desktop-full

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    python3-pip \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y \
    ros-humble-teleop-twist-keyboard \
    ros-humble-ign-ros2-control \
    ros-humble-urdf \
    ros-humble-urdf-tutorial \
    ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y \
    python3-colcon-common-extensions \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    python3-rosdep \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libhidapi-hidraw0 \
    libhidapi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pydualsense transform3d 

RUN rosdep update --include-eol-distros || true

WORKDIR /ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "if [ -f /ros2_ws/install/local_setup.bash ]; then source /ros2_ws/install/local_setup.bash; fi" >> /root/.bashrc
    
CMD ["bash"]