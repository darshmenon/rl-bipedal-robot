# scripts/install_dependencies.sh
#!/bin/bash

echo "Installing dependencies for RL Bipedal Walking..."

# Update system
sudo apt update

# Install ROS2 Humble dependencies
sudo apt install -y \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-rviz2 \
    ros-humble-xacro \
    ros-humble-tf-transformations \
    python3-pip \
    python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

echo "Dependencies installed successfully!"