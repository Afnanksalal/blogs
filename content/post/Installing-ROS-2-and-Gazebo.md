---
title: "A Robotics Beginner's Guide: Installing ROS 2 and Gazebo"
date: 2025-9-24
id: 10
author: "Afnan K Salal"
authorGithub: "https://github.com/afnanksalal"
tags:
  - ROS 2
  - Humble
  - Gazebo
  - Robotics
  - Installation
  - Ubuntu
  - Linux
  - Robotics Simulation
---

# Getting Your Computer Ready for Robotics

Welcome to the world of robotics. If you're building a robot, whether it's a physical one or a virtual one in a simulation, you need the right tools. The two most important are **ROS 2**, the software framework that makes everything talk to each other, and **Gazebo**, a powerful simulator where you can test your robots in a virtual world. This guide will walk you through setting up both of them on your computer.

First, let's make sure your system is ready. A simple but important step is to check your language settings. This helps prevent strange errors with certain packages later on.

First, check your current locale:

```bash
locale # check for UTF-8
````

If the settings don't show `en_US.UTF-8`, you need to set them. Here are the commands to do that:

```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale # verify settings
```

This first command makes sure you have the `locales` package, which is what you use to manage language settings. Then, we generate the specific locale file we need. The `update-locale` command applies this change system-wide. The `export` command makes sure the change is active for your current terminal session. After you're done, the final `locale` command should show that everything is set correctly.

---

# Installing ROS 2: The Brain of the Operation

Now we'll install **ROS 2 Humble**. The process involves adding the official ROS 2 software repository so your computer knows where to get the packages.

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools
source /opt/ros/humble/setup.bash
```

This block of code might look a bit complex, but it's a standard way to install software from a specific source. We install **curl** to download a small file that tells your system where to find ROS 2. The `export` command gets the latest version number from the internet and saves it, and then the `curl` command downloads the correct file. `sudo dpkg -i` installs that file, adding the ROS 2 repository to your computer's list of software sources.

After that, we run `sudo apt update` to refresh the list of available packages. `sudo apt upgrade` is a good practice to get any new updates for your existing software.

`sudo apt install ros-humble-desktop` installs the full **ROS 2 Desktop package**, which includes everything you need to get started: core libraries, development tools, and visualization tools. `sudo apt install ros-dev-tools` gives you extra command line tools for managing your ROS projects.

The last command, `source /opt/ros/humble/setup.bash`, is very important. It sets up your shell so it can find all the ROS 2 commands. You'll need to run this command every time you open a new terminal window to work on a ROS project.

---

# Adding Gazebo for Simulation

Next, we'll install **Gazebo**, the simulator where you can test your robot models and code without needing a physical robot.

```bash
sudo apt-get update
sudo apt-get install lsb-release gnupg
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install ignition-fortress
sudo apt install ros-humble-gazebo*
```

Just like with ROS 2, we first need to add Gazebo's software repository. The `sudo curl` command downloads a key that verifies the software packages are legitimate. The `echo "deb..."` command then adds the Gazebo repository to your system's source list.

After another `sudo apt-get update`, you can install Gazebo itself with `sudo apt-get install ignition-fortress`. Gazebo has been rebranded as **Ignition**, and *Fortress* is its latest version. `sudo apt install ros-humble-gazebo*` then installs the specific ROS 2 packages that let ROS and Gazebo talk to each other. This is crucial for running your robot code inside the simulation.

---

# Installing Tools for Building Robots

Now that you have the main systems in place, let's install some essential packages that will help you build and control your robots.

```bash
sudo apt install ros-humble-rqt*
sudo apt install ros-humble-moveit*
sudo apt install ros-humble-joint-state-publisher*
sudo apt install ros-humble-launch-param-builder
sudo apt install ros-humble-parameter-traits
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-ros2-controllers
sudo apt install ros-humble-controller-interface
sudo apt install ros-humble-joint-trajectory-controller
sudo apt install ros-humble-joint-state-broadcaster
sudo apt install ros-humble-gripper-controllers
sudo apt install ros-humble-xacro
sudo apt install ros-humble-realtime-tools
sudo apt install ros-humble-hardware-interface
sudo apt install ros-humble-control-toolbox
sudo apt install ros-humble-filters
sudo apt install ros-humble-ros2bag
sudo apt install ros-humble-plotjuggler*
```

This long list of commands installs a whole toolbox of packages. Here's a quick rundown of what they do:

* **RQt (ros-humble-rqt\*)**: A set of visual tools for debugging. It's great for seeing what your robot is doing in real-time.
* **MoveIt (ros-humble-moveit\*)**: For controlling robot arms. It helps you with things like planning paths for the arm to move and avoiding obstacles.
* **ROS 2 Control (ros-humble-ros2-control and related packages)**: The standard framework for controlling robot hardware. Includes specific controllers like `joint-trajectory-controller` and `joint-state-broadcaster`.
* **XACRO (ros-humble-xacro)**: A useful tool for writing robot models.
* **Rosbag (ros-humble-ros2bag)** and **PlotJuggler (ros-humble-plotjuggler\*)**: For recording and analyzing data from your robot.

With all these tools, you are well-equipped to start your first robotics project.

---

# Conclusion

You've successfully installed all the core software you need to get started with robotics. Your system now has **ROS 2 Humble** for software development, **Gazebo Fortress** for simulation, and a full set of packages for everything from hardware control to data analysis. You are ready to start building, coding, and simulating your own robots.

Good luck with your projects!
