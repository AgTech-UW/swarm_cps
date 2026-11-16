#!/bin/bash
# One-time set-up inside the AgTech ROS 2 Humble container (osrf/ros:humble-desktop-full).
# Run from the folder that contains swarm_cps/  (for the course repo: /workspaces/agtech_ros2).
#
#     bash swarm_cps/scripts/setup_container.sh
#
# Safe to run again. It installs nothing that replaces a ROS-provided Python package:
# pupil-apriltags is installed with --no-deps so pip cannot upgrade numpy under cv_bridge.
set -e
WS="$(pwd)"
if [ ! -f "$WS/swarm_cps/package.xml" ]; then
    echo "Run this from the folder that contains swarm_cps/ (found none in $WS)"; exit 1
fi

echo "=== 1/5 system packages ==="
apt-get update -qq
apt-get install -y -qq python3-pip v4l-utils usbutils udev ros-humble-rmw-cyclonedds-cpp >/dev/null
echo "  ok"

echo "=== 2/5 AprilTag library for Python ==="
if python3 -c "import pupil_apriltags" 2>/dev/null; then
    echo "  already installed"
else
    pip3 install --no-deps pupil-apriltags || echo "  !! could not install pupil-apriltags (no internet?). The OpenCV detector will be used; far tags will be missed."
fi
python3 - <<'PY'
import cv2, numpy
print("  OpenCV %s, numpy %s, aruco module: %s" % (cv2.__version__, numpy.__version__, hasattr(cv2, "aruco")))
try:
    import pupil_apriltags
    print("  pupil_apriltags: ok")
except Exception as e:
    print("  pupil_apriltags: NOT available (%s)" % e)
PY

echo "=== 3/5 build ==="
source /opt/ros/humble/setup.bash
colcon build --packages-select swarm_cps --symlink-install
echo "  ok"

echo "=== 4/5 your own copy of the configuration ==="
CFG="$WS/swarm_config"
if [ -d "$CFG" ]; then
    echo "  $CFG exists, left alone"
else
    cp -r "$WS/swarm_cps/config" "$CFG"
    if [ -f "$WS/config/default_cam.yaml" ]; then cp "$WS/config/default_cam.yaml" "$CFG/default_cam.yaml"; fi
    echo "  created $CFG  (edit the files there, not the ones inside swarm_cps/)"
fi

echo "=== 5/5 shell set-up ==="
add() { grep -qF "$1" /root/.bashrc 2>/dev/null || echo "$1" >> /root/.bashrc; }
add "source /opt/ros/humble/setup.bash"
add "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
add "[ -f $WS/install/setup.bash ] && source $WS/install/setup.bash"
add "export SWARM_CONFIG_DIR=$CFG"
echo "  added to /root/.bashrc"

echo
echo "Done. Open a NEW terminal, then:"
echo "    ros2 run swarm_cps simulate --check        # no hardware needed"
echo "    bash swarm_cps/scripts/preflight.sh        # camera, USB, environment"
