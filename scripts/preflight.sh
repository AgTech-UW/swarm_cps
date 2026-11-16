#!/bin/bash
# What can be checked before any ROS node runs: the USB link, the camera's
# modes, the Python pieces and the ROS environment. Prints PASS / FAIL lines.
#
#     bash swarm_cps/scripts/preflight.sh
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAILED=1; }
note() { echo "  [ -- ] $1"; }
FAILED=0

echo "camera"
DEV=$(v4l2-ctl --list-devices 2>/dev/null | grep -A1 -i "See3CAM" | grep "/dev/video" | head -1 | tr -d '[:space:]')
if [ -z "$DEV" ]; then
    fail "no See3CAM found. Plugged in? Container started with the camera attached (privileged / --device)? Try: v4l2-ctl --list-devices"
else
    pass "See3CAM at $DEV"
    FMT=$(v4l2-ctl -d "$DEV" --list-formats-ext 2>/dev/null)
    if echo "$FMT" | grep -q "MJPG"; then pass "camera offers MJPEG"; else fail "camera does not list MJPEG"; fi
    if echo "$FMT" | grep -A120 "MJPG" | grep -A8 "1920x1080" | grep -q "60.000 fps"; then
        pass "1920x1080 MJPEG at 60 fps is offered"
    else
        note "could not see 1920x1080 MJPEG at 60 fps in the mode list (v4l2-ctl -d $DEV --list-formats-ext). On a USB 2 link the camera hides its fast modes."
    fi
    SYS=$(udevadm info -q path -n "$DEV" 2>/dev/null)
    SPEED=""
    P="/sys$SYS"
    while [ "$P" != "/sys" ] && [ -n "$P" ]; do
        if [ -f "$P/speed" ]; then SPEED=$(cat "$P/speed"); break; fi
        P=$(dirname "$P")
    done
    if [ -n "$SPEED" ]; then
        if [ "${SPEED%.*}" -ge 5000 ]; then pass "USB link speed ${SPEED} Mbit/s"; else
            fail "USB link speed ${SPEED} Mbit/s: this is USB 2. Use a blue USB 3 port directly on the laptop and a USB 3 cable; over 3 m needs an ACTIVE cable."; fi
    else
        note "USB speed unknown here; on the host run: lsusb -t   (the camera's line must say 5000M)"
    fi
    if pgrep -f "usb_cam_node_exe|swarm_cps.*perception|cheese|guvcview" >/dev/null; then
        note "another program is using or may be using the camera (usb_cam / perception / cheese). Only one can hold it."
    fi
fi

echo "python"
python3 - <<'PY' || FAILED=1
import sys
ok = True
def line(good, text):
    global ok
    print("  [%s] %s" % ("PASS" if good else "FAIL", text)); ok = ok and good
try:
    import cv2, numpy
    line(hasattr(cv2, "aruco"), "OpenCV %s with aruco, numpy %s" % (cv2.__version__, numpy.__version__))
    line(int(numpy.__version__.split(".")[0]) < 2, "numpy is 1.x (numpy 2 breaks the ROS Python packages; fix: pip3 install 'numpy<2')")
except Exception as e:
    line(False, "OpenCV/numpy: %s" % e)
try:
    import pupil_apriltags
    print("  [PASS] pupil_apriltags (AprilTag library)")
except Exception as e:
    print("  [ -- ] pupil_apriltags missing: far tags will be missed. pip3 install --no-deps pupil-apriltags")
try:
    import yaml
    line(True, "PyYAML")
except Exception as e:
    line(False, "PyYAML: %s" % e)
sys.exit(0 if ok else 1)
PY

echo "ROS environment"
if [ -n "$ROS_DISTRO" ]; then pass "ROS $ROS_DISTRO sourced"; else fail "ROS is not sourced in this terminal"; fi
if [ "$RMW_IMPLEMENTATION" = "rmw_cyclonedds_cpp" ]; then pass "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"; else
    fail "RMW_IMPLEMENTATION is '${RMW_IMPLEMENTATION:-unset}'; the rovers use rmw_cyclonedds_cpp. export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"; fi
echo "  [ -- ] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} (the planner's own domain; each rover's is in fleet.yaml)"
if [ -n "$SWARM_CONFIG_DIR" ] && [ -f "$SWARM_CONFIG_DIR/arena.yaml" ]; then pass "SWARM_CONFIG_DIR=$SWARM_CONFIG_DIR"; else
    note "SWARM_CONFIG_DIR not set; the package's built-in config will be used"; fi
if ros2 pkg prefix swarm_cps >/dev/null 2>&1; then pass "swarm_cps is built and on the path"; else
    fail "swarm_cps is not on the path: colcon build --packages-select swarm_cps --symlink-install && source install/setup.bash"; fi

echo "language model (only needed for decider:=foreman)"
if curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    if curl -s -m 2 http://127.0.0.1:11434/api/tags | grep -q "qwen2.5"; then pass "Ollama is up and has a qwen2.5 model"; else
        note "Ollama is up but qwen2.5:7b is not pulled: ollama pull qwen2.5:7b"; fi
else
    note "no Ollama server on this machine: install from ollama.com, then: ollama serve & ollama pull qwen2.5:7b"
fi

echo
if [ "$FAILED" = "0" ]; then echo "RESULT: ready for  ros2 run swarm_cps cam_bench"; else echo "RESULT: fix the FAIL lines first"; fi
exit $FAILED
