# swarm_cps: side-mounted faux GPS, swarm planner, AI foreman

One ROS 2 Humble package that replaces the camera pipeline, runs the swarm
controller, and puts a language model in the supervisor's seat. It does not
need the code that is on the other computer.

```
camera --USB--> perception  --/roverN/gps/pose, /roverN/gps/vel-->  planner  --/cmd_vel in each rover's domain--> rovers
                (one process,                                          ^   |
                 no images on DDS)                    /swarm/decision  |   |  /swarm/observation
                                                                    foreman (LLM, optional)        console (you)
```

## Read this first: what has been run and what has not

**Run, and passing (73 automated tests, in Python 3.10 / NumPy 1.21 / OpenCV 4.5.4, the versions in the Humble container):**

- Perception end to end on rendered frames that use your real lens calibration
  (`default_cam.yaml`) and a camera looking across the floor from a corner:
  detect tags, solve the camera pose from floor tags, locate rovers.
- The controller, safety layers and deciders against simulated rovers with
  actuator lag, a turning limit, camera noise, latency and dropped detections,
  on floors from 3.6 x 3.0 m to 6 x 4 m with 2 to 4 rovers.
- Every ROS node, executed against a strict stand-in for `rclpy`
  (`test/fake_ros`): constructors, timers, callbacks, message field types, QoS
  matching, parameters, the multi-domain command bridge, logging.

**Not run, because the machine this was written on has no ROS 2, no camera, no rovers and no GPU:**

| Not run | Where it gets checked | If it fails |
|---|---|---|
| Real ROS 2 middleware | Tonight: `rehearse.launch.py` | See "If something does not work" |
| The real camera through OpenCV/V4L2 | Gate 1: `cam_bench` | `source:=ros_compressed` (your old usb_cam launch feeds it) |
| Commands crossing ROS domains | Gate 5: `doctor` | `bridge: relay` in `fleet.yaml` |
| A real language model on this prompt | Tonight: `simulate --decider foreman` | `decider:=rule` still runs the demo's other half |
| Rovers on the floor | Gates 6 and 7 | Tuning table at the end |

The "supervisor" numbers below were produced with an offline stand-in that
matches keywords. They show that a sentence reaches the wheels through every
layer. They say nothing about what qwen2.5:7b will decide; you find that out
tonight in simulation.

## Install (10 minutes, once)

Copy the `swarm_cps` folder into `/workspaces/agtech_ros2/` and, inside the container:

```bash
cd /workspaces/agtech_ros2
bash swarm_cps/scripts/setup_container.sh     # apt + pip (--no-deps), colcon build, creates ./swarm_config, edits ~/.bashrc
# open a NEW terminal
ros2 run swarm_cps simulate --check
```

Your settings live in `/workspaces/agtech_ros2/swarm_config/` (five small YAML
files). Edit those, not the copies inside the package.

If `colcon build` gives trouble, everything also runs without it:
`export PYTHONPATH=/workspaces/agtech_ros2/swarm_cps:$PYTHONPATH` and replace
`ros2 run swarm_cps X` with `python3 -m swarm_cps.X_node` (nodes) or
`python3 -m swarm_cps.tools.X` (tools).

## Tonight and tomorrow, with no hardware

1. **Fit the course to your floor.** Measure the floor the camera will see.

   ```bash
   ros2 run swarm_cps layout --width 4.2 --height 3.2 --rovers 3 --write
   ros2 run swarm_cps simulate --check          # where rovers, tape, rock and strip go; dry run
   ```
   Set the number of rovers and their tag ids in `swarm_config/fleet.yaml` first.

2. **Try the real model.** Install Ollama (ollama.com), then:

   ```bash
   ollama pull qwen2.5:7b
   ros2 run swarm_cps simulate --compare --decider foreman \
     --instruction "Neighbour's got cattle along the north fence. Keep the machines well back from that side." \
     --gif /tmp/run.gif --transcript /tmp/run.json
   ```
   This is the whole demo in simulation: the hand-written rule and the model get
   the same sentence. You want fewer seconds in the north strip for the model
   and zero contacts for both. Read `/tmp/run_foreman.json` to see exactly what
   the model was shown and what it said. Try other sentences.

3. **Rehearse the ROS graph.** Terminal 1: `ros2 launch swarm_cps rehearse.launch.py`
   (add `backend:=ollama` for the real model). Terminal 2: `ros2 run swarm_cps console`,
   type `say <sentence>`, then `go`. Watch `/tmp/swarm_view.jpg`. If this works,
   ROS, the planner, the foreman node and the console are all good, and
   Wednesday is only about the camera and the rovers.

4. **Print tags.** `ros2 run swarm_cps tag_sheet --ids 1 2 3 --size 0.15` and
   `--ids 100 101 102 103 104 105 106` for the floor. Measure the black square
   on paper and put that number in `fleet.yaml` (`tags.size`).

## Two stages

**Stage 1: swarm dynamics only, no language model.** The leader follows a route
with the vector field. Each follower drives the same vector field along a line
through its own slot behind the leader and uses its speed to hold its place.
Boid separation pushes any two rovers apart inside 0.38 m, a rover brakes for a
higher-ranked one ahead of it, the leader slows when the crew falls behind, and
any rover stops rather than drive into another. This is the planner's default
(`decider:=column`); nothing needs switching off. Give it a plain course:

```bash
ros2 run swarm_cps layout --width 4.2 --height 3.2 --rovers 3 --simple --laps 2 --write
ros2 run swarm_cps simulate --check
ros2 run swarm_cps simulate --decider column --gif /tmp/stage1.gif
ros2 launch swarm_cps swarm.launch.py              # on the floor; then console: go
```
That is an empty floor and a racetrack: no gate, no strip. Things to show on it:
put the rovers down crooked and off their marks and watch them form up; add
`--rock` to the layout line and watch each one swerve round it and rejoin;
`decider:=schedule` opens the line abreast on the straights and folds it for
the turns, still with no model. In simulation the three-rover loop finishes two
laps with no contact and neighbours between 0.30 and 0.62 m apart all the way round.

**Stage 2: the foreman.** Same planner, same safety layers, plus the course with
the gate and the strip (`layout` without `--simple`) and `decider:=foreman`.
Everything from Gate 7 on. Stage 1 working on the floor is the precondition.

`layout --write` keeps the previous course as `arena.yaml.bak`, so switching
between the two courses is one command each way.

## The short version: camera up, cards down, rovers moving

```bash
ros2 run swarm_cps cam_bench                                    # camera link ok?
ros2 run swarm_cps calibrate floor --auto --origin 100 --xaxis 101
ros2 run swarm_cps calibrate yaw --heading 0 --write            # rovers parked facing +x
ros2 run swarm_cps layout --width W --height H --rovers 2 --duel head-on --write
ros2 launch swarm_cps swarm.launch.py                           # then console: go
```
**Cards, nothing measured.** Throw 6 or more floor cards anywhere in view, flat,
spread out, some at the far side. Card 100 becomes the origin (0, 0) and +x
points from it toward card 101, so put those two where you want the corner and
the long edge, 2 m or more apart. W and H are then how far the usable floor
extends from card 100. The only number that must be right is the printed tag
size (`tags.size`): it sets the scale. Afterwards tape-measure the distance
between two far-apart cards and compare with the positions it printed; that
one measurement checks everything. On rendered frames this lands within 3 mm.

**Lens distortion** is handled. Every tag corner is run backwards through your
lens calibration before any geometry, so the bent image becomes straight rays.
Straight tape lines look curved in the picture; the 0.5 m grid the tool draws in
`/tmp/swarm_calibration.jpg` curves the same way and should lie on the floor.
If it peels away from the floor near the image edges, the lens calibration is
weak there: keep the working area away from the edges or redo the checkerboard.

**Collision course.** `--duel head-on` sends the last rover in `fleet.yaml`
straight at the others; `--duel crossing` brings it in from the side, timed to
arrive together. They use boat rules: head-on, both turn right; crossing, the
one with the other on its right goes behind it. With 3 rovers a column of two
meets one. In simulation every case passes with no contact, and with the rule
switched off they stop nose to nose instead of touching.

**Your own route.** `ros2 run swarm_cps draw_path` opens a top-down view of
your floor: click waypoints, `s` to save. The grey line is what will be driven
(waypoints joined by curves the rovers can make); the leader follows it and the
rest keep formation behind. Without a window: `draw_path --points 1.2,0.8 3.2,0.8 3.4,2.2`.
It refuses routes that leave the floor or loop back on themselves. Run
`simulate --check` afterwards. The window itself is the one part of this tool
I could not run here (no display); the typed-points path is tested.

## Wednesday: seven gates, in order

Do not skip ahead. Each gate has a pass line; each failure has a next step.

### Gate 0. Mount the camera (15 min)

High, rigid, on the **middle of the long wall** if you can, not the corner.
Predicted with your lens, 150 mm tags, camera 2.4 m up:

| Floor | Corner | Middle of long wall |
|---|---|---|
| 4.2 x 3.2 m | 93 % of floor usable, 69 % comfortable | 99 % usable, 96 % comfortable |
| 5.5 x 3.5 m | 67 % usable | 95 % usable |

The far corner is the problem from a corner mount: a flat tag 5 m away at a
shallow angle is one or two pixels per cell. Check your own position:

```bash
ros2 run swarm_cps plan_mount --camera 2.1 -0.2 2.4        # camera x y z in arena metres
```
It prints where to aim the image centre and writes `/tmp/swarm_mount_floor.png`
(coverage map) and `/tmp/swarm_mount_preview.jpg` (what the camera should see).
"Usable" and "comfortable" are thresholds I chose (1.8 and 2.5 pixels per tag
cell); `cam_bench` reports the real figure per tag once the camera is up.

### Gate 1. Camera link (10 min). Nothing else running.

```bash
bash swarm_cps/scripts/preflight.sh
ros2 run swarm_cps cam_bench               # 10 s at the mode in camera.yaml
ros2 run swarm_cps cam_bench --modes       # which of 30 / 60 / 120 fps the link really sustains
```
**Pass:** USB 5000 Mbit/s, delivered fps within 7 % of requested, no late frames.
**Fail:** USB speed 480 means a USB 2 port, hub or cable; a USB 3 run over 3 m needs an active cable. Low fps with
5000 Mbit/s means exposure is longer than a frame (`exposure_time_absolute` is in
units of 100 µs) or the CPU is busy. Drop `capture.fps` to 30 and move on; 30 is plenty at 0.13 m/s.
**Camera will not open at all from OpenCV:** start your old `usb_cam` launch and run
perception with `source:=ros_compressed`. You lose the frame-drop fix but keep the accuracy fix.

### Gate 2. Floor calibration (30 min). The step that decides accuracy.

1. Mark the origin and the x axis with tape. Tape 6 to 8 floor tags flat,
   spread over the whole floor including the far side and the middle.
2. Measure each tag centre's x and y and type them into `camera.yaml` (`anchors`).
   Two tape measures from two known marks beat squaring off a wall by eye.
3. ```bash
   ros2 run swarm_cps calibrate floor
   ```
**Pass:** grade GOOD, cross-check under 15 mm, and in `/tmp/swarm_calibration.jpg`
the drawn 0.5 m grid lies on the floor through your tape marks.
**Fail:** it names the worst tag; re-measure that one. "Nearly in a line" means spread them out.
A tag "not seen" is too far or too slanted: move it closer or accept fewer (minimum 4, no cross-check below 5).
No spare tags printed: `calibrate floor --walk --tag 1 --height 0` and carry one tag from spot to spot.

### Gate 3. Rover tags (10 min)

Measure the height of a mounted tag above the floor to the millimetre and set
`tags.height`. A 2 cm error here is a 3 to 5 cm position error at the far side.
Park all rovers facing +x, then:

```bash
ros2 run swarm_cps calibrate yaw --heading 0 --write     # tags may be stuck on at any rotation
ros2 run swarm_cps calibrate check                        # put a rover on a measured mark and compare
```
**Pass:** position within 2 cm of the mark, "size mm" within 6 % of the real tag.
**Fail on size:** `tags.height` or `tags.size` is wrong.

### Gate 4. Poses flowing (5 min)

```bash
ros2 run swarm_cps perception --ros-args -p show:=true     # terminal 1
ros2 run swarm_cps doctor                                  # terminal 2
```
**Pass:** every rover above 12 poses/s and under 250 ms old; reference tags within 30 mm.
`/tmp/swarm_camera.jpg` refreshes once a second with what the camera sees.

### Gate 5. Command path (10 min). Rovers on blocks, wheels off the floor.

Put each rover's real `domain_id` in `fleet.yaml` (in the course set-up it is
the last number of its IP address) and run `doctor` again.
**Pass:** one subscriber on `/cmd_vel` for every rover.
**Fail for all rovers with "cannot join":** set `bridge: relay` in `fleet.yaml`; the launch file then starts one relay per rover.
**Fail for one rover:** it is off, still booting, or its `domain_id` is wrong.

Then with wheels still up: `ros2 launch swarm_cps swarm.launch.py`, `console`, `go`.
Wheels turn; press Enter and they stop; close the console window and they stop.
Check one rover's response: `ros2 topic echo /rover1/gps/vel` shows measured
speed and yaw rate next to what was commanded on `/rover1/auto/cmd_vel`. If a
rover runs slow or turns the wrong way, set `v_scale` / `w_scale` for it.

### Gate 6. First drive: single file (15 min). This is Stage 1.

Use the `--simple` course first. Rovers on their start marks (the planner prints them; `simulate --check` too).
`ros2 launch swarm_cps swarm.launch.py`, then `console`: `go`.
**Pass:** the column drives its laps nose to tail and stops where it started (on the full course: through the gate,
round the rock, round the headland). Enter stops it at any time.
If the turns run wide, the real turning radius is above 0.35 m: raise `min_turn_radius` and `mission.turn_radius`
together and re-run `simulate --check`.

### Gate 7. The demonstration

```bash
ros2 launch swarm_cps swarm.launch.py decider:=rule       # run A
ros2 launch swarm_cps swarm.launch.py decider:=foreman    # run B (Ollama running)
```
In the console, before `go`:
`say Neighbour's got cattle along the north fence. Keep the machines well back from that side.`

Each run writes `~/swarm_logs/run_<time>/result.json` with contacts, time, and
machine-seconds in the strip; `decisions.jsonl` has every choice and the
foreman's one-line reason; `~/swarm_logs/foreman_<time>.jsonl` has every prompt and raw reply.

## What the demonstration shows

Same floor, same rovers, same code, same sentence. The route as programmed runs
pass 2 along the north side. The hand-written rule has handlers for "wet" and
"rain"; nobody wrote one for cattle, so it drives the route as programmed. The
foreman reads the sentence, and on pass 1 moves the passes south, so pass 2
stays out of the strip. Nothing about cattle, fences or "keep back" is in the
program; the scene it is given has a strip with a neutral name and coordinates.

In simulation on the default floor (stand-in model, see the caveat at the top):

| Decider | Finished | Contacts | Machine-seconds in the north strip |
|---|---|---|---|
| rule | yes | 0 | 41 |
| supervisor (stand-in) | yes | 0 | 0 |

Then say something else and run again: "floor's slippery, take it slow", "tight
in here today, single file". The behaviour changes; the program does not.

What makes this safe to show: the model chooses only formation, leader speed
and offset, a few times a minute. It never steers. Its reply is read leniently
and checked strictly; a formation that does not fit the gate or the turn is
held by the safety filter; a model that goes silent, talks nonsense or asks for
everything at once cannot make rovers touch (four tests cover exactly those).

**"The LLM in the robots."** Two honest ways to say it. As built, the model runs
on the laptop and supervises the rovers, which is the architecture of both
papers and the right place for it. If it must physically run on a rover: the
MentorPi is a Raspberry Pi 5, Ollama runs on it, and the foreman takes a host
(`ros2 launch swarm_cps swarm.launch.py decider:=foreman host:=http://<rover-ip>:11434 model:=qwen2.5:3b`).
I have not tried it. Expect tens of seconds per decision on a Pi, which the
design tolerates because the crew keeps its last choice in the meantime, and
expect a 3B model to choose worse than the 7B, as your own paper found.

## Why frames were dropping, and what changed

| Old pipeline | Here |
|---|---|
| usb_cam decodes each 1080p MJPEG to 6 MB RGB, re-encodes JPEG for the compressed topic, apriltag_ros decodes it again | One process: V4L2 frame to detector. No image is ever published |
| 6 MB messages through DDS, with DDS bound to the Wi-Fi interface the rovers use | Only poses (about 100 bytes) on DDS |
| Queues between nodes: a slow consumer falls behind | Capture thread keeps only the newest frame; a late frame is skipped, never queued |
| Full-frame detection every frame | Windowed search round each tracked tag (about 10 ms for 3 rovers on a 2-core machine here, against about 95 ms for a full frame); full frame every 5 s and when a rover is lost |
| No way to tell which link dropped | `cam_bench` and `/swarm/perception` report camera fps, late frames, skipped frames and detection time separately |

Two bugs found on the way that would have cost you Wednesday:
`pupil_apriltags` corrupts the heap when a detector is freed (reproduced: the
process aborts with exit code 134 every time); the package now keeps the detector alive.
And the AprilTag library and OpenCV disagree on where a pixel's centre is by half a pixel,
which is 5 mm on the floor at the far side; corrected, and covered by a test.

## Why a slanted camera is accurate here

The old chain took each tag's 3-D pose from the AprilTag pose estimator. Seen
at a slant from metres away that estimate is weak in range and can flip in
tilt. This package never uses it. Every rover tag lies on a horizontal plane
at a known height, so each corner pixel is a ray that meets that plane at one
point. Position comes from the four corners on the plane and heading from their
arrangement, which cannot flip. The camera's own pose comes from floor tags at
measured positions (Gate 2). On rendered frames with a perfect survey the error
is 1 to 2 mm; on the real floor expect 1 to 2 cm, set by how well you measure
the floor tags and the tag height. The measured edge length of every tag is
checked against the real one each frame, so a wrong height or a bad detection
is rejected instead of published.

## If something does not work

| Symptom | Do this |
|---|---|
| A launch file fails to start | Run the nodes by hand, one terminal each, to see the error. Rehearsal: `ros2 run swarm_cps sim`, `ros2 run swarm_cps planner --ros-args -p decider:=foreman -p robot_output:=false`, `ros2 run swarm_cps foreman --ros-args -p backend:=mock`. Real: `ros2 run swarm_cps perception`, `ros2 run swarm_cps planner --ros-args -p decider:=rule` |
| Planner says "could not open the other ROS domains" | `bridge: relay` in `fleet.yaml` |
| Rovers do not move, doctor shows subscribers | Is `robot_output` true? Is the console heartbeat alive ("not arming: no heartbeat")? |
| "holding: stale pose for roverN" | That tag is not being seen: glare, tag at the edge of view, or perception stopped |
| "FLEET STOP: geofence" | A rover left `bounds`. `reset` in the console, put rovers back. Widen `bounds` only if the camera sees that floor |
| Rover stops "at the gate post" | It cut the gate. Widen the gate, slow down, or check that rover's `yaw_offset_deg` |
| Followers weave | Lower `k_heading` to 1.8 in `fleet.yaml` `control:`; check pose age in `/swarm/status` |
| Followers lag, leader crawls | Leader waits for laggards by design. Raise `v_max`, or the rover is slow: set its `v_scale` |
| Wide formation never opens | Passes too short for the crew (the layout tool says so). Fewer rovers, or a longer floor |
| Foreman replies "UNREADABLE" often | Normal now and then; the crew keeps its last choice. If constant, check `ollama list` and the model name |
| Far tags flicker | `cam_bench` px/cell under 2: bigger tags, camera nearer the middle of the long wall, more light with shorter exposure |
| Poses jump when a person walks through | A reference tag was covered during a rescan; harmless. A rover tag covered = stale stop, by design |

To fall all the way back: `source:=apriltag_ros` uses your original usb_cam and
apriltag_ros launch for detection and only replaces the geometry.

## Assumptions to check

- Rovers take `geometry_msgs/Twist` on `/cmd_vel`: `linear.x` forward speed, `angular.z` yaw rate (as in Lab 8).
- Each rover is on its own `ROS_DOMAIN_ID` (as in the course). If yours share one domain with namespaced topics, see the comment at the top of `fleet.yaml`.
- Turning radius 0.35 m, body 213 x 159 mm. Both are in `fleet.yaml`.
- The lens calibration is the one in your repo, made at 1920 x 1080. Another resolution needs its own calibration; the code refuses a mismatch.
- If the camera's image looks upside down or sideways because of how it is mounted, nothing needs changing: the floor calibration absorbs any rotation.

## Plugging in the full AI Foreman later

The reply format (`thought`, `formation`, `speed`, `offset`), the preset names,
the fit check, the safety filter and the lenient-read / strict-check interface
follow `shepherd/farmsupervisor.py` and the paper, so a transcript from here
reads like one from the simulator. `swarm_cps/core/foreman.py` holds the prompt
(`SYSTEM`) and the clients; `core/controller.py` `observation()` builds what the
model sees. To use another model or prompt, change those two places only.
Differences from the paper are deliberate and small: lengths are metres on the
floor, slots follow the leader's track in every formation, steering is
proportional instead of bang-bang (smoother on real servos), and the leader
slows for laggards.

## Files

```
swarm_cps/core/      no ROS: camera.py (lens + floor map), tags.py (detectors), localize.py (tracking + filter),
                     calib.py, path.py (Dubins routes), formation.py, arena.py, controller.py, deciders.py,
                     foreman.py, sim.py, synth.py (rendered test frames), viz.py, layout.py
swarm_cps/           perception_node, planner_node, foreman_node, console, sim_node, cmd_relay, bridge, capture, settings
swarm_cps/tools/     calibrate, cam_bench, plan_mount, layout, simulate, tag_sheet, doctor
config/              arena.yaml, fleet.yaml, camera.yaml, default_cam.yaml  (camera_pose.yaml is written by calibrate)
launch/              swarm.launch.py (real), rehearse.launch.py (simulated rovers)
scripts/             setup_container.sh, preflight.sh
test/                73 tests: python3 -m pytest swarm_cps/test -q   (about a minute; needs no ROS)
```
