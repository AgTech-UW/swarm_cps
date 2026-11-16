"""Finding and reading the configuration files. No ROS here.

The configuration lives in one directory with five small files:

  arena.yaml        the floor: bounds, walls, rock, zones, mission route
  fleet.yaml        the rovers: names, tag ids, how to reach each one, control numbers
  camera.yaml       the camera: device, mode, exposure, detector, floor reference tags
  default_cam.yaml  lens calibration (from the checkerboard, as in the course)
  camera_pose.yaml  where the camera is (written by swarm_calibrate floor)

The directory is, in order: the SWARM_CONFIG_DIR environment variable, the
config_dir argument, ./swarm_config, the installed package share directory,
and finally the config/ directory next to this source tree.
"""
import math
import os

from .core.arena import Arena
from .core.camera import CameraModel, FloorMap
from .core.controller import Params
from .core.localize import RobotTag


def config_dir(arg=None):
    cands = [os.environ.get('SWARM_CONFIG_DIR'), arg, os.path.join(os.getcwd(), 'swarm_config')]
    try:
        from ament_index_python.packages import get_package_share_directory
        cands.append(os.path.join(get_package_share_directory('swarm_cps'), 'config'))
    except Exception:
        pass
    cands.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, 'arena.yaml')):
            return os.path.abspath(c)
    raise FileNotFoundError('no configuration directory found (looked in: %s). Set SWARM_CONFIG_DIR to the '
                            'folder that holds arena.yaml, fleet.yaml and camera.yaml.'
                            % ', '.join(str(c) for c in cands if c))


def _yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


class Settings:
    def __init__(self, directory=None):
        self.dir = config_dir(directory)
        self.fleet = _yaml(os.path.join(self.dir, 'fleet.yaml'))
        self.camera = _yaml(os.path.join(self.dir, 'camera.yaml'))
        self.arena_path = os.path.join(self.dir, 'arena.yaml')
        self.pose_path = os.path.join(self.dir, 'camera_pose.yaml')
        robots = self.fleet.get('robots') or []
        if not robots:
            raise ValueError('fleet.yaml lists no robots')
        names = [r['name'] for r in robots]
        if len(set(names)) != len(names):
            raise ValueError('fleet.yaml: robot names must be unique')
        ids = [int(r['tag_id']) for r in robots]
        if len(set(ids)) != len(ids):
            raise ValueError('fleet.yaml: tag ids must be unique')
        self.robots = robots
        self.names = names                                    # rank order: leader first

    # ---- pieces ----
    def arena(self):
        return Arena.load(self.arena_path)

    def params(self):
        return Params(**(self.fleet.get('control') or {}))

    def robot_tags(self):
        tg = self.fleet.get('tags') or {}
        out = []
        for r in self.robots:
            off = r.get('tag_offset', tg.get('offset', [0.0, 0.0]))
            out.append(RobotTag(r['name'], r['tag_id'], tag_height=float(r.get('tag_height', tg.get('height', 0.12))),
                                tag_size=float(r.get('tag_size', tg.get('size', 0.15))),
                                yaw_offset=math.radians(float(r.get('yaw_offset_deg', 0.0))), offset=off))
        return out

    def anchors(self):
        """Floor reference tags: dict tag_id -> dict(x, y, z, [yaw, size])."""
        out = {}
        for a in (self.camera.get('anchors') or []):
            d = dict(x=float(a['x']), y=float(a['y']), z=float(a.get('z', 0.0)))
            if a.get('yaw') is not None:
                d['yaw'] = float(a['yaw'])
                d['size'] = float(a.get('size', (self.fleet.get('tags') or {}).get('size', 0.15)))
            out[int(a['id'])] = d
        return out

    def camera_model(self):
        c = self.camera.get('capture') or {}
        path = self.camera.get('intrinsics', 'default_cam.yaml')
        if not os.path.isabs(path):
            path = os.path.join(self.dir, path)
        cam = CameraModel.from_ros_yaml(path)
        w, h = int(c.get('width', cam.size[0])), int(c.get('height', cam.size[1]))
        if (w, h) != cam.size:
            if abs(w / cam.size[0] - h / cam.size[1]) > 0.01:
                raise ValueError('camera.yaml asks for %dx%d but the lens calibration is for %dx%d, a different '
                                 'shape. Calibrate at the resolution you run at.' % (w, h, cam.size[0], cam.size[1]))
            cam = cam.scaled(w, h)
        return cam

    def floor_map(self, camera=None):
        """The solved camera pose, or None if the floor has not been calibrated."""
        if not os.path.isfile(self.pose_path):
            return None
        camera = camera or self.camera_model()
        d = _yaml(self.pose_path)
        if list(d.get('image_size', camera.size)) != list(camera.size):
            raise ValueError('camera_pose.yaml was made at %s but the camera now runs at %s; recalibrate the floor'
                             % (d.get('image_size'), list(camera.size)))
        return FloorMap.from_dict(camera, d)

    def save_floor_map(self, fm, report=None):
        import yaml
        d = fm.to_dict()
        if report:
            d['report'] = {k: v for k, v in report.items() if k not in ('per_point_m',)}
        with open(self.pose_path, 'w') as f:
            f.write('# Written by swarm_calibrate floor. Where the camera is, relative to the arena.\n')
            yaml.safe_dump(d, f, default_flow_style=None)
        return self.pose_path
