"""Synthetic camera: renders AprilTags lying on horizontal planes as the real
(distorted, oblique) camera would see them. Used by the tests and by the
rehearsal tools to check the perception chain end to end with no hardware:
what is rendered here goes through exactly the code the real frames go through.

Rendering is by ray casting, so lens distortion and perspective are exact: for
every output pixel the ray is intersected with the tag's plane and the tag
bitmap is sampled there. Images are rendered at `ss` times the resolution and
box-filtered down, which stands in for the sensor's pixel integration.
"""
import math

import numpy as np

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None

from .tags import tag_image


_RAY_CACHE = {}


class Scene:
    def __init__(self, floor_map, floor_gray=120, texture=8.0, seed=0):
        self.fm = floor_map
        self.floor_gray = floor_gray
        self.texture = texture
        self.rng = np.random.default_rng(seed)
        self.tags = []                 # (id, x, y, yaw, z, size)

    def add_tag(self, tag_id, x, y, yaw, z=0.0, size=0.15):
        """A tag whose "up" edge points along `yaw` in the arena, centre at
        (x, y), flat at height z, black-square edge length `size` metres."""
        self.tags.append((int(tag_id), float(x), float(y), float(yaw), float(z), float(size)))

    def _ray_grid(self, ss):
        cam = self.fm.camera
        key = (ss, cam.size, cam.K.tobytes(), cam.D.tobytes(), self.fm.R.tobytes())
        if key in _RAY_CACHE:
            return _RAY_CACHE[key]
        w, h = cam.size
        W, H = w * ss, h * ss
        u = (np.arange(W) + 0.5) / ss - 0.5
        v = (np.arange(H) + 0.5) / ss - 0.5
        uu, vv = np.meshgrid(u, v)
        px = np.stack([uu.ravel(), vv.ravel()], 1)
        xn = np.empty_like(px)
        step = 400000
        for i in range(0, len(px), step):                     # chunked: 2M Newton solves
            xn[i:i + step] = cam.undistort(px[i:i + step], iters=12, tol=1e-7)
        d = np.concatenate([xn, np.ones((len(xn), 1))], 1) @ self.fm.R
        if len(_RAY_CACHE) > 3:
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = d.reshape(H, W, 3).astype(np.float32)
        return _RAY_CACHE[key]

    def render(self, ss=2, noise=2.0, blur=0.6):
        cam = self.fm.camera
        w, h = cam.size
        d = self._ray_grid(ss)
        H, W = d.shape[:2]
        C = self.fm.center
        img = np.full((H, W), float(self.floor_gray), np.float32)
        if self.texture:
            # low-frequency floor texture so the detector sees a real background
            small = self.rng.normal(0, self.texture, (H // 16 + 1, W // 16 + 1)).astype(np.float32)
            img += cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
        # the floor only exists below the horizon
        down = d[..., 2] < -1e-6
        img[~down] = 40.0
        for (tid, x, y, yaw, z, size) in sorted(self.tags, key=lambda t: t[4]):
            bmp = tag_image(tid, cells_px=16, border_cells=1).astype(np.float32)
            n_cells = 10.0                                    # 8 + 2 border cells
            full = size * n_cells / 8.0                       # edge incl. white border
            # bounding box of the tag in the image, to limit the work
            cu, cr = np.array([math.cos(yaw), math.sin(yaw)]), np.array([math.sin(yaw), -math.cos(yaw)])
            cs = [np.array([x, y]) + a * full / 2 * cu + b * full / 2 * cr for a in (-1, 1) for b in (-1, 1)]
            pxs, front = self.fm.world_to_pixel([[c[0], c[1], z] for c in cs])
            if not front.all():
                continue
            x0 = int(max(0, math.floor((pxs[:, 0].min() - 2) * ss)))
            x1 = int(min(W, math.ceil((pxs[:, 0].max() + 3) * ss)))
            y0 = int(max(0, math.floor((pxs[:, 1].min() - 2) * ss)))
            y1 = int(min(H, math.ceil((pxs[:, 1].max() + 3) * ss)))
            if x1 <= x0 or y1 <= y0:
                continue
            dd = d[y0:y1, x0:x1]
            with np.errstate(divide='ignore', invalid='ignore'):
                s = (z - C[2]) / dd[..., 2]
            Px = C[0] + s * dd[..., 0] - x
            Py = C[1] + s * dd[..., 1] - y
            a_up = Px * cu[0] + Py * cu[1]                    # along the tag's up axis
            a_rt = Px * cr[0] + Py * cr[1]                    # along the tag's right axis
            col = (a_rt / full + 0.5) * bmp.shape[1]
            row = (0.5 - a_up / full) * bmp.shape[0]
            inside = (s > 0) & (col >= 0) & (col < bmp.shape[1]) & (row >= 0) & (row < bmp.shape[0])
            ci = np.clip(col.astype(int), 0, bmp.shape[1] - 1)
            ri = np.clip(row.astype(int), 0, bmp.shape[0] - 1)
            val = bmp[ri, ci] * (200.0 / 255.0) + 25.0        # printed paper: not pure black/white
            sub = img[y0:y1, x0:x1]
            sub[inside] = val[inside]
        out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if blur:
            out = cv2.GaussianBlur(out, (0, 0), blur)
        if noise:
            out = out + self.rng.normal(0, noise, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
