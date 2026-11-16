"""Printable AprilTags (family 36h11) at an exact size, one per page.

    tag_sheet --ids 1 2 3 --size 0.15                    # rover tags
    tag_sheet --ids 100 101 102 103 104 105 --size 0.15  # floor reference tags
    tag_sheet --ids 1 2 3 --size 0.20 --paper a4

Print at 100 % ("actual size", no "fit to page"). Each page has a 100 mm bar:
measure it. If it is not 100 mm, the printer scaled the page; measure the black
square instead and put that number in fleet.yaml (tags.size).

The white border is part of the tag: do not trim it off, and mount the tag
flat and matte (no glossy lamination; glare erases it).
"""
import argparse
import os
import sys

import numpy as np

from ..core.tags import tag_image
from .common import cv2, say

PAPER_MM = dict(letter=(215.9, 279.4), a4=(210.0, 297.0), a3=(297.0, 420.0), tabloid=(279.4, 431.8))


def make_page(tag_id, size_m, paper='letter', dpi=300):
    wmm, hmm = PAPER_MM[paper]
    W, H = int(round(wmm / 25.4 * dpi)), int(round(hmm / 25.4 * dpi))
    page = np.full((H, W), 255, np.uint8)
    cell = int(round(size_m * 1000 / 8 / 25.4 * dpi))          # pixels per tag cell
    tag = tag_image(tag_id, cells_px=cell, border_cells=1)
    th, tw = tag.shape
    if tw > W - 2 * dpi // 4 or th > H - 2 * dpi:
        raise ValueError('a %.0f mm tag does not fit on %s paper; use a3 or tabloid' % (1e3 * size_m, paper))
    x0, y0 = (W - tw) // 2, (H - th) // 2
    page[y0:y0 + th, x0:x0 + tw] = tag
    cv2.rectangle(page, (x0 - 2, y0 - 2), (x0 + tw + 1, y0 + th + 1), 200, 1)       # cut line outside the white border
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(page, 'TOP OF TAG. With yaw_offset_deg 0 this edge faces the FRONT of the rover.', (x0, y0 - 40), f, 1.0, 0, 2, cv2.LINE_AA)
    cv2.arrowedLine(page, (W // 2, y0 - 100), (W // 2, y0 - 220), 0, 6, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(page, 'AprilTag 36h11  id %d   black square %.0f mm' % (tag_id, round(8 * cell / dpi * 25.4)),
                (x0, y0 + th + 70), f, 1.3, 0, 3, cv2.LINE_AA)
    bar = int(round(100 / 25.4 * dpi))
    by = y0 + th + 150
    cv2.rectangle(page, (x0, by), (x0 + bar, by + 12), 0, -1)
    cv2.putText(page, 'this bar is 100 mm if the page printed at 100 %', (x0, by + 60), f, 1.0, 0, 2, cv2.LINE_AA)
    return page, 8 * cell / dpi * 25.4 / 1000.0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='tag_sheet', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ids', nargs='+', type=int, required=True)
    ap.add_argument('--size', type=float, default=0.15, help='edge of the black square in metres')
    ap.add_argument('--paper', default='letter', choices=sorted(PAPER_MM))
    ap.add_argument('--out', default='tags')
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    pages = []
    actual = args.size
    for i in args.ids:
        try:
            page, actual = make_page(i, args.size, args.paper)
        except ValueError as e:
            say(str(e))
            return 1
        path = os.path.join(args.out, 'tag36h11_%03d.png' % i)
        cv2.imwrite(path, page)
        pages.append(page)
    say('wrote %d pages to %s/ (PNG, 300 dpi, %s)' % (len(pages), args.out, args.paper))
    try:
        from PIL import Image
        ims = [Image.fromarray(p).convert('L') for p in pages]
        pdf = os.path.join(args.out, 'tags.pdf')
        ims[0].save(pdf, save_all=True, append_images=ims[1:], resolution=300.0)
        say('and %s: print this one, at 100 %% scale' % pdf)
    except Exception:
        say('Pillow is not installed, so no PDF. Print the PNGs at 300 dpi, 100 % scale.')
    say('black square as drawn: %.1f mm. Put the size you MEASURE on paper in fleet.yaml (tags.size).' % (1e3 * actual))
    return 0


if __name__ == '__main__':
    sys.exit(main())
