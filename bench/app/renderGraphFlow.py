"""bench/app/renderGraphFlow.py — python3 -I bench/app/renderGraphFlow.py <out.mp4> <headline> <frames.json> [more.json ...]

Turns recordings of the flat-graph flow into one clip with the panels SIDE BY SIDE. Each json
is `{panels: [{title, a, b, nodeCount, hi, frames: [{x, label, asleep?}]}]}` as written by
recordGraphFlow.ts (or by pkmGraphs.ts with FRAMES=<file>); the panels of all files are shown
left to right, frame k of each at the same time. A panel that has run out of frames holds its
last one.

One recorded step per video frame, 30 per second, no interpolation: what moves on screen is
what a step did. ONE zoom for the whole clip and for all panels (the union of everything that
is ever drawn), so sizes and distances can be compared across panels and over time.
Colours: blue edges; red dots = the graph's own nodes; green = the highlighted group or node;
in a panel that records sleeping, nodes that are asleep are grey and awake ones orange.
Needs Pillow and ffmpeg. Why not a screen capture of the app: deterministic, no headless
browser. @see bench/app/recordGraphFlow.ts
"""
import json, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

out, headline, files = sys.argv[1], sys.argv[2], sys.argv[3:]
panels = [p for f in files for p in json.load(open(f))['panels']]
P = len(panels)
W = 1280 // P if P > 1 else 900          # panel width in output pixels
H = 640 if P > 1 else 900
HEAD, FOOT, SS, FPS = 64, 34, 2, 30      # SS: drawn at twice the size, then scaled down
W -= W % 2

def font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        if os.path.exists(path):
            return ImageFont.truetype(path, size * SS)
    return ImageFont.load_default()
F_HEAD, F_TITLE, F_LABEL = font(17), font(13), font(13)

# One view for everything.
xs, ys = [], []
for p in panels:
    for fr in (p['frames'][0], p['frames'][len(p['frames']) // 2], p['frames'][-1]):
        xs += fr['x'][0::2]; ys += fr['x'][1::2]
for p in panels:
    for fr in p['frames'][::10]:
        xs += (min(fr['x'][0::2]), max(fr['x'][0::2])); ys += (min(fr['x'][1::2]), max(fr['x'][1::2]))
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
pad = 24
scale = min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0)) * SS
ox = (W * SS - scale * (x1 - x0)) / 2
oy = (H * SS - scale * (y1 - y0)) / 2

total = max(len(p['frames']) for p in panels)
big = sum(p["nodeCount"] for p in panels) / P > 700
dot = (1.6 if big else 2.6) * SS
line = max(1, round((0.8 if big else 1.3) * SS))
tmp = tempfile.mkdtemp(prefix='graphflow-')
for k in range(total + FPS * 2):                 # hold the last frame for two seconds
    img = Image.new('RGB', (W * P * SS, (H + HEAD + FOOT) * SS), 'white')
    dr = ImageDraw.Draw(img)
    dr.text((14 * SS, 8 * SS), headline, fill=(20, 20, 20), font=F_HEAD)
    for j, p in enumerate(panels):
        fr = p['frames'][min(k, len(p['frames']) - 1)]
        x = fr['x']
        left, top = j * W * SS, HEAD * SS
        at = lambda i: (left + ox + (x[2 * i] - x0) * scale, top + oy + (y1 - x[2 * i + 1]) * scale)
        if j > 0:
            dr.line([(left, top - 30 * SS), (left, (H + HEAD + FOOT) * SS)], fill=(200, 200, 200), width=SS)
        dr.text((left + 14 * SS, top - 28 * SS), p['title'], fill=(60, 60, 60), font=F_TITLE)
        for a, b in zip(p['a'], p['b']):
            dr.line([at(a), at(b)], fill=(45, 80, 170), width=line)
        hi = set(p['hi'])
        asleep = set(fr['asleep']) if 'asleep' in fr else None
        for i in range(p['nodeCount']):
            px, py = at(i)
            if i in hi: colour, rad = (20, 150, 60), dot * 1.5
            elif asleep is None: colour, rad = (200, 60, 40), dot
            elif i in asleep: colour, rad = (175, 175, 175), dot
            else: colour, rad = (235, 130, 20), dot
            dr.ellipse([px - rad, py - rad, px + rad, py + rad], fill=colour)
        dr.text((left + 14 * SS, (HEAD + H + 6) * SS), fr['label'], fill=(20, 20, 20), font=F_LABEL)
    img.resize((W * P, H + HEAD + FOOT), Image.LANCZOS).save(os.path.join(tmp, f'{k:05d}.png'))
subprocess.run(['ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(tmp, '%05d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print(f'{out}: {P} panel(s), {total} steps, {(total + FPS * 2) / FPS:.0f} s')
