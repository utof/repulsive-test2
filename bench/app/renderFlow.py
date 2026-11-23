"""bench/app/renderFlow.py — python3 -I bench/app/renderFlow.py <flow.json> <out-stem> [title]

Turns a trajectory written by recordFlow.ts into <out-stem>.mp4 and <out-stem>.gif.

One recorded iterate per descent step. Between two iterates the picture is LINEARLY
INTERPOLATED (SUB sub-frames) so the motion reads smoothly; the step counter, the energy and
the solver time shown in the corner change only at real steps. The camera turns slowly so the
3-D shape is readable; the scale is fixed for the whole clip, so growth on screen is real.
Needs Pillow and ffmpeg. Why not screen-capture the app: this is deterministic and does not
depend on a headless browser producing WebGPU frames.
"""
import json, math, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

flow_path, stem = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else ''
flow = json.load(open(flow_path))
frames = flow['frames']
n = flow['n']
W = H = 720
SUB = 5          # sub-frames per step
HOLD = 20        # extra frames on the first and last iterate
FPS = 30

def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()
F_BIG, F_SMALL = font(22), font(16)

# one scale and centre for the whole clip
cx = cy = cz = 0.0
r = 0.0
for fr in frames:
    xyz = fr['xyz']
    mx = sum(xyz[0::3]) / n; my = sum(xyz[1::3]) / n; mz = sum(xyz[2::3]) / n
    fr['c'] = (mx, my, mz)
    r = max(r, max(math.sqrt((xyz[3*i]-mx)**2 + (xyz[3*i+1]-my)**2 + (xyz[3*i+2]-mz)**2) for i in range(n)))
scale = 0.44 * W / r

def draw(xyz, c, angle, label):
    img = Image.new('RGB', (W, H), (16, 18, 30))
    d = ImageDraw.Draw(img)
    ca, sa = math.cos(angle), math.sin(angle)
    ct, st = math.cos(0.45), math.sin(0.45)   # fixed tilt
    P = []
    for i in range(n):
        x = xyz[3*i] - c[0]; y = xyz[3*i+1] - c[1]; z = xyz[3*i+2] - c[2]
        x, z = ca*x + sa*z, -sa*x + ca*z          # turn about the vertical axis
        y, z = ct*y - st*z, st*y + ct*z           # tilt
        P.append((W/2 + scale*x, H/2 - scale*y, z))
    segs = []
    for i in range(n):
        a, b = P[i], P[(i+1) % n]
        segs.append(((a[2]+b[2]) / 2, a, b))
    segs.sort(key=lambda s: s[0])                 # far first
    for depth, a, b in segs:
        t = max(0.0, min(1.0, 0.5 + 0.5 * depth / r))   # 0 far .. 1 near
        col = (int(40 + 60*t), int(110 + 130*t), int(170 + 85*t))
        d.line((a[0], a[1], b[0], b[1]), fill=col, width=2 if t < 0.5 else 3)
    if title:
        d.text((18, 14), title, fill=(235, 235, 245), font=F_BIG)
    d.text((18, H - 52), label[0], fill=(235, 235, 245), font=F_SMALL)
    d.text((18, H - 28), label[1], fill=(150, 160, 185), font=F_SMALL)
    return img

tmp = tempfile.mkdtemp(prefix='renderflow-')
k = 0
total = (len(frames) - 1) * SUB + 2 * HOLD
def emit(xyz, c, fr):
    global k
    angle = 0.6 + 2 * math.pi * 0.35 * k / total
    label = (f"step {fr['step']:>3}   energy {fr['energy']:.3f}",
             f"{n} vertices   solver time {fr['ms']/1000:.1f} s   route: {flow['route']}")
    draw(xyz, c, angle, label).save(os.path.join(tmp, f'{k:05d}.png'))
    k += 1

for _ in range(HOLD):
    emit(frames[0]['xyz'], frames[0]['c'], frames[0])
for a, b in zip(frames, frames[1:]):
    for s in range(SUB):
        u = (s + 1) / SUB
        xyz = [p + (q - p) * u for p, q in zip(a['xyz'], b['xyz'])]
        c = tuple(p + (q - p) * u for p, q in zip(a['c'], b['c']))
        emit(xyz, c, b if s == SUB - 1 else a)
for _ in range(HOLD):
    emit(frames[-1]['xyz'], frames[-1]['c'], frames[-1])

run = lambda *a: subprocess.run(a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
run('ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(tmp, '%05d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', stem + '.mp4')
run('ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(tmp, '%05d.png'),
    '-vf', 'fps=15,scale=420:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse',
    stem + '.gif')
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print(f'{k} frames -> {stem}.mp4 ({os.path.getsize(stem + ".mp4")//1024} KB), {stem}.gif ({os.path.getsize(stem + ".gif")//1024} KB)')
