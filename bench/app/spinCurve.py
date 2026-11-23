"""bench/app/spinCurve.py — python3 -I bench/app/spinCurve.py <curve.obj> <out-stem> [title]

Renders ONE full turn of a static closed polyline (an OBJ with `v` and `l` records, the format
of the Repulsive Curves knot datasets) to <out-stem>.mp4, with three numbers in the corner that
say how "tight" the embedding is:
  - mean turning angle per vertex (a smooth curve sampled this finely turns well under 1 degree),
  - longest / shortest edge,
  - closest approach of two strands, in units of the mean edge length (strands = vertices at
    least 10 edges apart along the curve).
Why it exists: the hierarchical route gives no speed-up on such inputs (97% of pairs stay
exact, GMRES needs 8x the iterations), and the picture explains why better than the numbers.
@see bench/app/renderFlow.py (the same drawing, for a recorded descent)
Needs Pillow and ffmpeg.
"""
import math, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

obj_path, stem = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(obj_path)
V, adj = [], {}
for line in open(obj_path):
    t = line.split()
    if not t:
        continue
    if t[0] == 'v':
        V.append(tuple(float(x) for x in t[1:4]))
    elif t[0] == 'l':
        idx = [int(x.split('/')[0]) - 1 for x in t[1:]]
        for a, b in zip(idx, idx[1:]):
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
# walk the loop so consecutive list entries are neighbours on the curve
order, prev, cur = [0], None, 0
while True:
    nxt = [x for x in adj[cur] if x != prev]
    prev, cur = cur, nxt[0]
    if cur == 0:
        break
    order.append(cur)
P3 = [V[i] for i in order]
n = len(P3)
sub = lambda a, b: (a[0]-b[0], a[1]-b[1], a[2]-b[2])
norm = lambda a: math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
E = [sub(P3[(i+1) % n], P3[i]) for i in range(n)]
L = [norm(e) for e in E]
turn = []
for i in range(n):
    a, b = E[i-1], E[i]
    c = max(-1.0, min(1.0, (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) / (L[i-1] * L[i])))
    turn.append(math.degrees(math.acos(c)))
mean_edge = sum(L) / n
closest = min(norm(sub(P3[i], P3[j])) for i in range(n) for j in range(i + 10, n) if (i + n - j) % n >= 10)

W = H = 720
FPS, TURN_FRAMES = 30, 360
def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()
F_BIG, F_SMALL = font(22), font(16)
c = tuple(sum(p[k] for p in P3) / n for k in range(3))
r = max(norm(sub(p, c)) for p in P3)
scale = 0.44 * W / r
label = (f"{n} vertices   turn per vertex {sum(turn)/n:.1f} deg (max {max(turn):.0f})",
         f"edge longest/shortest {max(L)/min(L):.0f}x   closest strands {closest/mean_edge:.2f} mean edges")

tmp = tempfile.mkdtemp(prefix='spincurve-')
for k in range(TURN_FRAMES):
    ang = 2 * math.pi * k / TURN_FRAMES
    ca, sa, ct, st = math.cos(ang), math.sin(ang), math.cos(0.45), math.sin(0.45)
    img = Image.new('RGB', (W, H), (16, 18, 30))
    d = ImageDraw.Draw(img)
    Q = []
    for p in P3:
        x, y, z = sub(p, c)
        x, z = ca*x + sa*z, -sa*x + ca*z
        y, z = ct*y - st*z, st*y + ct*z
        Q.append((W/2 + scale*x, H/2 - scale*y, z))
    segs = sorted(((Q[i][2] + Q[(i+1) % n][2]) / 2, Q[i], Q[(i+1) % n]) for i in range(n))
    for depth, a, b in segs:
        t = max(0.0, min(1.0, 0.5 + 0.5 * depth / r))
        col = (int(30 + 70*t), int(80 + 160*t), int(130 + 125*t))
        d.line((a[0], a[1], b[0], b[1]), fill=col, width=1 if t < 0.35 else 2 if t < 0.7 else 3)
    d.text((18, 14), title, fill=(235, 235, 245), font=F_BIG)
    d.text((18, H - 52), label[0], fill=(235, 235, 245), font=F_SMALL)
    d.text((18, H - 28), label[1], fill=(150, 160, 185), font=F_SMALL)
    img.save(os.path.join(tmp, f'{k:05d}.png'))
subprocess.run(['ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(tmp, '%05d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', stem + '.mp4'],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print(f'{stem}.mp4 {os.path.getsize(stem + ".mp4")//1024} KB | {label[0]} | {label[1]}')
