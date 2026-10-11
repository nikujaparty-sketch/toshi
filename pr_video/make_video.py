"""高知小津高校 vs 高知国際高等学校 PR動画（5秒）ジェネレーター.

ポスター画像を素材に、4カット構成の縦型ショート動画（1080x1920 / 30fps）と
効果音を生成し、ffmpeg で mp4 に書き出す。

使い方:
    python3 make_video.py <poster.png> <fonts_dir> <out.mp4> [--size 1080x1920]
"""
import math
import random
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FPS = 30
DURATION = 5.0
N_FRAMES = int(FPS * DURATION)

GOLD = (236, 196, 92)
GOLD_HI = (255, 236, 160)
PURPLE = (120, 50, 200)
WHITE = (255, 255, 255)

# カット境界（フレーム）
CUT_A = (0, 36)     # 選手アップ + キャッチコピー
CUT_B = (36, 69)    # エンブレム激突 VS
CUT_C = (69, 105)   # チーム集合 + タイトル
CUT_D = (105, 150)  # ポスター + 試合情報


# ---------------------------------------------------------------- utilities
def ease_out_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_out_back(t, s=1.9):
    t = min(max(t, 0.0), 1.0) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def prog(f, start, length):
    return min(max((f - start) / float(length), 0.0), 1.0)


def cover(img, W, H, zoom=1.0, cx=0.5, cy=0.5):
    """img を W x H に cover でフィットさせ、中心(cx,cy)・ズームで切り出す."""
    iw, ih = img.size
    scale = max(W / iw, H / ih) * zoom
    cw, ch = W / scale, H / scale
    x0 = max(min(cx * iw - cw / 2, iw - cw), 0)
    y0 = max(min(cy * ih - ch / 2, ih - ch), 0)
    return img.resize((W, H), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))


def font(fonts, name, size):
    f = ImageFont.truetype(str(fonts / name), size)
    if name.startswith("NotoSansJP"):
        try:
            f.set_variation_by_name("Black")
        except Exception:
            pass
    return f


def text_layer(text, fnt, fill, stroke=0, stroke_fill=(0, 0, 0), glow=0,
               glow_color=None, spacing=0, vertical=False, bold=0):
    """グロー・縁取り付きテキストの RGBA レイヤーを作る."""
    pad = glow * 3 + stroke + 10
    if vertical:
        boxes = [fnt.getbbox(c) for c in text]
        size = fnt.size
        w = size + pad * 2
        h = int(len(text) * (size + spacing)) + pad * 2
    else:
        bb = fnt.getbbox(text, stroke_width=stroke)
        w = bb[2] - bb[0] + pad * 2
        h = bb[3] - bb[1] + pad * 2
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    smask = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(smask)
    if vertical:
        for i, c in enumerate(text):
            bb = boxes[i]
            cw = bb[2] - bb[0]
            x = pad + (fnt.size - cw) / 2 - bb[0]
            y = pad + i * (fnt.size + spacing)
            if c in "、。":  # 縦書き句読点は右上へ
                x += fnt.size * 0.55
                y -= fnt.size * 0.55
            d.text((x, y), c, font=fnt, fill=255)
            if stroke:
                sd.text((x, y), c, font=fnt, fill=255, stroke_width=stroke)
    else:
        bb = fnt.getbbox(text, stroke_width=stroke)
        d.text((pad - bb[0], pad - bb[1]), text, font=fnt, fill=255, stroke_width=bold)
        if stroke:
            sd.text((pad - bb[0], pad - bb[1]), text, font=fnt, fill=255,
                    stroke_width=stroke + bold)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if glow:
        g = (smask if stroke else mask).filter(ImageFilter.GaussianBlur(glow))
        g = g.point(lambda v: min(255, int(v * 1.8)))
        out.paste(Image.new("RGBA", (w, h), (*(glow_color or fill), 255)), (0, 0), g)
    if stroke:
        out.paste(Image.new("RGBA", (w, h), (*stroke_fill, 255)), (0, 0), smask)
    if isinstance(fill, Image.Image):
        out.paste(fill.resize((w, h)), (0, 0), mask)
    else:
        out.paste(Image.new("RGBA", (w, h), (*fill, 255)), (0, 0), mask)
    return out


def gold_gradient(w=64, h=256):
    arr = np.zeros((h, w, 3), np.float32)
    t = np.linspace(0, 1, h)[:, None]
    top = np.array([255, 244, 190]) / 255
    mid = np.array([232, 182, 70]) / 255
    bot = np.array([160, 105, 30]) / 255
    c = np.where(t < 0.5, top + (mid - top) * (t / 0.5)[..., None][:, 0],
                 mid + (bot - mid) * ((t - 0.5) / 0.5)[..., None][:, 0])
    arr[:] = c[:, None, :] if c.ndim == 2 else c
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGBA")


def paste_t(base, layer, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    """中心座標 (cx, cy) にスケール・回転・不透明度付きで合成."""
    if alpha <= 0.001 or scale <= 0.001:
        return
    lw, lh = layer.size
    nw, nh = max(1, int(lw * scale)), max(1, int(lh * scale))
    ly = layer.resize((nw, nh), Image.BICUBIC)
    if rot:
        ly = ly.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1:
        a = ly.getchannel("A").point(lambda v: int(v * alpha))
        ly.putalpha(a)
    if _inside(base, ly, cx, cy):
        base.alpha_composite(ly, (int(cx - ly.width / 2), int(cy - ly.height / 2)))
    else:
        _composite_clipped(base, ly, cx, cy)


def _inside(base, ly, cx, cy):
    x, y = int(cx - ly.width / 2), int(cy - ly.height / 2)
    return x >= 0 and y >= 0 and x + ly.width <= base.width and y + ly.height <= base.height


def _composite_clipped(base, ly, cx, cy):
    x, y = int(cx - ly.width / 2), int(cy - ly.height / 2)
    tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    tmp.paste(ly, (x, y))
    base.alpha_composite(tmp)


def add_light(base_rgb, light_rgb, amount=1.0):
    """加算合成（numpy float32）."""
    return np.clip(base_rgb + light_rgb * amount, 0, 1)


def to_np(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")


def shake(f, hit_frames, amp=26, decay=6.0):
    dx = dy = 0.0
    for hf in hit_frames:
        if f >= hf:
            k = math.exp(-(f - hf) / decay * 2.2) * amp
            dx += math.sin((f - hf) * 2.7 + hf) * k
            dy += math.cos((f - hf) * 3.3 + hf * 0.7) * k
    return int(dx), int(dy)


def chroma(arr, shift):
    """色収差（R/B チャンネルずらし）."""
    s = int(shift)
    if s == 0:
        return arr
    out = arr.copy()
    out[:, :, 0] = np.roll(arr[:, :, 0], s, axis=1)
    out[:, :, 2] = np.roll(arr[:, :, 2], -s, axis=1)
    return out


def shape_mask(size, kind, feather):
    w, h = size
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if kind == "circle":
        d.ellipse((feather, feather, w - feather, h - feather), fill=255)
    else:  # shield: 上辺がやや丸い盾形
        d.rounded_rectangle((feather, feather, w - feather, int(h * 0.62)), radius=int(w * 0.12), fill=255)
        d.polygon([(feather, int(h * 0.55)), (w - feather, int(h * 0.55)), (w / 2, h - feather)], fill=255)
        d.ellipse((feather, int(h * 0.35), w - feather, h - feather * 3), fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather / 2))


# --------------------------------------------------------------- particles
class Sparks:
    def __init__(self, W, H, n=140, seed=7):
        rnd = random.Random(seed)
        self.W, self.H = W, H
        self.p = []
        for _ in range(n):
            self.p.append(dict(
                x=rnd.uniform(0, W), y=rnd.uniform(0, H),
                vx=rnd.uniform(-2.5, 2.5), vy=rnd.uniform(-9, -2.5),
                r=rnd.choice([2, 2, 3, 3, 4, 6]), ph=rnd.uniform(0, 6.28),
                c=rnd.choice([GOLD, GOLD_HI, GOLD, (200, 140, 255)])))
        self.burst = []

    def add_burst(self, cx, cy, n, f0, seed):
        rnd = random.Random(seed)
        for _ in range(n):
            a = rnd.uniform(0, 2 * math.pi)
            sp = rnd.uniform(8, 42)
            self.burst.append(dict(cx=cx, cy=cy, vx=math.cos(a) * sp, vy=math.sin(a) * sp,
                                   f0=f0, life=rnd.randint(12, 26), r=rnd.choice([2, 3, 4, 5]),
                                   c=rnd.choice([GOLD, GOLD_HI, WHITE])))

    def render(self, f, density=1.0):
        layer = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        cnt = int(len(self.p) * density)
        for s in self.p[:cnt]:
            x = (s["x"] + s["vx"] * f + math.sin(f * 0.15 + s["ph"]) * 12) % self.W
            y = (s["y"] + s["vy"] * f) % self.H
            tw = 0.55 + 0.45 * math.sin(f * 0.6 + s["ph"])
            c = tuple(int(v * tw) for v in s["c"])
            r = s["r"]
            d.ellipse((x - r, y - r, x + r, y + r), fill=c)
        for b in self.burst:
            t = f - b["f0"]
            if 0 <= t < b["life"]:
                drag = (1 - 0.92 ** (t + 1)) / 0.08
                x = b["cx"] + b["vx"] * drag
                y = b["cy"] + b["vy"] * drag + 0.6 * t * t
                fade = 1 - t / b["life"]
                c = tuple(int(v * fade) for v in b["c"])
                # 光の尾
                x2 = x - b["vx"] * 0.9 * fade
                y2 = y - b["vy"] * 0.9 * fade
                d.line((x2, y2, x, y), fill=c, width=b["r"])
                d.ellipse((x - b["r"], y - b["r"], x + b["r"], y + b["r"]), fill=c)
        arr = to_np(layer)
        glow = to_np(layer.filter(ImageFilter.GaussianBlur(7)))
        return arr + glow * 1.6


# ------------------------------------------------------------------- build
def main():
    args = sys.argv[1:]
    size = "1080x1920"
    if "--size" in args:
        i = args.index("--size")
        size = args[i + 1]
        del args[i:i + 2]
    poster_path, fonts_dir, out_path = Path(args[0]), Path(args[1]), Path(args[2])
    W, H = map(int, size.split("x"))
    S = W / 1080.0  # 文字・座標スケール
    tall = H / W > 1.5

    poster = Image.open(poster_path).convert("RGB")
    PW, PH = poster.size
    sx, sy = PW / 1122.0, PH / 1402.0

    def pbox(x0, y0, x1, y1):
        return poster.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))

    # 素材切り出し（元ポスター 1122x1402 基準）
    player = pbox(300, 0, 700, 418)
    team = pbox(0, 740, 1122, 1265)
    ozu = pbox(48, 22, 348, 378)
    kok = pbox(782, 78, 1090, 362)
    ozu.putalpha(shape_mask(ozu.size, "shield", 10))
    kok.putalpha(shape_mask(kok.size, "circle", 6))
    bg_blur = cover(poster, W, H).filter(ImageFilter.GaussianBlur(28 * S))
    bg_dark = to_np(bg_blur) * 0.30

    # フォント
    F_brush = lambda s: font(fonts_dir, "YujiSyuku.ttf", int(s * S))
    F_black = lambda s: font(fonts_dir, "NotoSansJP-Black.ttf", int(s * S))
    F_dela = lambda s: font(fonts_dir, "DelaGothicOne.ttf", int(s * S))
    F_anton = lambda s: font(fonts_dir, "Anton.ttf", int(s * S))
    goldfill = gold_gradient()

    # テキストレイヤー
    copy_lines = ["仲間と、", "この一戦を超えていく。"]
    copy_layers = []
    for line in copy_lines:
        chars = [text_layer(c, F_brush(78), WHITE, stroke=2, stroke_fill=(30, 0, 50),
                            glow=10, glow_color=(170, 90, 255), bold=1) for c in line]
        copy_layers.append((line, chars))

    t_murasaki = text_layer("紫", F_brush(330), goldfill, stroke=5, stroke_fill=(70, 30, 0),
                            glow=18, glow_color=(255, 190, 60), bold=int(5 * S))
    t_rest = text_layer("の誇りを、", F_brush(190), WHITE, stroke=4, stroke_fill=(40, 0, 60),
                        glow=14, glow_color=(180, 110, 255), bold=int(4 * S))
    t_rest2 = text_layer("ここに。", F_brush(250), WHITE, stroke=4, stroke_fill=(40, 0, 60),
                         glow=14, glow_color=(180, 110, 255), bold=int(4 * S))
    t_vs = text_layer("VS", F_anton(330), goldfill, stroke=6, stroke_fill=(60, 25, 0),
                      glow=26, glow_color=(255, 170, 40))
    t_ozu = text_layer("高知小津高校", F_black(84), WHITE, stroke=5, stroke_fill=(60, 10, 110),
                       glow=12, glow_color=(150, 70, 255))
    t_kok = text_layer("高知国際高等学校", F_black(72), WHITE, stroke=5, stroke_fill=(10, 30, 110),
                       glow=12, glow_color=(70, 120, 255))
    t_taikai = text_layer("高校サッカー選手権大会", F_black(58), WHITE, stroke=3,
                          stroke_fill=(40, 0, 70), glow=10, glow_color=(160, 90, 255))
    t_date = text_layer("10/18", F_anton(190), WHITE, stroke=4, stroke_fill=(40, 0, 70),
                        glow=14, glow_color=(170, 100, 255))
    t_sun = text_layer("SUN", F_anton(110), (255, 60, 70), stroke=3, stroke_fill=(60, 0, 0),
                       glow=10, glow_color=(255, 60, 60))
    t_kick = text_layer("14:00 KICK OFF", F_anton(104), goldfill, stroke=3,
                        stroke_fill=(60, 25, 0), glow=12, glow_color=(255, 170, 40))
    t_venue = text_layer("野市ふれあい広場 サッカー場 A面", F_black(52), WHITE, stroke=3,
                         stroke_fill=(30, 0, 60), glow=8, glow_color=(150, 80, 255))
    t_cheer = text_layer("小津魂、全開。応援よろしくお願いします！", F_black(50), GOLD_HI,
                         stroke=2, stroke_fill=(40, 10, 0), glow=8, glow_color=(255, 160, 40))
    t_school_en = text_layer("KOCHI OZU HIGH SCHOOL FOOTBALL", F_anton(54), (210, 180, 255),
                             glow=8, glow_color=(140, 60, 255))

    sparks = Sparks(W, H, n=int(150 * S))
    sparks.add_burst(W / 2, H * 0.42, 90, 48, 1)          # エンブレム激突
    sparks.add_burst(W * 0.25, H * (0.36 if tall else 0.30), 70, 75, 2)  # タイトル「紫」
    sparks.add_burst(W / 2, H * 0.5, 60, CUT_D[0], 3)    # ラストカット

    # 光の筋（ストリーク）用ベクトル
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1 - 0.55 * np.clip(((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2 * 0.8) * 2.4, 0, 1)
    vign = vign[..., None]
    rng = np.random.default_rng(3)
    grain = [rng.normal(0, 0.035, (H // 4, W // 4)).astype(np.float32) for _ in range(6)]

    def light_sweep(f, f0, length, angle=0.35, width=0.08, strength=0.9, color=(1.0, 0.85, 0.55)):
        t = prog(f, f0, length)
        if t <= 0 or t >= 1:
            return 0
        pos = -0.3 + 1.6 * ease_in_out(t)
        u = (xx / W) + (yy / H) * angle
        band = np.exp(-((u - pos) / width) ** 2) * strength
        return band[..., None] * np.array(color, np.float32)

    def flash(f, f0, length=6, strength=1.0, color=(1, 1, 1)):
        t = f - f0
        if t < 0 or t > length:
            return 0.0
        return strength * (1 - t / length) ** 2 * np.array(color, np.float32)

    def flare(cx, cy, rad, strength, color=(1.0, 0.8, 0.5)):
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (rad * rad)
        core = np.exp(-d2 * 3) + 0.25 * np.exp(-((yy - cy) / (rad * 0.05)) ** 2) * np.exp(-((xx - cx) / (rad * 4)) ** 2)
        return (core * strength)[..., None] * np.array(color, np.float32)

    hits = [0, 48, 75, CUT_D[0]]
    frames_dir = out_path.parent / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for f in range(N_FRAMES):
        if CUT_A[0] <= f < CUT_B[0]:
            # ---------------- CUT A: 選手アップ
            t = prog(f, 0, CUT_B[0])
            z = 1.18 - 0.14 * ease_out_cubic(t)
            img = cover(player, W, H, zoom=z, cx=0.52, cy=0.45 - 0.04 * t)
            arr = to_np(img)
            # 紫のスモーク感：下部をグラデで沈める
            fade_bot = np.clip((yy / H - 0.55) / 0.45, 0, 1)[..., None]
            arr = arr * (1 - 0.75 * fade_bot) + fade_bot * np.array([0.18, 0.05, 0.32]) * 0.75
            arr *= 0.55 + 0.45 * ease_out_cubic(prog(f, 0, 8))
            base = to_img(arr).convert("RGBA")
            # キャッチコピー：縦書き 1文字ずつ（右→左の2行）
            col_x = [W * 0.86, W * 0.73]
            y0 = H * (0.12 if tall else 0.08)
            k = 0
            for li, (line, chars) in enumerate(copy_layers):
                for ci, ch in enumerate(chars):
                    appear = 3 + k * 1.6
                    p = prog(f, appear, 5)
                    if p > 0:
                        cy_ = y0 + ci * 84 * S
                        if line[ci] in "、。":
                            cy_ -= 40 * S
                        paste_t(base, ch, col_x[li] + (1 - p) * 30 * S, cy_,
                                scale=1.0 + 0.6 * (1 - ease_out_cubic(p)), alpha=p)
                    k += 1
            arr = to_np(base)
            arr = arr + light_sweep(f, 4, 26, angle=-0.5, width=0.06, strength=0.55)
            arr = arr + flash(f, 0, 10, 1.0)
            sp = sparks.render(f, 0.6)
            arr = arr + sp
            cshift = 0
        elif CUT_B[0] <= f < CUT_C[0]:
            # ---------------- CUT B: エンブレム激突
            lf = f - CUT_B[0]
            arr = bg_dark.copy()
            # 中央に紫の放射光
            pulse = 0.5 + 0.5 * ease_out_cubic(prog(f, 48, 8))
            arr = arr + flare(W / 2, H * 0.42, W * 0.7, 0.35 * pulse, (0.55, 0.25, 1.0))
            base = to_img(arr).convert("RGBA")
            em_y = H * 0.42
            p = ease_out_back(prog(f, CUT_B[0], 12), 1.2)
            gap = W * 0.25
            esc = 1.15 * S * (1.0 + 0.08 * math.exp(-max(0, f - 48) / 4.0) * (f >= 48))
            # 衝突後に少し離れる
            sep = 0 if f < 48 else 18 * S * ease_out_cubic(prog(f, 48, 10))
            lx = -W * 0.4 + (W / 2 - gap - sep + W * 0.4) * p
            rx = W * 1.4 - (W * 1.4 - (W / 2 + gap + sep)) * p
            paste_t(base, ozu, lx, em_y, scale=esc, rot=(1 - p) * 25)
            paste_t(base, kok, rx, em_y, scale=esc * 1.02, rot=-(1 - p) * 25)
            # VS
            pv = prog(f, 47, 9)
            if pv > 0:
                paste_t(base, t_vs, W / 2, em_y + 10 * S, scale=(2.6 - 1.6 * ease_out_back(pv, 1.6)) * 0.62,
                        alpha=min(1, pv * 3))
            # 校名
            pn = prog(f, 52, 8)
            if pn > 0:
                ny = em_y + 300 * S
                paste_t(base, t_ozu, W / 2 - (1 - ease_out_cubic(pn)) * W * 0.6, ny,
                        alpha=pn, scale=0.95)
                paste_t(base, t_kok, W / 2 + (1 - ease_out_cubic(pn)) * W * 0.6, ny + 130 * S,
                        alpha=pn, scale=0.95)
                vs_small = prog(f, 54, 6)
                if vs_small > 0:
                    paste_t(base, t_vs, W / 2, ny + 66 * S, scale=0.18, alpha=vs_small)
            pt = prog(f, 40, 8)
            if pt > 0:
                paste_t(base, t_taikai, W / 2, em_y - 300 * S, alpha=pt,
                        scale=0.9 + 0.1 * ease_out_cubic(pt))
            arr = to_np(base)
            # スピードライン（衝突前）
            if f < 48:
                streak = np.exp(-((yy - em_y) / (H * 0.13)) ** 2)[..., None] * 0.12 * (lf / 12)
                arr = arr + streak * np.array([0.7, 0.5, 1.0])
            arr = arr + flash(f, 48, 5, 0.9, (1.0, 0.95, 0.85))
            arr = arr + flash(f, CUT_B[0], 4, 0.6)
            arr = arr + sparks.render(f, 0.5)
            cshift = 14 * S * math.exp(-max(0, f - 48) / 3) if f >= 48 else 0
        elif CUT_C[0] <= f < CUT_D[0]:
            # ---------------- CUT C: チーム集合 + タイトル
            lf = f - CUT_C[0]
            t = prog(f, CUT_C[0], CUT_D[0] - CUT_C[0])
            img = cover(team, W, H, zoom=1.0 + 0.06 * t, cx=0.5 - 0.08 * t + 0.04, cy=0.55)
            arr = to_np(img)
            # 上部を暗く（タイトル可読性）
            topd = np.clip(1 - yy / (H * 0.62), 0, 1)[..., None]
            arr = arr * (1 - 0.65 * topd) + topd * np.array([0.12, 0.03, 0.22]) * 0.65
            arr = arr + flare(W * (0.85 - 0.3 * t), H * 0.12, W * 0.45, 0.45, (0.9, 0.75, 1.0))
            base = to_img(arr).convert("RGBA")
            ty = H * (0.27 if tall else 0.22)
            # 「紫」スラム
            pm = prog(f, 72, 5)
            if pm > 0:
                paste_t(base, t_murasaki, W * 0.19, ty, scale=0.9 * (2.4 - 1.4 * ease_out_back(pm, 1.4)),
                        alpha=min(1, pm * 2.5), rot=(1 - pm) * -12)
            pr = prog(f, 77, 6)
            if pr > 0:
                paste_t(base, t_rest, W * 0.615 + (1 - ease_out_cubic(pr)) * 120 * S, ty + 40 * S,
                        scale=0.84, alpha=pr)
            pk = prog(f, 82, 6)
            if pk > 0:
                paste_t(base, t_rest2, W * 0.58, ty + 290 * S, scale=0.9 * (1.5 - 0.5 * ease_out_back(pk, 1.6)),
                        alpha=min(1, pk * 2))
            # 金のブラシ下線
            pl = prog(f, 84, 8)
            if pl > 0:
                ld = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(ld)
                x_end = W * 0.08 + (W * 0.84) * ease_out_cubic(pl)
                yl = ty + 450 * S
                for i in range(7):
                    off = (i - 3) * 3 * S
                    dd.line((W * 0.08, yl + off + i * 0.6, x_end, yl - 18 * S + off), fill=(*GOLD, 210 - abs(i - 3) * 50),
                            width=int(5 * S))
                ld = Image.alpha_composite(ld.filter(ImageFilter.GaussianBlur(6 * S)), ld)
                base.alpha_composite(ld)
            arr = to_np(base)
            arr = arr + flash(f, CUT_C[0], 5, 0.8)
            arr = arr + flash(f, 75, 5, 0.7, (1.0, 0.85, 0.5))
            arr = arr + light_sweep(f, 86, 16, angle=0.25, width=0.05, strength=0.5)
            arr = arr + sparks.render(f, 0.8)
            cshift = 10 * S * math.exp(-max(0, f - 75) / 3) if f >= 75 else 0
        else:
            # ---------------- CUT D: ポスター + 試合情報
            lf = f - CUT_D[0]
            arr = bg_dark.copy() * 0.9
            base = to_img(arr).convert("RGBA")
            pz = ease_out_cubic(prog(f, CUT_D[0], 14))
            pw_target = W * (0.80 if tall else 0.62)
            psc = pw_target / PW * (1.12 - 0.12 * pz) * (1 + 0.015 * prog(f, CUT_D[0] + 14, 31))
            pimg = poster.convert("RGBA")
            # 影
            py = H * (0.355 if tall else 0.40)
            shadow = Image.new("RGBA", (int(PW * psc) + 80, int(PH * psc) + 80), (0, 0, 0, 0))
            ImageDraw.Draw(shadow).rectangle((40, 40, shadow.width - 40, shadow.height - 40), fill=(10, 0, 25, 200))
            shadow = shadow.filter(ImageFilter.GaussianBlur(24))
            paste_t(base, shadow, W / 2, py + 16 * S, alpha=pz)
            # 金枠
            frame = Image.new("RGBA", (PW + 16, PH + 16), (*GOLD, 255))
            frame.paste(pimg, (8, 8))
            paste_t(base, frame, W / 2, py, scale=psc, alpha=min(1, pz * 1.4))
            ph_disp = PH * psc
            if tall:
                pe = prog(f, CUT_D[0] + 2, 8)
                paste_t(base, t_school_en, W / 2, py - ph_disp / 2 - 70 * S, scale=0.9,
                        alpha=pe)
                info_y = py + ph_disp / 2 + 115 * S
                pd = prog(f, CUT_D[0] + 6, 7)
                if pd > 0:
                    e = ease_out_back(pd, 1.5)
                    paste_t(base, t_date, W / 2 - 140 * S, info_y, scale=0.9 * (1.6 - 0.6 * e), alpha=min(1, pd * 2))
                    paste_t(base, t_sun, W / 2 + 140 * S, info_y + 22 * S, scale=0.9 * (1.6 - 0.6 * e), alpha=min(1, pd * 2))
                pko = prog(f, CUT_D[0] + 10, 7)
                if pko > 0:
                    paste_t(base, t_kick, W / 2 + (1 - ease_out_cubic(pko)) * W * 0.5, info_y + 160 * S,
                            scale=0.9, alpha=pko)
                pv_ = prog(f, CUT_D[0] + 14, 7)
                if pv_ > 0:
                    paste_t(base, t_venue, W / 2, info_y + 270 * S, scale=0.95, alpha=pv_)
                pc = prog(f, CUT_D[0] + 20, 8)
                if pc > 0:
                    paste_t(base, t_cheer, W / 2, info_y + 360 * S, alpha=pc * (0.75 + 0.25 * math.sin(lf * 0.5)))
            else:
                rx_ = W * 0.82
                for i, (ly, yv, sc) in enumerate([(t_date, 0.30, 0.55), (t_sun, 0.40, 0.55), (t_kick, 0.52, 0.42),
                                                  (t_venue, 0.64, 0.33), (t_cheer, 0.74, 0.3)]):
                    pp = prog(f, CUT_D[0] + 6 + i * 4, 7)
                    if pp > 0:
                        paste_t(base, ly, rx_ + (1 - ease_out_cubic(pp)) * 200 * S, H * yv, scale=sc * 1.7, alpha=pp)
            arr = to_np(base)
            # ポスター上の光沢スイープ
            arr = arr + light_sweep(f, CUT_D[0] + 14, 18, angle=0.4, width=0.05, strength=0.45)
            arr = arr + flash(f, CUT_D[0], 7, 1.1, (1.0, 0.92, 0.8))
            arr = arr + sparks.render(f, 0.9)
            # 最後は軽くフェードアウト
            arr *= 1 - 0.35 * ease_in_out(prog(f, N_FRAMES - 4, 4))
            cshift = 8 * S * math.exp(-lf / 3)

        # 共通ポスト処理
        arr = chroma(arr, cshift)
        dx, dy = shake(f, hits, amp=24 * S)
        if dx or dy:
            arr = np.roll(arr, (dy, dx), axis=(0, 1))
        arr = arr * vign
        g = np.asarray(Image.fromarray(((grain[f % 6] + 0.5) * 255).clip(0, 255).astype(np.uint8)).resize((W, H)),
                       np.float32) / 255 - 0.5
        arr = arr + g[..., None] * 0.6
        # 軽いトーン：ハイライトを少し金寄りに
        arr = np.clip(arr, 0, 1)
        arr = arr ** 0.95
        to_img(arr).save(frames_dir / f"{f:04d}.png", compress_level=1)
        if f % 15 == 0:
            print(f"frame {f}/{N_FRAMES}", flush=True)

    audio = out_path.parent / "_sfx.wav"
    make_audio(audio, hits_sec=[h / FPS for h in hits])
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(frames_dir / "%04d.png"), "-i", str(audio),
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out_path)
    ], check=True)
    print("written", out_path)


# -------------------------------------------------------------------- audio
def make_audio(path, hits_sec, sr=44100):
    n = int(sr * DURATION)
    t = np.arange(n) / sr
    out = np.zeros(n, np.float64)
    rng = np.random.default_rng(11)

    def boom(t0, gain=1.0, f_hi=120, f_lo=38, dur=1.1):
        i0 = int(t0 * sr)
        m = min(n - i0, int(dur * sr))
        if m <= 0:
            return
        tt = np.arange(m) / sr
        freq = f_lo + (f_hi - f_lo) * np.exp(-tt * 9)
        ph = 2 * np.pi * np.cumsum(freq) / sr
        env = np.exp(-tt * 3.2)
        body = np.sin(ph) * env
        click = rng.normal(0, 1, m) * np.exp(-tt * 60) * 0.5
        crack = rng.normal(0, 1, m) * np.exp(-tt * 9) * 0.18
        out[i0:i0 + m] += gain * (body + click + crack)

    def whoosh(t0, t1, gain=0.35):
        i0, i1 = int(t0 * sr), int(t1 * sr)
        m = i1 - i0
        noise = rng.normal(0, 1, m)
        # 簡易ローパス（移動平均の段数で明るさ変化）
        k = np.linspace(40, 4, m).astype(int)
        cs = np.cumsum(np.concatenate([[0], noise]))
        idx = np.arange(m)
        lo = np.maximum(idx - k, 0)
        filt = (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)
        env = np.linspace(0, 1, m) ** 2.2
        out[i0:i1] += gain * filt * env * 6

    def riser(t0, t1, gain=0.12):
        i0, i1 = int(t0 * sr), int(t1 * sr)
        tt = np.arange(i1 - i0) / sr
        L = (t1 - t0)
        freq = 220 * (2 ** (tt / L * 2))
        ph = 2 * np.pi * np.cumsum(freq) / sr
        env = (tt / L) ** 1.5
        sig = (np.sin(ph) + 0.5 * np.sin(ph * 1.5) + 0.3 * np.sin(ph * 2.01)) * env
        out[i0:i1] += gain * sig

    def shimmer(t0, dur, gain=0.08):
        i0 = int(t0 * sr)
        m = min(n - i0, int(dur * sr))
        tt = np.arange(m) / sr
        sig = np.zeros(m)
        for fr in [1318.5, 1567.98, 1975.5, 2637.0, 3135.9]:
            sig += np.sin(2 * np.pi * fr * tt + rng.uniform(0, 6)) * (0.5 + 0.5 * np.sin(2 * np.pi * 9 * tt))
        out[i0:i0 + m] += gain * sig * np.exp(-tt * 1.6)

    def kick(t0, gain=0.45):
        boom(t0, gain, f_hi=150, f_lo=48, dur=0.35)

    # 鼓動のようなビート（ 128BPM ）
    beat = 60 / 128
    bt = 0.0
    while bt < DURATION - 0.6:
        if not any(abs(bt - h) < 0.12 for h in hits_sec):
            kick(bt, 0.38)
        bt += beat
    riser(0.05, hits_sec[1] - 0.02, 0.10)
    whoosh(0.9, hits_sec[1], 0.5)
    whoosh(2.15, hits_sec[2], 0.4)
    whoosh(3.15, hits_sec[3], 0.4)
    boom(hits_sec[0], 0.6)
    boom(hits_sec[1], 1.0)
    boom(hits_sec[2], 0.9)
    boom(hits_sec[3], 1.0, dur=1.6)
    shimmer(hits_sec[3] + 0.1, 1.5)
    shimmer(hits_sec[2] + 0.1, 0.8, 0.05)
    # フェードアウト
    fo = int(0.35 * sr)
    out[-fo:] *= np.linspace(1, 0, fo)
    out = np.tanh(out * 1.4) * 0.85
    pcm = (out * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
