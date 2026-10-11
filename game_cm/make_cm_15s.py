"""『ぼくと、あの秋。』15秒CM（PS5 ゲームCM風）ジェネレーター.

横型 1920x1080 / 30fps / シネマスコープ風の黒帯 / 秋っぽいBGM + トレーラー風のヒット音.

カット構成:
    0.0- 1.2s  導入：暗い青の空間に光の筋と粒子「IKEMOTO STUDIO PRESENTS」
    1.2- 3.6s  キーアート（夕焼けの縁側）  「どこか懐かしい、あの秋を、もう一度。」
               人物と背景を分けた2.5Dカメラ / 呼吸 / 首の動き / 柴犬のハァハァ・まばたき / ススキ / 赤とんぼ
    3.6- 4.2s  黒カード「何もない町。」
    4.2- 6.2s  思い出の写真が次々と落ちてくる（川・自転車・秋祭り・花火）
    6.2- 6.8s  黒カード「あの頃の、なかまたち。」
    6.8- 9.6s  切り抜いた4人が順にジャンプして登場 → それぞれの仕草
               （タイチ：アゴしゃくり / やっさん：笑い・指差し / トシヒコ：網振り / クルミ：のぞき込み・まばたき）
    9.6-11.8s  集合写真  「何もなかった。だから、全部あった。」
   11.8-15.0s  エンドカード：タイトル（光が走る）/ PS5 / 発売日 近日発表 / CERO 審査予定 / ©

本編カットには「※画面は開発中のものです。」を表示する。
人物の切り抜きと背景は make_layers.py で事前に作る（assets/layers/）。

使い方:
    python3 make_cm_15s.py <assets_dir> <fonts_dir> <out.mp4>
      assets_dir: keyart.png（ポスター） / group.png / layers/（make_layers.py の出力）
      fonts_dir : ShipporiMincho-800.ttf / KleeOne-600.ttf / InstrumentSans-Bold.ttf / InstrumentSans-Regular.ttf / ipagp.ttf
"""
import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 30
DURATION = 15.0
N = int(FPS * DURATION)
SR = 44100

# カット（開始, 終了）フレーム
CUTS = {"intro": (0, 36), "open": (36, 108), "card1": (108, 126), "photos": (126, 186), "card2": (186, 204),
        "chars": (204, 288), "group": (288, 354), "title": (354, N)}
# 各カットに入るときのクロスフェード長（0 = PS5 トレーラー風のハードカット）
XFS = {"open": 8, "card1": 0, "photos": 0, "card2": 0, "chars": 0, "group": 6, "title": 0}
XF = 6
DROPS = [128, 138, 148, 158]  # 写真が落ちるフレーム（効果音と同期）
DROP_LEN = 8
BAR = 110  # シネマスコープ風の黒帯の高さ
FOOTAGE = ("open", "photos", "chars", "group")

CREAM = (250, 240, 222)
BROWN = (74, 46, 30)
ORANGE = (236, 128, 52)


# ---------------------------------------------------------------- 基本ユーティリティ
def clamp01(x):
    return max(0.0, min(1.0, x))


def prog(f, f0, length):
    return clamp01((f - f0) / float(length))


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out_back(t, s=1.4):
    t -= 1
    return t * t * ((s + 1) * t + s) + 1


def to_np(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def cover(img, w, h):
    s = max(w / img.width, h / img.height)
    r = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


def kenburns(img, f, f0, f1, z0, z1, cx0=0.5, cy0=0.5, cx1=0.5, cy1=0.5):
    """img（出力より大きめ）から、ズーム z と中心を補間して W x H を切り出す."""
    t = ease_in_out(prog(f, f0, f1 - f0))
    z = z0 + (z1 - z0) * t
    cx, cy = cx0 + (cx1 - cx0) * t, cy0 + (cy1 - cy0) * t
    bw, bh = img.width / z, img.width / z * H / W
    if bh > img.height / z:
        bh = img.height / z
        bw = bh * W / H
    x0 = min(max(cx * img.width - bw / 2, 0), img.width - bw)
    y0 = min(max(cy * img.height - bh / 2, 0), img.height - bh)
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh))


def paste_alpha(base, layer, x, y, alpha=1.0):
    """base: float RGB 配列, layer: RGBA Image. 左上 (x, y) に alpha 合成."""
    if alpha <= 0:
        return base
    la = np.asarray(layer, dtype=np.float32) / 255.0
    lh, lw = la.shape[:2]
    x, y = int(round(x)), int(round(y))
    bx0, by0 = max(x, 0), max(y, 0)
    bx1, by1 = min(x + lw, base.shape[1]), min(y + lh, base.shape[0])
    if bx0 >= bx1 or by0 >= by1:
        return base
    sub = la[by0 - y:by1 - y, bx0 - x:bx1 - x]
    a = sub[..., 3:4] * alpha
    base[by0:by1, bx0:bx1] = base[by0:by1, bx0:bx1] * (1 - a) + sub[..., :3] * a
    return base


def text_layer(text, fnt, fill, shadow=(30, 15, 5), shadow_blur=10, shadow_alpha=0.75,
               glow=None, pad=40, tracking=0):
    """1行のテキストを RGBA 画像で返す（ぼかし影 / グロー付き）."""
    widths = [fnt.getlength(c) for c in text]
    tw = int(sum(widths) + tracking * (len(text) - 1))
    asc, desc = fnt.getmetrics()
    th = asc + desc
    size = (tw + pad * 2, th + pad * 2)
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    x = pad
    for c, w in zip(text, widths):
        d.text((x, pad), c, font=fnt, fill=255)
        x += w + tracking
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    if shadow is not None:
        sh = mask.filter(ImageFilter.GaussianBlur(shadow_blur)).point(lambda v: int(v * shadow_alpha))
        out.paste(Image.new("RGBA", size, shadow + (255,)), (0, 0), sh)
    if glow is not None:
        gl = mask.filter(ImageFilter.GaussianBlur(glow[1])).point(lambda v: min(255, int(v * 1.6)))
        out.paste(Image.new("RGBA", size, glow[0] + (255,)), (0, 0), gl)
    out.paste(Image.new("RGBA", size, fill + (255,)), (0, 0), mask)
    return out


def char_layers(text, fnt, fill, **kw):
    """1文字ずつのレイヤーと x オフセット（文字送り演出用）."""
    layers, xs, x = [], [], 0.0
    for c in text:
        layers.append(text_layer(c, fnt, fill, **kw))
        xs.append(x)
        x += fnt.getlength(c) + kw.get("tracking", 0)
    return layers, xs, x


def draw_chars(base, layers, xs, x0, y0, f, f_start, step=2, dur=10, rise=18):
    pad = 40
    for i, (lay, cx) in enumerate(zip(layers, xs)):
        t = prog(f, f_start + i * step, dur)
        if t <= 0:
            continue
        e = ease_out_cubic(t)
        paste_alpha(base, lay, x0 + cx - pad, y0 - pad + (1 - e) * rise, e)
    return base


# ---------------------------------------------------------------- 紅葉パーティクル
def maple_sprite(size, color):
    """もみじの葉（7裂）を描いた RGBA."""
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = s / 2, s * 0.56
    pts = []
    lobes = [(-90, 1.0), (-45, 0.9), (0, 0.72), (35, 0.42), (145, 0.42), (180, 0.72), (225, 0.9)]
    for ang, ln in lobes:
        a = math.radians(ang)
        a_l, a_r = a - math.radians(14), a + math.radians(14)
        r = s * 0.45 * ln
        pts += [(cx + r * 0.42 * math.cos(a_l - 0.2), cy + r * 0.42 * math.sin(a_l - 0.2)),
                (cx + r * 0.78 * math.cos(a_l), cy + r * 0.78 * math.sin(a_l)),
                (cx + r * math.cos(a), cy + r * math.sin(a)),
                (cx + r * 0.78 * math.cos(a_r), cy + r * 0.78 * math.sin(a_r))]
    d.polygon(pts, fill=color + (255,))
    vein = tuple(int(c * 0.72) for c in color) + (200,)
    for ang, ln in lobes:
        a = math.radians(ang)
        r = s * 0.45 * ln * 0.85
        d.line([(cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a))], fill=vein, width=max(2, s // 60))
    d.line([(cx, cy), (cx + s * 0.05, cy + s * 0.4)], fill=vein, width=max(3, s // 40))
    return img.resize((size, size), Image.LANCZOS)


class Leaves:
    def __init__(self, n=26, seed=7):
        rng = np.random.default_rng(seed)
        colors = [(198, 52, 30), (226, 92, 34), (238, 140, 40), (214, 168, 48), (176, 40, 34)]
        self.items = []
        for i in range(n):
            size = int(rng.uniform(34, 92))
            depth = size / 92.0
            self.items.append(dict(
                spr=maple_sprite(size, colors[i % len(colors)]),
                x=rng.uniform(-100, W + 100), y=rng.uniform(-H, H),
                vy=rng.uniform(2.2, 4.2) * (0.6 + depth), vx=rng.uniform(0.6, 2.2),
                sway=rng.uniform(30, 80), ph=rng.uniform(0, 6.28), fr=rng.uniform(0.04, 0.09),
                rot=rng.uniform(0, 360), vr=rng.uniform(-6, 6), blur=depth > 0.85))

    def draw(self, base, f, alpha=1.0):
        for it in self.items:
            y = (it["y"] + it["vy"] * f) % (H + 260) - 130
            x = (it["x"] + it["vx"] * f + it["sway"] * math.sin(it["ph"] + it["fr"] * f)) % (W + 260) - 130
            # 葉がひらひら裏返る感じ：横方向に潰す
            flip = abs(math.cos(it["ph"] + f * it["fr"] * 1.7))
            spr = it["spr"]
            spr = spr.resize((max(4, int(spr.width * (0.35 + 0.65 * flip))), spr.height))
            spr = spr.rotate(it["rot"] + it["vr"] * f, expand=True, resample=Image.BICUBIC)
            if it["blur"]:
                spr = spr.filter(ImageFilter.GaussianBlur(2))
            paste_alpha(base, spr, x - spr.width / 2, y - spr.height / 2, alpha * 0.92)
        return base


def dragonfly_sprites(size=110):
    """赤とんぼ（羽ばたき2コマ）. 右向き."""
    out = []
    for phase in (0, 1):
        s = size * 4
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, cy = s * 0.5, s * 0.5
        wing = (255, 245, 230, 110) if phase == 0 else (255, 245, 230, 70)
        span = s * (0.42 if phase == 0 else 0.30)
        for dx, dy in ((-0.02, -1), (0.08, -1), (-0.02, 1), (0.08, 1)):
            x = cx + dx * s
            d.ellipse([x - s * 0.05, cy + dy * span - s * 0.035 if dy < 0 else cy,
                       x + s * 0.05, cy if dy < 0 else cy + dy * span + s * 0.035], fill=wing,
                      outline=(120, 70, 40, 90), width=3)
        d.line([(cx - s * 0.38, cy), (cx + s * 0.12, cy)], fill=(196, 52, 30, 255), width=int(s * 0.035))
        d.ellipse([cx + s * 0.08, cy - s * 0.04, cx + s * 0.18, cy + s * 0.04], fill=(150, 40, 25, 255))
        out.append(img.resize((size, size), Image.LANCZOS))
    return out


class Dragonflies:
    def __init__(self, specs):
        self.spr = dragonfly_sprites()
        self.specs = specs  # (x0, y0, vx, vy, scale, phase)

    def draw(self, base, f, f0=0):
        lf = f - f0
        for x0, y0, vx, vy, sc, ph in self.specs:
            # ふわっと進んでは止まる（ホバリング）動き
            hover = 0.5 + 0.5 * math.sin(lf * 0.11 + ph)
            x = x0 + vx * (lf + 18 * math.sin(lf * 0.11 + ph)) + 12 * math.sin(lf * 0.3 + ph)
            y = y0 + vy * lf + 22 * math.sin(lf * 0.17 + ph * 2) * hover
            dxdt = vx * (1 + 2 * math.cos(lf * 0.11 + ph))
            dydt = vy + 22 * 0.17 * math.cos(lf * 0.17 + ph * 2) * hover
            ang = -math.degrees(math.atan2(dydt, dxdt if abs(dxdt) > 1e-3 else 1e-3)) * 0.4
            spr = self.spr[(lf // 1 + int(ph * 10)) % 2]
            if vx < 0:
                spr = spr.transpose(Image.FLIP_LEFT_RIGHT)
                ang = -ang
            spr = spr.resize((int(spr.width * sc), int(spr.height * sc))).rotate(ang, resample=Image.BICUBIC)
            paste_alpha(base, spr, x - spr.width / 2, y - spr.height / 2, 0.95)
        return base


def breathe(layer, amount, anchor_y):
    """RGBA レイヤーを anchor_y（下端側）を固定して縦に少し伸縮させる＝呼吸."""
    if abs(amount) < 1e-4:
        return layer
    h2 = int(round(layer.height * (1 + amount)))
    r = layer.resize((layer.width, h2), Image.BICUBIC)
    top = int(round(anchor_y * (1 + amount) - anchor_y))
    return r.crop((0, top, layer.width, top + layer.height))


def sway_grass(arr, f, region, amp=7.0):
    """region=(x0, y0, x1, y1) 内を風で横に揺らす（上ほど大きく揺れる）."""
    x0, y0, x1, y1 = region
    sub = np.ascontiguousarray(arr[y0:y1, x0:x1])
    hh, ww = sub.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    w = (1 - yy / hh) ** 1.5 * np.clip(xx / 60, 0, 1) * np.clip((ww - xx) / 60, 0, 1)
    gust = 0.6 + 0.4 * math.sin(f * 0.05)
    dx = amp * gust * w * np.sin(f * 0.19 - xx * 0.012 + yy * 0.004)
    arr[y0:y1, x0:x1] = cv2.remap(sub, xx + dx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return arr


# ---------------------------------------------------------------- パペット変形（キャラ自体の動き）
# 切り抜いた人物画像をボーンごとに滑らかに変形させる。座標はレイヤー元画像のピクセル。
#   ("rot",   中心, (rx, ry), 支点, 角度deg)   … 首かしげ・うなずき・腕振り・網振り
#   ("move",  中心, (rx, ry), (dx, dy))        … 口のパクパク（犬のハァハァ）
#   ("blink", 目の中心, (rx, ry), 閉じ具合0-1) … まばたき
def blink_amount(lf, times, dur=5):
    """times のフレームでまばたき（閉じ→開く）."""
    a = 0.0
    for t0 in times:
        d = lf - t0
        if 0 <= d <= dur:
            a = max(a, math.sin(math.pi * d / dur))
    return a


def soft_ellipse(xx, yy, c, r, inner=0.55):
    d = np.sqrt(((xx - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2)
    t = np.clip((d - inner) / (1 - inner), 0, 1)
    return 1 - t * t * (3 - 2 * t)


_grid_cache = {}


def puppet(layer, bones, scale=1.0, off=(0.0, 0.0)):
    """layer: RGBA Image. bones の座標は (元座標 - off) * scale でこのレイヤーに対応させる."""
    w, h = layer.size
    if (w, h) not in _grid_cache:
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        _grid_cache[(w, h)] = (gx, gy)
    gx, gy = _grid_cache[(w, h)]
    mx, my = gx.copy(), gy.copy()
    T = lambda pt: ((pt[0] - off[0]) * scale, (pt[1] - off[1]) * scale)
    src = np.asarray(layer)
    # まばたきは変形前の画像に描く（首の回転に目がついていくように）
    blinks = [b for b in bones if b[0] == "blink" and b[3] > 0.02]
    if blinks:
        src = src.astype(np.float32)
        for _, c, r, k in blinks:
            c, r = T(c), (r[0] * scale, r[1] * scale)
            x0, x1 = int(max(0, c[0] - r[0] * 1.3)), int(min(w, c[0] + r[0] * 1.3 + 1))
            y0, y1 = int(max(0, c[1] - r[1] * 1.6)), int(min(h, c[1] + r[1] * 1.6 + 1))
            lid_y = int(max(0, c[1] - r[1] * 1.55))
            if x0 >= x1 or y0 >= y1:
                continue
            xx, yy = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
            # まぶた：目のすぐ上の肌の行を縦に引き伸ばして目を覆う
            lid = src[lid_y:lid_y + 2, x0:x1].mean(0, keepdims=True)
            m = soft_ellipse(xx, yy, c, (r[0] * 1.2, r[1] * 1.45), inner=0.45)[..., None] * k
            src[y0:y1, x0:x1, :3] = src[y0:y1, x0:x1, :3] * (1 - m) + lid[..., :3] * m
            # まつ毛の線（閉じた目の下側の弧）
            u = (xx - c[0]) / r[0]
            ly = c[1] + r[1] * 0.45 * (1 - u ** 2)
            line = np.clip(1 - np.abs(yy - ly) / max(1.0, 0.9 * scale), 0, 1) * (np.abs(u) < 1) * k ** 2
            src[y0:y1, x0:x1, :3] *= (1 - 0.6 * line)[..., None]
    for b in bones:
        kind, c, r = b[0], T(b[1]), (b[2][0] * scale, b[2][1] * scale)
        if kind == "blink":
            continue
        ext = 1.0
        x0, x1 = int(max(0, c[0] - r[0] * ext)), int(min(w, c[0] + r[0] * ext + 1))
        y0, y1 = int(max(0, c[1] - r[1] * ext)), int(min(h, c[1] + r[1] * ext + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        xx, yy = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
        if kind == "rot":
            pv, ang = T(b[3]), b[4]
            if abs(ang) < 1e-3:
                continue
            wgt = soft_ellipse(xx, yy, c, r)
            th = -np.radians(ang) * wgt
            dx, dy = xx - pv[0], yy - pv[1]
            mx[y0:y1, x0:x1] += (pv[0] + dx * np.cos(th) - dy * np.sin(th)) - xx
            my[y0:y1, x0:x1] += (pv[1] + dx * np.sin(th) + dy * np.cos(th)) - yy
        elif kind == "move":
            wgt = soft_ellipse(xx, yy, c, r, inner=0.35)
            mx[y0:y1, x0:x1] -= wgt * b[3][0] * scale
            my[y0:y1, x0:x1] -= wgt * b[3][1] * scale
    arr = cv2.remap(np.clip(src, 0, 255).astype(np.uint8), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0))
    return Image.fromarray(arr)


def osc(lf, period, phase=0.0):
    return math.sin(2 * math.pi * lf / period + phase)


def bones_key(f):
    """縁側：主人公は空を見上げながらゆっくり首を動かし、柴犬は首をかしげてハァハァ."""
    return [
        ("rot", (590, 100), (85, 80), (585, 165), 2.5 * osc(f, 70, 0.5)),
        ("rot", (665, 315), (75, 62), (705, 372), 5.0 * osc(f, 46, 1.0) + (6 if 50 < f < 75 else 0) *
         math.sin(math.pi * clamp01((f - 50) / 25))),
        ("move", (645, 342), (32, 20), (0, 2.6 * (0.5 + 0.5 * osc(f, 6)))),
        ("blink", (670, 292), (8, 4), blink_amount(f, [28, 78])),
    ]


def bones_char(i, lf, land):
    d = lf - land  # 着地からのフレーム数
    on = clamp01(d / 6) if d >= 0 else 0.0
    if i == 0:  # タイチ：着地で「ついてこいや！」とアゴをしゃくる
        jerk = -9 * math.exp(-max(d - 3, 0) / 7) * math.sin(clamp01((d - 3) / 6) * math.pi) if d >= 3 else 0
        return [("rot", (198, 80), (75, 90), (200, 155), jerk + 2 * on * osc(lf, 50)),
                ("blink", (163, 90), (11, 5), blink_amount(d, [30, 66])),
                ("blink", (205, 86), (11, 5), blink_amount(d, [30, 66]))]
    if i == 1:  # やっさん：笑って頭を揺らし、指差しの手をクイクイ、ジュースを振る
        return [("rot", (212, 90), (78, 82), (215, 168), on * 4 * osc(lf, 9)),
                ("rot", (150, 245), (80, 48), (45, 290), on * 8 * osc(lf, 14)),
                ("rot", (282, 200), (48, 62), (300, 330), on * 6 * osc(lf, 11, 1.0)),
                ("blink", (207, 94), (11, 5), blink_amount(d, [40])),
                ("blink", (252, 103), (11, 5), blink_amount(d, [40]))]
    if i == 2:  # トシヒコ：虫取り網をゆらゆら、首をかしげる
        return [("rot", (60, 200), (95, 175), (62, 455), on * 9 * osc(lf, 30)),
                ("rot", (218, 95), (72, 82), (218, 172), on * 4 * osc(lf, 40, 2.0)),
                ("blink", (199, 110), (10, 5), blink_amount(d, [22, 58])),
                ("blink", (241, 105), (10, 5), blink_amount(d, [22, 58]))]
    # クルミちゃん：上半身をゆらしてのぞき込み、首をかしげる
    return [("rot", (170, 165), (150, 190), (160, 330), on * 2.5 * osc(lf, 44, 1.0)),
            ("rot", (188, 85), (68, 78), (182, 150), on * 5 * osc(lf, 44, 2.6)),
            ("blink", (173, 82), (10, 5), blink_amount(d, [15, 52])),
            ("blink", (208, 75), (10, 5), blink_amount(d, [15, 52]))]


def bones_group(lf):
    """集合写真：それぞれ首を動かし、やっさんは親指をクイッ、柴犬はハァハァ."""
    return [
        ("rot", (340, 330), (85, 95), (350, 430), 2.0 * osc(lf, 46, 0.3)),
        ("rot", (630, 180), (80, 90), (630, 268), 3.0 * osc(lf, 10)),
        ("rot", (528, 262), (55, 66), (470, 372), 7.0 * osc(lf, 14)),
        ("rot", (810, 330), (95, 100), (820, 432), 2.0 * osc(lf, 60, 1.2)),
        ("rot", (1030, 220), (80, 90), (1040, 300), 3.0 * osc(lf, 38, 2.0)),
        ("rot", (1215, 420), (75, 88), (1210, 505), 4.0 * osc(lf, 42, 0.8)),
        ("rot", (1040, 630), (95, 90), (1040, 735), 4.0 * osc(lf, 34, 1.7)),
        ("move", (1045, 688), (48, 30), (0, 4.0 * (0.5 + 0.5 * osc(lf, 6)))),
        ("blink", (293, 362), (13, 6), blink_amount(lf, [30])),
        ("blink", (349, 353), (13, 6), blink_amount(lf, [30])),
        ("blink", (998, 216), (12, 5), blink_amount(lf, [46])),
        ("blink", (1052, 211), (12, 5), blink_amount(lf, [46])),
        ("blink", (1210, 409), (12, 5), blink_amount(lf, [18, 62])),
        ("blink", (1261, 412), (12, 5), blink_amount(lf, [18, 62])),
        ("blink", (1011, 607), (11, 6), blink_amount(lf, [54])),
        ("blink", (1068, 607), (11, 6), blink_amount(lf, [54])),
    ]


# ---------------------------------------------------------------- 仕上げ（色調・ビネット・粒子）
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = 1 - 0.38 * (((_xx - W / 2) / (W * 0.62)) ** 2 + ((_yy - H / 2) / (H * 0.68)) ** 2)
VIGNETTE = np.clip(VIGNETTE, 0.45, 1)[..., None]
_rng = np.random.default_rng(3)
GRAIN = [(_rng.standard_normal((H // 2, W // 2, 1)).astype(np.float32) * 0.018) for _ in range(6)]


def finish(arr, f):
    # 夕焼け寄りの暖色グレード + 軽いリフト（フィルムっぽさ）
    arr = arr * np.array([1.04, 1.0, 0.92], np.float32) + np.array([0.025, 0.012, 0.0], np.float32)
    arr = arr * VIGNETTE
    g = GRAIN[f % len(GRAIN)].repeat(2, 0).repeat(2, 1)
    return np.clip(arr + g, 0, 1)


# ---------------------------------------------------------------- 写真（ポラロイド）
def polaroid(photo, caption, fnt, pw=600):
    ph = int(pw * photo.height / photo.width)
    photo = photo.resize((pw, ph), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 60, 2))
    b, bb = 22, 84
    card = Image.new("RGBA", (pw + b * 2, ph + b + bb), (252, 248, 240, 255))
    card.paste(photo, (b, b))
    d = ImageDraw.Draw(card)
    tw = fnt.getlength(caption)
    d.text(((card.width - tw) / 2, ph + b + 18), caption, font=fnt, fill=BROWN)
    # マスキングテープ
    tape = Image.new("RGBA", (150, 40), (232, 200, 120, 170))
    card.alpha_composite(tape, ((card.width - 150) // 2, 0))
    pad = 40
    canvas = Image.new("RGBA", (card.width + pad * 2, card.height + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(150, (pad + 8, pad + 14, pad + 8 + card.width, pad + 14 + card.height))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    canvas.paste(Image.new("RGBA", canvas.size, (40, 22, 10, 255)), (0, 0), shadow)
    canvas.alpha_composite(card, (pad, pad))
    return canvas


def paper_bg(seed=1):
    rng = np.random.default_rng(seed)
    base = np.ones((H, W, 3), np.float32) * (np.array(CREAM, np.float32) / 255)
    noise = rng.standard_normal((H // 4, W // 4)).astype(np.float32)
    noise = np.asarray(Image.fromarray(((noise * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
                       .resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    base *= (0.95 + 0.07 * noise)[..., None]
    # 右上から差し込む夕日
    glow = np.exp(-(((_xx - W * 0.85) / (W * 0.6)) ** 2 + ((_yy + H * 0.1) / (H * 0.8)) ** 2))[..., None]
    base = base * (1 - 0.25 * glow) + np.array([1.0, 0.72, 0.42], np.float32) * 0.25 * glow
    return base


# ---------------------------------------------------------------- メイン
def main():
    assets, fonts, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    F_min = lambda s: ImageFont.truetype(str(fonts / "ShipporiMincho-800.ttf"), int(s))
    F_hand = lambda s: ImageFont.truetype(str(fonts / "KleeOne-600.ttf"), int(s))

    poster = Image.open(assets / "keyart.png").convert("RGB")

    # --- 素材の切り出し
    keyart = poster.crop((0, 330, 805, 783))
    keyart = keyart.resize((W + 300, int((W + 300) * keyart.height / keyart.width)), Image.LANCZOS)
    keyart = keyart.filter(ImageFilter.UnsharpMask(2.5, 70, 2))
    thumbs = [(poster.crop((1005, 550, 1170, 652)), "川で、日が暮れるまで", -5.0),
              (poster.crop((207, 804, 383, 890)), "自転車で、どこまでも", 4.0),
              (poster.crop((1184, 550, 1348, 652)), "秋祭りの夜", 3.0),
              (poster.crop((584, 804, 761, 890)), "最後の花火", -3.5)]
    pol_font = F_hand(34)
    polaroids = [polaroid(p, c, pol_font, pw=480) for p, c, a in thumbs]
    pol_angles = [a for _, _, a in thumbs]
    pol_pos = [(600, 345), (1340, 355), (650, 745), (1300, 735)]  # 中心座標

    layers = assets / "layers"
    key_bg = Image.open(layers / "key_bg.png").convert("RGB").resize(keyart.size, Image.LANCZOS)
    key_bg = key_bg.filter(ImageFilter.UnsharpMask(2.5, 70, 2))
    key_fg = Image.open(layers / "key_fg.png").convert("RGBA").resize(keyart.size, Image.LANCZOS)
    key_fg = key_fg.filter(ImageFilter.UnsharpMask(2.5, 70, 2))
    KEY_S = keyart.width / 805.0
    GRP_S = (W + 240) / 1536.0

    # キャラクター（切り抜き）: 元パネルと同じ縮尺で並べる
    CH_S = 0.78
    CH_PAD = 40
    char_imgs = []
    for i in range(4):
        c = Image.open(layers / f"char{i}.png").convert("RGBA")
        c = c.resize((int(c.width * CH_S), int(c.height * CH_S)), Image.LANCZOS).filter(
            ImageFilter.UnsharpMask(2, 50, 2))
        # 首を回したときに頭が切れないよう上と左右に余白
        padded = Image.new("RGBA", (c.width + CH_PAD * 2, c.height + CH_PAD), (0, 0, 0, 0))
        padded.paste(c, (CH_PAD, CH_PAD))
        char_imgs.append(padded)
    char_x = [330, 755, 1170, 1590]
    GROUND = 955
    tag_specs = [("タイチ", (52, 110, 196)), ("やっさん", (232, 132, 30)), ("トシヒコ", (58, 140, 66)),
                 ("クルミちゃん", (214, 86, 120))]
    sh = Image.new("L", (360, 80), 0)
    ImageDraw.Draw(sh).ellipse([30, 15, 330, 65], fill=150)
    ground_shadow = Image.new("RGBA", sh.size, (30, 14, 4, 255))
    ground_shadow.putalpha(sh.filter(ImageFilter.GaussianBlur(10)))

    def g_big(im):
        im = im.crop((0, 40, 1536, 904)).resize((W + 240, int((W + 240) * 864 / 1536)), Image.LANCZOS)
        return im.filter(ImageFilter.UnsharpMask(2, 60, 2))
    group_bg = g_big(Image.open(layers / "group_bg.png").convert("RGB"))
    group_fg = g_big(Image.open(layers / "group_fg.png").convert("RGBA"))
    chars_bg = to_np(cover(Image.open(layers / "key_bg.png").convert("RGB"), W, H).filter(ImageFilter.GaussianBlur(16)))
    chars_bg = chars_bg * 0.85 + 0.06
    chars_bg *= (1 - 0.35 * np.clip((_yy - H * 0.6) / (H * 0.4), 0, 1))[..., None]

    paper = paper_bg()
    bottom_grad = np.clip((_yy - H * 0.45) / (H * 0.55), 0, 1)[..., None] ** 1.4

    # --- テキスト
    open_l1 = char_layers("どこか懐かしい、", F_hand(84), (255, 250, 240), shadow=(60, 25, 5), shadow_blur=12)
    open_l2 = char_layers("あの秋を、もう一度。", F_hand(84), (255, 250, 240), shadow=(60, 25, 5), shadow_blur=12)
    tags = []
    for name, col in tag_specs:
        fnt = F_hand(46)
        tw = int(fnt.getlength(name))
        tg = Image.new("RGBA", (tw + 56, 78), (0, 0, 0, 0))
        dd = ImageDraw.Draw(tg)
        dd.rounded_rectangle([4, 6, tw + 52, 74], radius=16, fill=col + (235,))
        dd.text((28, 10), name, font=fnt, fill=(255, 255, 255))
        tags.append(tg.rotate(-3, expand=True, resample=Image.BICUBIC))
    grp_l1 = char_layers("何もなかった。", F_min(96), (255, 252, 245), shadow=(20, 8, 0), shadow_blur=14,
                         shadow_alpha=0.9)
    grp_l2 = char_layers("だから、全部あった。", F_min(96), (255, 190, 110), shadow=(20, 8, 0), shadow_blur=14,
                         shadow_alpha=0.9)
    title = char_layers("ぼくと、あの秋。", F_min(180), (255, 251, 242), shadow=(0, 0, 0), shadow_blur=20,
                        shadow_alpha=0.6, glow=((255, 190, 110), 24))
    F_sans = lambda s: ImageFont.truetype(str(fonts / "InstrumentSans-Bold.ttf"), int(s))
    F_sansr = lambda s: ImageFont.truetype(str(fonts / "InstrumentSans-Regular.ttf"), int(s))
    F_goth = lambda s: ImageFont.truetype(str(fonts / "ipagp.ttf"), int(s))
    card_txt = {"card1": text_layer("何もない町。", F_min(92), (250, 250, 252), shadow=None,
                                    glow=((120, 160, 255), 18)),
                "card2": text_layer("あの頃の、なかまたち。", F_min(92), (250, 250, 252), shadow=None,
                                    glow=((120, 160, 255), 18))}
    presents = text_layer("IKEMOTO  STUDIO  PRESENTS", F_sansr(34), (225, 235, 255), shadow=None,
                          glow=((90, 140, 255), 10), tracking=9)
    disclaimer = text_layer("※画面は開発中のものです。", F_goth(24), (215, 215, 215), shadow=None, pad=10)
    title_full = text_layer("ぼくと、あの秋。", F_min(180), (255, 252, 246), shadow=(0, 0, 0), shadow_blur=20,
                            shadow_alpha=0.6, glow=((255, 190, 110), 24))
    sub2 = text_layer("―  山里で過ごした30日間  ―", F_hand(44), (235, 238, 250), shadow=None)
    ps5 = text_layer("PS5", F_sans(92), (255, 255, 255), shadow=None, tracking=4, pad=20)
    ps5_small = text_layer("PlayStation®5", F_sansr(26), (210, 220, 240), shadow=None, tracking=2, pad=10)
    release = text_layer("発売日  近日発表", F_goth(58), (255, 255, 255), shadow=None, pad=20)
    release_s = text_layer("COMING SOON", F_sansr(26), (210, 220, 240), shadow=None, tracking=6, pad=10)
    copyright_t = text_layer("©IKEMOTO STUDIO", F_sansr(24), (170, 180, 200), shadow=None, tracking=3, pad=10)
    # CERO マーク（審査予定）
    cero = Image.new("RGBA", (150, 170), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cero)
    cd.rounded_rectangle([3, 3, 146, 166], radius=14, outline=(255, 255, 255, 255), width=4)
    cd.rectangle([3, 60, 146, 64], fill=(255, 255, 255, 255))
    for txt, fnt, y in (("CERO", F_sans(40), 10), ("審査予定", F_goth(30), 92)):
        tw = fnt.getlength(txt)
        cd.text(((150 - tw) / 2, y), txt, font=fnt, fill=(255, 255, 255, 255))

    # 粒子（PS5 の UI のような、ゆっくり上る光の粒）
    prng = np.random.default_rng(21)
    particles = [(prng.uniform(0, W), prng.uniform(0, H), prng.uniform(0.4, 2.2), prng.uniform(1.5, 5.5),
                  prng.uniform(0.25, 1.0), prng.uniform(0, 6.28)) for _ in range(110)]
    glow_navy = np.exp(-(((_xx - W / 2) / (W * 0.55)) ** 2 + ((_yy - H * 0.62) / (H * 0.45)) ** 2))[..., None]
    navy_bg = (np.array([0.008, 0.014, 0.04], np.float32)
               + glow_navy * np.array([0.04, 0.10, 0.30], np.float32))

    def draw_particles(arr, f, alpha=1.0):
        lay = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(lay)
        for x, y, vy, r, a, ph in particles:
            yy = (y - vy * f * 2.2) % H
            xx = x + 14 * math.sin(f * 0.03 + ph)
            tw = 0.6 + 0.4 * math.sin(f * 0.15 + ph * 3)
            d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=int(255 * a * tw))
        m = np.asarray(lay.filter(ImageFilter.GaussianBlur(1.6)), np.float32)[..., None] / 255.0
        return arr + m * np.array([0.75, 0.85, 1.0], np.float32) * alpha

    leaves = Leaves()
    flies_open = Dragonflies([(1500, 160, -6.5, 0.6, 0.9, 0.0), (260, 420, 5.0, -1.0, 0.65, 2.1),
                              (1050, 90, 3.2, 0.8, 0.5, 4.0)])
    flies_title = Dragonflies([(-60, 200, 8.0, 0.5, 0.8, 1.0), (1980, 760, -7.0, -1.2, 0.6, 3.0)])

    # --- 各カットの描画
    def cut_intro(f):
        arr = navy_bg.copy()
        # 中央を走る光の筋
        t = ease_out_cubic(prog(f, 2, 22))
        streak = np.exp(-((_yy - H / 2) / (2.0 + 26 * t)) ** 2) * np.exp(-((_xx - W / 2) / (W * 0.5 * t + 1)) ** 2)
        core = np.exp(-((_yy - H / 2) / 1.6) ** 2) * np.exp(-((_xx - W / 2) / (W * 0.42 * t + 1)) ** 2)
        fade = 1 - 0.6 * prog(f, 26, 10)
        arr = arr + (streak[..., None] * np.array([0.15, 0.32, 0.9], np.float32) * 0.8
                     + core[..., None] * np.array([0.9, 0.95, 1.0], np.float32)) * fade
        arr = draw_particles(arr, f, 0.9)
        a = ease_out_cubic(prog(f, 8, 12)) * (1 - prog(f, 28, 8))
        paste_alpha(arr, presents, (W - presents.width) / 2, H / 2 + 40, a)
        return arr

    def cut_card(key):
        def render_card(f):
            lf = f - CUTS[key][0]
            arr = navy_bg.copy() * 0.55
            arr = draw_particles(arr, f, 0.4)
            lay = card_txt[key]
            a = ease_out_cubic(prog(lf, 0, 6))
            sc = 1.06 - 0.06 * ease_out_cubic(prog(lf, 0, 18))
            l2 = lay.resize((int(lay.width * sc), int(lay.height * sc)), Image.BICUBIC)
            paste_alpha(arr, l2, (W - l2.width) / 2, (H - l2.height) / 2, a)
            return arr
        return render_card

    def cut_open(f):
        lf = f - CUTS["open"][0]
        # 背景はゆっくり、人物はやや速く寄る＝奥行き（2.5D）
        arr = to_np(kenburns(key_bg, lf, -8, 72, 1.0, 1.08, 0.45, 0.5, 0.52, 0.48))
        sway_grass(arr, f, (0, int(H * 0.55), int(W * 0.42), H), amp=9)
        fg = puppet(key_fg, bones_key(lf + 10), scale=KEY_S)
        fg = kenburns(fg, lf, -8, 72, 1.0, 1.17, 0.47, 0.52, 0.62, 0.5)
        fg = breathe(fg, 0.010 * math.sin(f * 2 * math.pi / 55), H * 0.95)
        paste_alpha(arr, fg, 0, 0)
        # 左上から差し込む光
        t = lf / 72.0
        glow = np.exp(-(((_xx - W * (0.2 + 0.05 * t)) / (W * 0.45)) ** 2 + ((_yy - H * 0.15) / (H * 0.6)) ** 2))
        arr = arr + glow[..., None] * np.array([0.35, 0.18, 0.04], np.float32) * (0.6 + 0.4 * math.sin(f * 0.07))
        top = np.clip(1 - _yy / (H * 0.55), 0, 1)[..., None]
        arr = arr * (1 - 0.35 * top)
        draw_chars(arr, *open_l1[:2], 130, 160, lf, 4, step=2, dur=12)
        draw_chars(arr, *open_l2[:2], 210, 285, lf, 22, step=2, dur=12)
        flies_open.draw(arr, f)
        leaves.draw(arr, f)
        return arr

    def cut_photos(f):
        lf = f - CUTS["photos"][0]
        arr = paper.copy()
        drift = lf * 0.35
        for i, (pol, (cx, cy)) in enumerate(zip(polaroids, pol_pos)):
            t = prog(f, DROPS[i], DROP_LEN)
            if t <= 0:
                continue
            s = 1.0 + 0.35 * (1 - ease_out_cubic(t))
            d = f - DROPS[i] - DROP_LEN
            wob = 7 * math.exp(-max(d, 0) / 5) * math.cos(max(d, 0) * 0.9) if d >= 0 else 10 * (1 - t)
            p = pol if s == 1.0 else pol.resize((int(pol.width * s), int(pol.height * s)), Image.BICUBIC)
            p = p.rotate(pol_angles[i] + wob, expand=True, resample=Image.BICUBIC)
            paste_alpha(arr, p, cx - p.width / 2 - drift, cy - p.height / 2 - drift * 0.3, ease_out_cubic(t))
        leaves.draw(arr, f, 0.7)
        return arr

    def cut_chars(f):
        lf = f - CUTS["chars"][0]
        arr = chars_bg.copy()
        for i, c in enumerate(char_imgs):
            st = 2 + i * 7
            t = prog(lf, st, 14)
            if t <= 0:
                continue
            land = lf - (st + 14)
            # 下から飛び出して、少し上まで跳ねてから着地
            y_off = (1 - ease_out_back(t, 2.6)) * 900
            if land < 0:
                sy = 1 + 0.07 * (1 - t)
            else:
                sy = 1 - 0.08 * math.exp(-land / 3.5) * math.cos(land * 0.95)
            sx = 1 / math.sqrt(sy)
            idle = clamp01(land / 8) if land >= 0 else 0
            bob = 7 * idle * abs(math.sin(lf * 2 * math.pi / 28 + i * 1.3))
            rot = 1.8 * idle * math.sin(lf * 2 * math.pi / 42 + i * 2.0)
            c = puppet(c, bones_char(i, lf, st + 14), scale=CH_S, off=(-CH_PAD / CH_S, -CH_PAD / CH_S))
            cw, chh = int(c.width * sx), int(c.height * sy)
            im = c.resize((cw, chh), Image.BICUBIC).rotate(rot, expand=True, resample=Image.BICUBIC,
                                                          center=(cw / 2, chh))
            feet_y = GROUND + y_off - bob
            # 足元の影（高く跳ぶほど薄く小さく）
            hgt = clamp01((GROUND - feet_y) / 300)
            shd = ground_shadow.resize((int(ground_shadow.width * (1 - 0.4 * hgt)), ground_shadow.height))
            paste_alpha(arr, shd, char_x[i] - shd.width / 2, GROUND - 40, (1 - 0.7 * hgt) * clamp01(t * 3))
            paste_alpha(arr, im, char_x[i] - im.width / 2, feet_y - im.height)
            # 名札：着地したらポンと出る
            if land >= 0:
                ts = ease_out_back(prog(land, 2, 9), 2.0)
                if ts > 0:
                    tg = tags[i].resize((max(1, int(tags[i].width * ts)), max(1, int(tags[i].height * ts))))
                    top = GROUND - c.height + CH_PAD - bob
                    paste_alpha(arr, tg, char_x[i] - tg.width / 2, top - 70 - tg.height / 2 + 30, clamp01(ts))
        leaves.draw(arr, f, 0.6)
        return arr

    def cut_group(f):
        lf = f - CUTS["group"][0]
        f0, f1 = CUTS["group"][0] - 6, CUTS["group"][1] + 6
        arr = to_np(kenburns(group_bg, f, f0, f1, 1.07, 1.0, 0.5, 0.45, 0.5, 0.5))
        fg = puppet(group_fg, bones_group(lf), scale=GRP_S, off=(0, 40))
        fg = kenburns(fg, f, f0, f1, 1.16, 1.0, 0.5, 0.42, 0.5, 0.5)
        fg = breathe(fg, 0.008 * math.sin(lf * 2 * math.pi / 40), H)
        paste_alpha(arr, fg, 0, 0)
        arr = arr * (1 - 0.62 * bottom_grad)
        draw_chars(arr, *grp_l1[:2], (W - grp_l1[2]) / 2, 690, lf, 4, step=2, dur=10)
        draw_chars(arr, *grp_l2[:2], (W - grp_l2[2]) / 2, 810, lf, 26, step=2, dur=10)
        leaves.draw(arr, f, 0.85)
        return arr

    def cut_title(f):
        lf = f - CUTS["title"][0]
        arr = navy_bg.copy()
        # 下から夕焼け色がにじむ
        warm = np.exp(-(((_xx - W / 2) / (W * 0.7)) ** 2 + ((_yy - H * 1.05) / (H * 0.45)) ** 2))[..., None]
        arr = arr + warm * np.array([0.35, 0.14, 0.03], np.float32) * ease_out_cubic(prog(lf, 0, 40))
        arr = draw_particles(arr, f, 0.7)
        leaves.draw(arr, f, 0.6)
        # タイトル：1文字ずつ浮かび上がる → 光が走る
        layers, xs, tw = title
        pad = 40
        ty = 230
        for i, (lay, cx) in enumerate(zip(layers, xs)):
            t = prog(lf, 2 + i * 2, 14)
            if t <= 0:
                continue
            e = ease_out_cubic(t)
            l = lay if e >= 1 else lay.filter(ImageFilter.GaussianBlur((1 - e) * 10))
            paste_alpha(arr, l, (W - tw) / 2 + cx - pad, ty - pad + (1 - e) * 20, e)
        sw = prog(lf, 26, 16)
        if 0 < sw < 1:
            tx0 = int((W - title_full.width) / 2)
            ty0 = ty - 40
            al = np.asarray(title_full, np.float32)[..., 3] / 255.0
            hh, ww = al.shape
            xs_ = np.arange(ww)[None, :] + np.arange(hh)[:, None] * 0.45
            band = np.exp(-((xs_ - (-300 + sw * (ww + 600))) / 60.0) ** 2) * al
            sub_ = arr[ty0:ty0 + hh, tx0:tx0 + ww]
            arr[ty0:ty0 + hh, tx0:tx0 + ww] = sub_ + band[..., None] * np.array([1.0, 0.95, 0.85], np.float32) * 0.9
        a = ease_out_cubic(prog(lf, 20, 12))
        paste_alpha(arr, sub2, (W - sub2.width) / 2, 470 + (1 - a) * 10, a)
        # 下段：PS5 / 発売日 / CERO
        a = ease_out_cubic(prog(lf, 34, 12))
        if a > 0:
            y = 700
            ln = int(700 * a)
            arr[y:y + 2, W // 2 - ln:W // 2 + ln] = arr[y:y + 2, W // 2 - ln:W // 2 + ln] * 0.3 + 0.7 * np.array(
                [0.85, 0.9, 1.0], np.float32)
            paste_alpha(arr, ps5, 420 - ps5.width / 2, 735, a)
            paste_alpha(arr, ps5_small, 420 - ps5_small.width / 2, 860, a)
            paste_alpha(arr, release, W / 2 - release.width / 2, 755, a)
            paste_alpha(arr, release_s, W / 2 - release_s.width / 2, 850, a)
            paste_alpha(arr, cero, 1500 - cero.width / 2, 730, a)
            paste_alpha(arr, copyright_t, W / 2 - copyright_t.width / 2, 1005, a * 0.9)
        flies_title.draw(arr, f, CUTS["title"][0])
        return arr

    render = {"intro": cut_intro, "open": cut_open, "card1": cut_card("card1"), "photos": cut_photos,
              "card2": cut_card("card2"), "chars": cut_chars, "group": cut_group, "title": cut_title}
    order = list(CUTS)

    audio = out_path.parent / "_bgm15.wav"
    make_audio(audio)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(audio),
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N):
        idx = next(i for i, k in enumerate(order) if CUTS[k][0] <= f < CUTS[k][1])
        arr = render[order[idx]](f)
        # 次のカットへのクロスフェード
        if idx + 1 < len(order):
            nk = order[idx + 1]
            nxt, xf = CUTS[nk][0], XFS[nk]
            if xf and f >= nxt - xf:
                w = ease_in_out((f - (nxt - xf)) / xf)
                arr = arr * (1 - w) + render[nk](f) * w
        cur = order[idx]
        if cur in FOOTAGE:
            arr = finish(arr, f)
        else:
            arr = np.clip(arr + GRAIN[f % len(GRAIN)].repeat(2, 0).repeat(2, 1) * 0.6, 0, 1)
        # 本編はシネマスコープ風の黒帯 + 開発中表記
        if cur in FOOTAGE:
            arr[:BAR] = 0
            arr[H - BAR:] = 0
            paste_alpha(arr, disclaimer, W - disclaimer.width - 40, H - BAR + (BAR - disclaimer.height) / 2, 0.9)
        # 冒頭は黒からフェードイン、最後は軽くフェードアウト
        arr = arr * ease_in_out(prog(f, 0, 6)) * (1 - 0.9 * ease_in_out(prog(f, N - 10, 10)))
        proc.stdin.write((np.clip(arr, 0, 1) * 255).astype(np.uint8).tobytes())
        if f % 30 == 0:
            print(f"frame {f}/{N}", flush=True)
    proc.stdin.close()
    proc.wait()
    audio.unlink()
    print("wrote", out_path)


# ---------------------------------------------------------------- BGM（秋っぽいアコースティック）
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nn(s):
    return 12 * (int(s[-1]) + 1) + NOTE[s[0]] + (1 if "#" in s else 0)


def pluck(freq, dur, sr=SR, bright=0.5, seed=0):
    """Karplus-Strong（ナイロン弦風）. ブロック単位でベクトル化."""
    n = int(dur * sr)
    p = max(2, int(round(sr / freq)))
    rng = np.random.default_rng(seed)
    exc = rng.uniform(-1, 1, p)
    # 柔らかい指弾き：励振をローパス
    for _ in range(int(3 + (1 - bright) * 6)):
        exc = 0.5 * (exc + np.roll(exc, 1))
    y = np.zeros(n + p + 1)
    y[:p] = exc
    decay = 0.996
    i = p
    while i < n + p:
        j = min(i + p, n + p)
        seg = np.arange(i, j)
        y[seg] = decay * 0.5 * (y[seg - p] + y[seg - p - 1])
        i = j
    out = y[:n]
    fade = np.minimum(1, np.arange(n)[::-1] / (0.05 * sr))
    return out * fade


def piano(freq, dur, sr=SR, gain=1.0):
    n = int((dur + 1.4) * sr)
    t = np.arange(n) / sr
    env = np.exp(-t * 2.2) * np.minimum(1, t / 0.004)
    rel = np.clip((dur + 1.4 - t) / 1.4, 0, 1)
    sig = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t * 3)
           + 0.12 * np.sin(2 * np.pi * 3 * freq * t) * np.exp(-t * 5)
           + 0.05 * np.sin(2 * np.pi * 4.01 * freq * t) * np.exp(-t * 8))
    return sig * env * rel * gain


def pad_tone(freqs, dur, sr=SR):
    n = int(dur * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for fq in freqs:
        for det in (-0.12, 0.0, 0.13):
            ph = 2 * np.pi * fq * (1 + det / 100) * t
            sig += np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    env = np.minimum(1, t / 0.8) * np.minimum(1, (dur - t) / 0.9)
    return sig * np.clip(env, 0, 1) / (len(freqs) * 3)


def onepole_lp(x, cutoff, sr=SR):
    a = math.exp(-2 * math.pi * cutoff / sr)
    # IIR を FFT 畳み込みで近似（減衰インパルス）
    k = int(sr * 5 / cutoff) + 1
    h = (1 - a) * a ** np.arange(k)
    m = len(x) + k - 1
    L = 1 << (m - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, L) * np.fft.rfft(h, L), L)[:len(x)]


def make_audio(path):
    total = int(DURATION * SR)
    L = np.zeros(total + SR * 3)
    R = np.zeros_like(L)

    def add(t0, sig, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        i1 = min(i0 + len(sig), len(L))
        s = sig[:i1 - i0] * gain
        L[i0:i1] += s * math.sqrt(0.5 * (1 - pan))
        R[i0:i1] += s * math.sqrt(0.5 * (1 + pan))

    e8 = 0.375  # 80BPM の8分音符
    # (開始, 終了, ベース, アルペジオ構成音)
    T_END = CUTS["title"][0] / FPS  # 11.8s
    chords = [
        (1.2, 3.6, "A2", ["A3", "E4", "B4", "C5", "E5"]),       # Am9
        (3.6, 6.2, "F2", ["F3", "C4", "E4", "A4", "C5"]),       # Fmaj7
        (6.2, 7.4, "G2", ["G3", "D4", "A4", "B4", "D5"]),       # Gadd9
        (7.4, 9.6, "E2", ["E3", "B3", "D4", "G4", "B4"]),       # Em7
        (9.6, 10.8, "D2", ["D3", "A3", "C4", "F4", "A4"]),      # Dm7
        (10.8, T_END, "G2", ["G3", "D4", "F4", "B4", "D5"]),    # G7
        (T_END, 15.0, "C2", ["C3", "G3", "B3", "D4", "E4", "G4"]),  # Cmaj9
    ]
    pattern = [0, 2, 1, 3, 2, 4, 3, 1]
    seed = 0
    for c0, c1, bass, tones in chords:
        last = c0 >= T_END
        add(c0, pluck(midi(nn(bass)), 3.5 if last else c1 - c0 + 0.6, seed=seed), 0.55, -0.1)
        if last:
            # 最後はゆっくりストローク
            for k, tn in enumerate(tones):
                seed += 1
                add(c0 + 0.045 * k, pluck(midi(nn(tn)), 3.4, bright=0.55, seed=seed), 0.32, -0.3 + 0.12 * k)
            continue
        t, k = c0, 0
        while t < c1 - 1e-6:
            seed += 1
            tn = tones[pattern[k % len(pattern)] % len(tones)]
            add(t, pluck(midi(nn(tn)), 1.6, bright=0.5, seed=seed), 0.30, -0.25 + 0.1 * (k % 5))
            t += e8
            k += 1

    # パッド（オルガンっぽくならないようローパス）
    pad_sig = np.zeros_like(L)
    for c0, c1, bass, tones in chords:
        seg = pad_tone([midi(nn(x)) for x in tones[:4]], c1 - c0 + 0.8)
        i0 = int(c0 * SR)
        pad_sig[i0:i0 + len(seg)] += seg[:len(pad_sig) - i0]
    pad_sig = onepole_lp(pad_sig, 1400)
    L += pad_sig * 0.10
    R += pad_sig * 0.10

    # ピアノのメロディ（郷愁のある下降ライン）
    melody = [(1.95, "E5", 0.75), (2.7, "D5", 0.375), (3.075, "C5", 0.375), (3.3, "D5", 0.3),
              (3.6, "C5", 0.75), (4.35, "A4", 0.75), (5.1, "C5", 0.375), (5.475, "E5", 0.7),
              (6.2, "D5", 0.6), (6.8, "B4", 0.6),
              (7.4, "G5", 0.75), (8.15, "E5", 0.85), (9.0, "D5", 0.5),
              (9.6, "F5", 0.6), (10.2, "E5", 0.3), (10.5, "D5", 0.3),
              (10.8, "D5", 0.5), (11.3, "B4", 0.5),
              (T_END, "C5", 2.5), (T_END, "E5", 2.5), (T_END + 0.7, "G5", 2.0), (T_END + 1.4, "C6", 1.5)]
    for t0, n_, d in melody:
        g = 0.22 if t0 < T_END else 0.16
        add(t0, piano(midi(nn(n_)), d), g, 0.15)
        add(t0 + 0.23, piano(midi(nn(n_)), d), g * 0.18, -0.4)  # ディレイ

    rng = np.random.default_rng(11)
    # 鈴虫（秋の夜）: 4.3kHz 前後のトリル
    def cricket(t0, dur, freq, gain, pan):
        n = int(dur * SR)
        t = np.arange(n) / SR
        trill = (np.sin(2 * np.pi * 38 * t) > 0.2).astype(float)
        burst = (np.sin(2 * np.pi * 1.6 * t + rng.uniform(0, 6)) > -0.1).astype(float)
        env = onepole_lp(trill * burst, 300)
        add(t0, np.sin(2 * np.pi * freq * t) * env * np.minimum(1, t / 0.3) * np.minimum(1, (dur - t) / 0.4),
            gain, pan)
    cricket(1.2, 3.0, 4300, 0.035, -0.6)
    cricket(1.5, 2.4, 4620, 0.025, 0.6)
    cricket(T_END, 3.2, 4400, 0.022, 0.5)

    # --- トレーラー風の効果音
    def boom(t0, gain=1.0, dur=1.6):
        n = int(dur * SR)
        t = np.arange(n) / SR
        fr = 34 + 70 * np.exp(-t * 9)
        body = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 2.4)
        click = onepole_lp(rng.standard_normal(n), 2500) * np.exp(-t * 40) * 0.6
        add(t0, (body + click) * gain)

    def whoosh(t0, dur, gain=0.3, pan=0.0):
        n = int(dur * SR)
        t = np.arange(n) / SR
        env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
        x = rng.standard_normal(n)
        x = onepole_lp(x, 2400) - onepole_lp(x, 300)
        add(t0, x * env / (np.abs(x).max() + 1e-9) * gain, pan)

    def riser(t0, t1, gain=0.12):
        n = int((t1 - t0) * SR)
        t = np.arange(n) / SR
        k = t / (t1 - t0)
        fr = 180 * 2 ** (k * 2.2)
        tone = np.sin(2 * np.pi * np.cumsum(fr) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(fr * 1.5) / SR)
        x = rng.standard_normal(n)
        noise = onepole_lp(x, 5000) - onepole_lp(x, 800)
        add(t0, (tone * 0.4 + noise / (np.abs(noise).max() + 1e-9)) * k ** 2.5 * gain)

    def drone(t0, dur, gain=0.12):
        n = int(dur * SR)
        t = np.arange(n) / SR
        env = np.minimum(1, t / 0.3) * np.clip((dur - t) / 0.4, 0, 1)
        sig = sum(np.sin(2 * np.pi * f_ * t + i) for i, f_ in enumerate((65.4, 98.0, 130.8, 196.2)))
        shimmer = sum(np.sin(2 * np.pi * f_ * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t + i))
                      for i, f_ in enumerate((1046.5, 1568.0, 2093.0)))
        add(t0, (sig / 4 + shimmer * 0.08) * env * gain)

    drone(0.0, 1.4, 0.16)                 # 導入：低いドローンときらめき
    riser(0.2, 1.2, 0.08)
    whoosh(0.85, 0.5, 0.25)
    boom(1.2, 0.35)                       # 本編イン
    boom(CUTS["card1"][0] / FPS, 0.6)     # 黒カード
    whoosh(CUTS["photos"][0] / FPS - 0.25, 0.4, 0.22, 0.3)
    boom(CUTS["card2"][0] / FPS, 0.6)
    whoosh(CUTS["chars"][0] / FPS - 0.25, 0.4, 0.22, -0.3)
    for i in range(4):                    # キャラのジャンプ
        whoosh((CUTS["chars"][0] + 2 + i * 7) / FPS, 0.35, 0.10, -0.6 + 0.4 * i)
    whoosh(CUTS["group"][0] / FPS - 0.2, 0.4, 0.18)
    riser(10.6, T_END, 0.12)
    boom(T_END, 1.0, 2.4)                 # タイトルのヒット

    # 風（秋風・枯れ葉のざわめき）
    n = total
    wind = onepole_lp(rng.standard_normal(n), 900) - onepole_lp(rng.standard_normal(n), 120)
    t = np.arange(n) / SR
    wenv = 0.5 + 0.5 * np.sin(2 * np.pi * 0.21 * t) * np.sin(2 * np.pi * 0.07 * t + 1)
    wind = wind / (np.abs(wind).max() + 1e-9) * wenv * 0.05
    L[:n] += wind
    R[:n] += np.roll(wind, 2000)

    # 写真が机に置かれる「パサッ」
    for i, fr in enumerate(DROPS):
        t0 = (fr + DROP_LEN) / FPS - 0.03
        k = int(0.12 * SR)
        noise = rng.standard_normal(k) * np.exp(-np.arange(k) / (0.025 * SR))
        add(t0, noise - onepole_lp(noise, 600), 0.10, -0.3 if i % 2 == 0 else 0.3)

    # タイトルのウインドチャイム
    for k, nm in enumerate(["E6", "G6", "B6", "D7", "G7"]):
        add(T_END + 0.15 + 0.07 * k, piano(midi(nn(nm)), 0.2), 0.05, -0.5 + 0.25 * k)

    # 簡易リバーブ（くし形ディレイの和）
    def reverb(x):
        out = np.zeros_like(x)
        for dly, g in ((0.031, 0.32), (0.047, 0.28), (0.071, 0.24), (0.113, 0.2), (0.167, 0.15), (0.241, 0.1)):
            d = int(dly * SR)
            out[d:] += x[:-d] * g
        return onepole_lp(out, 4000)
    L, R = L + reverb(R) * 0.55, R + reverb(L) * 0.55

    L, R = L[:total], R[:total]
    t = np.arange(total) / SR
    master = np.minimum(1, t / 0.05) * np.clip((DURATION - t) / 1.2, 0, 1) ** 1.2
    st = np.stack([L * master, R * master], 1)
    st = np.tanh(st / (np.abs(st).max() + 1e-9) * 1.1) * 0.85
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
