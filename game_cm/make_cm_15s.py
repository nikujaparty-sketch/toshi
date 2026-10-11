"""『ぼくと、あの秋。』15秒CM ジェネレーター.

横型 1920x1080 / 30fps / 秋っぽいBGM（ナイロンギター・ピアノ・パッド・鈴虫）付き。

カット構成:
    0.0- 3.0s  キーアート（夕焼けの縁側）  「どこか懐かしい、あの秋を、もう一度。」
               人物と背景を分けた2.5Dカメラ / 呼吸 / ススキの揺れ / 赤とんぼ
    3.0- 6.0s  思い出の写真が机に落ちてくる（川・自転車・秋祭り・花火）  着地でゆらっと揺れる
    6.0- 9.5s  なかまたち（キャラクター4人）  切り抜いた4人が順にジャンプして登場 → 揺れながら待機
    9.5-12.0s  集合写真  「何もなかった。だから、全部あった。」  人物と背景を分けた2.5Dカメラ
   12.0-15.0s  タイトル『ぼくと、あの秋。』

人物の切り抜きと背景は make_layers.py で事前に作る（assets/layers/）。

使い方:
    python3 make_cm_15s.py <assets_dir> <fonts_dir> <out.mp4>
      assets_dir: keyart.png（ポスター） / group.png / layers/（make_layers.py の出力）
      fonts_dir : ShipporiMincho-800.ttf / KleeOne-600.ttf / InstrumentSerif-Regular.ttf
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

# カット（開始, 終了）フレーム。境界は XF フレームでクロスフェード
CUTS = {"open": (0, 90), "photos": (90, 180), "chars": (180, 285), "group": (285, 360), "title": (360, N)}
XF = 8
DROPS = [96, 116, 136, 156]  # 写真が落ちるフレーム（効果音と同期）

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
    F_lat = lambda s: ImageFont.truetype(str(fonts / "InstrumentSerif-Regular.ttf"), int(s))

    poster = Image.open(assets / "keyart.png").convert("RGB")
    group = Image.open(assets / "group.png").convert("RGB")

    # --- 素材の切り出し
    keyart = poster.crop((0, 330, 805, 783))
    keyart = keyart.resize((W + 300, int((W + 300) * keyart.height / keyart.width)), Image.LANCZOS)
    keyart = keyart.filter(ImageFilter.UnsharpMask(2.5, 70, 2))
    thumbs = [(poster.crop((1005, 550, 1170, 652)), "川で、日が暮れるまで", -5.0),
              (poster.crop((207, 804, 383, 890)), "自転車で、どこまでも", 4.0),
              (poster.crop((1184, 550, 1348, 652)), "秋祭りの夜", 3.0),
              (poster.crop((584, 804, 761, 890)), "最後の花火", -3.5)]
    pol_font = F_hand(34)
    polaroids = [polaroid(p, c, pol_font, pw=560) for p, c, a in thumbs]
    pol_angles = [a for _, _, a in thumbs]
    pol_pos = [(560, 285), (1370, 300), (620, 800), (1320, 790)]  # 中心座標

    layers = assets / "layers"
    key_bg = Image.open(layers / "key_bg.png").convert("RGB").resize(keyart.size, Image.LANCZOS)
    key_bg = key_bg.filter(ImageFilter.UnsharpMask(2.5, 70, 2))
    key_fg = Image.open(layers / "key_fg.png").convert("RGBA").resize(keyart.size, Image.LANCZOS)
    key_fg = key_fg.filter(ImageFilter.UnsharpMask(2.5, 70, 2))

    # キャラクター（切り抜き）: 元パネルと同じ縮尺で並べる
    CH_S = 0.84
    char_imgs = []
    for i in range(4):
        c = Image.open(layers / f"char{i}.png").convert("RGBA")
        char_imgs.append(c.resize((int(c.width * CH_S), int(c.height * CH_S)), Image.LANCZOS)
                         .filter(ImageFilter.UnsharpMask(2, 50, 2)))
    char_x = [330, 755, 1170, 1590]
    GROUND = 1040
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
    title_bg = to_np(cover(poster.crop((0, 300, 805, 790)), W, H).filter(ImageFilter.GaussianBlur(26)))
    title_bg = title_bg * 0.78 + np.array([0.20, 0.08, 0.0], np.float32) * 0.22
    bottom_grad = np.clip((_yy - H * 0.45) / (H * 0.55), 0, 1)[..., None] ** 1.4

    # --- テキスト
    open_l1 = char_layers("どこか懐かしい、", F_hand(84), (255, 250, 240), shadow=(60, 25, 5), shadow_blur=12)
    open_l2 = char_layers("あの秋を、もう一度。", F_hand(84), (255, 250, 240), shadow=(60, 25, 5), shadow_blur=12)
    chars_head = text_layer("あの頃の、なかまたち。", F_hand(58), (255, 250, 240), shadow=(60, 25, 5), shadow_blur=10)
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
    title = char_layers("ぼくと、あの秋。", F_min(190), (255, 251, 242), shadow=(70, 25, 0), shadow_blur=22,
                        shadow_alpha=0.85, glow=((255, 200, 120), 26))
    sub = text_layer("―  山里で過ごした30日間  ―", F_hand(50), (255, 244, 225), shadow=(70, 25, 0),
                     shadow_blur=10)
    genre = text_layer("ノスタルジック田舎生活アドベンチャー", F_hand(36), (255, 236, 205), shadow=(70, 25, 0),
                       shadow_blur=8)
    soon = text_layer("COMING  SOON", F_lat(64), (255, 248, 235), shadow=(70, 25, 0), shadow_blur=10,
                      tracking=6)

    leaves = Leaves()
    flies_open = Dragonflies([(1500, 160, -6.5, 0.6, 0.9, 0.0), (260, 420, 5.0, -1.0, 0.65, 2.1),
                              (1050, 90, 3.2, 0.8, 0.5, 4.0)])
    flies_title = Dragonflies([(-60, 200, 8.0, 0.5, 0.8, 1.0), (1980, 760, -7.0, -1.2, 0.6, 3.0)])

    # --- 各カットの描画
    def cut_open(f):
        # 背景はゆっくり、人物はやや速く寄る＝奥行き（2.5D）
        arr = to_np(kenburns(key_bg, f, 0, 100, 1.0, 1.08, 0.45, 0.5, 0.52, 0.48))
        sway_grass(arr, f, (0, int(H * 0.55), int(W * 0.42), H), amp=9)
        fg = kenburns(key_fg, f, 0, 100, 1.0, 1.17, 0.47, 0.52, 0.62, 0.5)
        fg = breathe(fg, 0.010 * math.sin(f * 2 * math.pi / 55), H * 0.95)
        paste_alpha(arr, fg, 0, 0)
        # 左上から差し込む光
        t = f / 90.0
        glow = np.exp(-(((_xx - W * (0.2 + 0.05 * t)) / (W * 0.45)) ** 2 + ((_yy - H * 0.15) / (H * 0.6)) ** 2))
        arr = arr + glow[..., None] * np.array([0.35, 0.18, 0.04], np.float32) * (0.6 + 0.4 * math.sin(f * 0.07))
        top = np.clip(1 - _yy / (H * 0.55), 0, 1)[..., None]
        arr = arr * (1 - 0.35 * top)
        draw_chars(arr, *open_l1[:2], 130, 120, f, 12, step=3, dur=14)
        draw_chars(arr, *open_l2[:2], 210, 245, f, 40, step=3, dur=14)
        flies_open.draw(arr, f)
        leaves.draw(arr, f)
        return arr

    def cut_photos(f):
        lf = f - CUTS["photos"][0]
        arr = paper.copy()
        drift = lf * 0.35
        for i, (pol, (cx, cy)) in enumerate(zip(polaroids, pol_pos)):
            t = prog(f, DROPS[i], 10)
            if t <= 0:
                continue
            s = 1.0 + 0.35 * (1 - ease_out_cubic(t))
            d = f - DROPS[i] - 10
            wob = 7 * math.exp(-max(d, 0) / 5) * math.cos(max(d, 0) * 0.9) if d >= 0 else 10 * (1 - t)
            p = pol if s == 1.0 else pol.resize((int(pol.width * s), int(pol.height * s)), Image.BICUBIC)
            p = p.rotate(pol_angles[i] + wob, expand=True, resample=Image.BICUBIC)
            paste_alpha(arr, p, cx - p.width / 2 - drift, cy - p.height / 2 - drift * 0.3, ease_out_cubic(t))
        leaves.draw(arr, f, 0.7)
        return arr

    def cut_chars(f):
        lf = f - CUTS["chars"][0]
        arr = chars_bg.copy()
        a = ease_out_cubic(prog(lf, 2, 12))
        paste_alpha(arr, chars_head, (W - chars_head.width) / 2, 18 - (1 - a) * 10, a)
        for i, c in enumerate(char_imgs):
            st = 4 + i * 9
            t = prog(lf, st, 16)
            if t <= 0:
                continue
            land = lf - (st + 16)
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
                    top = GROUND - c.height - bob
                    paste_alpha(arr, tg, char_x[i] - tg.width / 2, top - 70 - tg.height / 2 + 30, clamp01(ts))
        leaves.draw(arr, f, 0.6)
        return arr

    def cut_group(f):
        lf = f - CUTS["group"][0]
        f0, f1 = CUTS["group"][0] - XF, CUTS["group"][1] + XF
        arr = to_np(kenburns(group_bg, f, f0, f1, 1.07, 1.0, 0.5, 0.45, 0.5, 0.5))
        fg = kenburns(group_fg, f, f0, f1, 1.16, 1.0, 0.5, 0.42, 0.5, 0.5)
        fg = breathe(fg, 0.008 * math.sin(lf * 2 * math.pi / 40), H)
        paste_alpha(arr, fg, 0, 0)
        arr = arr * (1 - 0.62 * bottom_grad)
        draw_chars(arr, *grp_l1[:2], (W - grp_l1[2]) / 2, 750, lf, 4, step=2, dur=10)
        draw_chars(arr, *grp_l2[:2], (W - grp_l2[2]) / 2, 880, lf, 30, step=2, dur=10)
        leaves.draw(arr, f, 0.85)
        return arr

    def cut_title(f):
        lf = f - CUTS["title"][0]
        arr = title_bg.copy()
        # 夕日の光がゆっくり広がる
        r = 0.35 + 0.15 * ease_out_cubic(prog(lf, 0, 60))
        sun = np.exp(-(((_xx - W / 2) / (W * r)) ** 2 + ((_yy - H * 0.38) / (H * r * 1.2)) ** 2))[..., None]
        arr = arr + sun * np.array([0.28, 0.14, 0.03], np.float32)
        layers, xs, tw = title
        pad = 40
        for i, (lay, cx) in enumerate(zip(layers, xs)):
            t = prog(lf, 6 + i * 3, 16)
            if t <= 0:
                continue
            e = ease_out_cubic(t)
            l = lay
            if e < 1:
                l = lay.filter(ImageFilter.GaussianBlur((1 - e) * 10))
            paste_alpha(arr, l, (W - tw) / 2 + cx - pad, 300 - pad + (1 - e) * 24, e)
        a = ease_out_cubic(prog(lf, 34, 14))
        paste_alpha(arr, sub, (W - sub.width) / 2, 560 + (1 - a) * 12, a)
        a = ease_out_cubic(prog(lf, 44, 14))
        paste_alpha(arr, genre, (W - genre.width) / 2, 650, a)
        a = ease_out_cubic(prog(lf, 54, 16))
        if a > 0:
            y = 820
            ln = int(380 * a)
            arr[y:y + 2, W // 2 - ln:W // 2 + ln] = arr[y:y + 2, W // 2 - ln:W // 2 + ln] * 0.4 + 0.6 * np.array(
                [1.0, 0.92, 0.8], np.float32)
            paste_alpha(arr, soon, (W - soon.width) / 2, 830, a)
        flies_title.draw(arr, f, CUTS["title"][0])
        leaves.draw(arr, f, 0.9)
        return arr

    render = {"open": cut_open, "photos": cut_photos, "chars": cut_chars, "group": cut_group, "title": cut_title}
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
            nxt = CUTS[order[idx + 1]][0]
            if f >= nxt - XF:
                w = ease_in_out((f - (nxt - XF)) / XF)
                arr = arr * (1 - w) + render[order[idx + 1]](f) * w
        arr = finish(arr, f)
        # 冒頭は黒からフェードイン、最後は軽くフェードアウト
        arr = arr * ease_in_out(prog(f, 0, 14)) * (1 - 0.85 * ease_in_out(prog(f, N - 10, 10)))
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
    chords = [
        (0.0, 3.0, "A2", ["A3", "E4", "B4", "C5", "E5"]),       # Am9
        (3.0, 6.0, "F2", ["F3", "C4", "E4", "A4", "C5"]),       # Fmaj7
        (6.0, 7.5, "G2", ["G3", "D4", "A4", "B4", "D5"]),       # Gadd9
        (7.5, 9.375, "E2", ["E3", "B3", "D4", "G4", "B4"]),     # Em7
        (9.375, 10.875, "D2", ["D3", "A3", "C4", "F4", "A4"]),  # Dm7
        (10.875, 12.0, "G2", ["G3", "D4", "F4", "B4", "D5"]),   # G7
        (12.0, 15.0, "C2", ["C3", "G3", "B3", "D4", "E4", "G4"]),  # Cmaj9
    ]
    pattern = [0, 2, 1, 3, 2, 4, 3, 1]
    seed = 0
    for c0, c1, bass, tones in chords:
        last = c0 >= 12.0
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
    melody = [(0.75, "E5", 0.75), (1.5, "D5", 0.375), (1.875, "C5", 0.375), (2.25, "D5", 0.75),
              (3.0, "C5", 0.75), (3.75, "A4", 0.75), (4.5, "C5", 0.375), (4.875, "E5", 1.1),
              (6.0, "D5", 0.75), (6.75, "B4", 0.5), (7.25, "D5", 0.25),
              (7.5, "G5", 0.75), (8.25, "E5", 1.1),
              (9.375, "F5", 0.75), (10.125, "E5", 0.375), (10.5, "D5", 0.375),
              (10.875, "D5", 0.5), (11.375, "B4", 0.6),
              (12.0, "C5", 2.5), (12.0, "E5", 2.5), (12.75, "G5", 2.0), (13.5, "C6", 1.5)]
    for t0, n_, d in melody:
        g = 0.22 if t0 < 12 else 0.16
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
    cricket(0.0, 4.2, 4300, 0.035, -0.6)
    cricket(0.4, 3.6, 4620, 0.025, 0.6)
    cricket(12.0, 3.0, 4400, 0.025, 0.5)

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
        t0 = fr / FPS + 0.18
        k = int(0.12 * SR)
        noise = rng.standard_normal(k) * np.exp(-np.arange(k) / (0.025 * SR))
        add(t0, noise - onepole_lp(noise, 600), 0.10, -0.3 if i % 2 == 0 else 0.3)

    # タイトルのウインドチャイム
    for k, nm in enumerate(["E6", "G6", "B6", "D7", "G7"]):
        add(12.15 + 0.07 * k, piano(midi(nn(nm)), 0.2), 0.05, -0.5 + 0.25 * k)

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
    master = np.minimum(1, t / 0.6) * np.clip((DURATION - t) / 1.2, 0, 1) ** 1.2
    st = np.stack([L * master, R * master], 1)
    st = np.tanh(st / (np.abs(st).max() + 1e-9) * 1.1) * 0.85
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
