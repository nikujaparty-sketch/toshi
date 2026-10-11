"""高知小津高校 vs 高知国際高等学校 PR動画（10秒版）ジェネレーター.

make_video.py（5秒版）の部品を流用し、6カット構成の縦型動画
（1080x1920 / 30fps / 効果音・BGM付き）を書き出す。

使い方:
    python3 make_video_10s.py <poster.png> <fonts_dir> <out.mp4>
"""
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from make_video import (FPS, GOLD, GOLD_HI, WHITE, Sparks, chroma, cover, ease_in_out,
                        ease_out_back, ease_out_cubic, font, gold_gradient, paste_t,
                        prog, shake, shape_mask, text_layer, to_img, to_np)

DURATION = 10.0
N_FRAMES = int(FPS * DURATION)

# カット構成（開始フレーム）
C_OPEN = 0      # オープニング：照明点灯 + MATCH DAY
C_PLAYER = 45   # 選手アップ + キャッチコピー
C_BEAT = 96     # ビートカット 3連（誇り / 絆 / 勝利）
C_VS = 141      # エンブレム激突
C_TITLE = 186   # チーム集合 + タイトル
C_END = 228     # ポスター + 試合情報
VS_HIT = C_VS + 12
TITLE_HIT = C_TITLE + 3
HITS = [C_PLAYER, C_BEAT, C_BEAT + 15, C_BEAT + 30, VS_HIT, TITLE_HIT, C_END]


def main():
    poster_path, fonts_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    W, H = 1080, 1920
    S = 1.0

    poster = Image.open(poster_path).convert("RGB")
    PW, PH = poster.size
    sx, sy = PW / 1122.0, PH / 1402.0

    def pbox(x0, y0, x1, y1):
        return poster.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))

    player = pbox(300, 0, 700, 418)
    team = pbox(0, 740, 1122, 1265)
    beat_crops = [pbox(110, 760, 480, 1262), pbox(380, 760, 750, 1262), pbox(690, 760, 1060, 1262)]
    ozu = pbox(48, 22, 348, 378)
    kok = pbox(782, 78, 1090, 362)
    ozu.putalpha(shape_mask(ozu.size, "shield", 10))
    kok.putalpha(shape_mask(kok.size, "circle", 6))
    bg_blur = cover(poster, W, H).filter(ImageFilter.GaussianBlur(28))
    bg_dark = to_np(bg_blur) * 0.30
    # オープニング用：スタジアム（ポスター中段の観客席）をぼかした背景
    stadium = cover(pbox(0, 770, 1122, 1265), W, H).filter(ImageFilter.GaussianBlur(34))
    stadium_np = to_np(stadium)

    F_brush = lambda s: font(fonts_dir, "YujiSyuku.ttf", int(s))
    F_black = lambda s: font(fonts_dir, "NotoSansJP-Black.ttf", int(s))
    F_anton = lambda s: font(fonts_dir, "Anton.ttf", int(s))
    goldfill = gold_gradient()
    pur_glow = (180, 110, 255)

    # ---- テキスト素材
    copy_layers = []
    for line in ["仲間と、", "この一戦を超えていく。"]:
        copy_layers.append((line, [text_layer(c, F_brush(78), WHITE, stroke=2, stroke_fill=(30, 0, 50),
                                              glow=10, glow_color=(170, 90, 255)) for c in line]))
    t_matchday = text_layer("MATCH DAY", F_anton(220), goldfill, stroke=5, stroke_fill=(60, 25, 0),
                            glow=24, glow_color=(255, 170, 40))
    t_taikai_big = text_layer("高校サッカー選手権大会", F_black(76), WHITE, stroke=4,
                              stroke_fill=(40, 0, 70), glow=14, glow_color=(160, 90, 255))
    t_school_en = text_layer("KOCHI OZU HIGH SCHOOL FOOTBALL", F_anton(54), (210, 180, 255),
                             glow=8, glow_color=(140, 60, 255))
    beat_words = [
        (text_layer("誇り", F_brush(380), goldfill, stroke=7, stroke_fill=(60, 25, 0), glow=26,
                    glow_color=(255, 170, 40), bold=6), "PRIDE"),
        (text_layer("絆", F_brush(480), WHITE, stroke=7, stroke_fill=(40, 0, 70), glow=26,
                    glow_color=pur_glow, bold=6), "UNITY"),
        (text_layer("勝利", F_brush(380), goldfill, stroke=7, stroke_fill=(60, 25, 0), glow=26,
                    glow_color=(255, 170, 40), bold=6), "VICTORY"),
    ]
    beat_words = [(w, text_layer(en, F_anton(120), (235, 225, 255), stroke=3, stroke_fill=(40, 0, 70),
                                 glow=10, glow_color=pur_glow)) for w, en in beat_words]
    t_murasaki = text_layer("紫", F_brush(330), goldfill, stroke=5, stroke_fill=(70, 30, 0),
                            glow=18, glow_color=(255, 190, 60), bold=5)
    t_rest = text_layer("の誇りを、", F_brush(190), WHITE, stroke=4, stroke_fill=(40, 0, 60),
                        glow=14, glow_color=pur_glow, bold=4)
    t_rest2 = text_layer("ここに。", F_brush(250), WHITE, stroke=4, stroke_fill=(40, 0, 60),
                         glow=14, glow_color=pur_glow, bold=4)
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

    sparks = Sparks(W, H, n=160)
    sparks.add_burst(W / 2, H * 0.47, 60, 30, 10)          # MATCH DAY
    for i in range(3):
        sparks.add_burst(W / 2, H * 0.70, 55, C_BEAT + i * 15 + 1, 20 + i)
    sparks.add_burst(W / 2, H * 0.42, 90, VS_HIT, 1)
    sparks.add_burst(W * 0.19, H * 0.27, 70, TITLE_HIT, 2)
    sparks.add_burst(W / 2, H * 0.5, 60, C_END, 3)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = (1 - 0.55 * np.clip(((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2 * 0.8) * 2.4, 0, 1))[..., None]
    rng = np.random.default_rng(3)
    grain = [rng.normal(0, 0.035, (H // 4, W // 4)).astype(np.float32) for _ in range(6)]

    def light_sweep(f, f0, length, angle=0.35, width=0.08, strength=0.9, color=(1.0, 0.85, 0.55)):
        t = prog(f, f0, length)
        if t <= 0 or t >= 1:
            return 0
        u = (xx / W) + (yy / H) * angle
        band = np.exp(-((u - (-0.3 + 1.6 * ease_in_out(t))) / width) ** 2) * strength
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

    def god_rays(cx, cy, strength, t, color=(0.75, 0.6, 1.0)):
        """照明から下に伸びる光の筋."""
        ang = np.arctan2(yy - cy, xx - cx)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / H
        rays = (0.5 + 0.5 * np.sin(ang * 26 + t * 0.6)) ** 6 * np.exp(-dist * 2.2)
        return (rays * strength)[..., None] * np.array(color, np.float32)

    def whip(arr, lf, direction=1):
        """カット頭の横流れモーションブラー."""
        if lf >= 4:
            return arr
        k = int((4 - lf) * 22) * direction
        acc = arr.copy()
        for i in range(1, 6):
            acc += np.roll(arr, k * i // 5, axis=1)
        return acc / 6

    frames_dir = out_path.parent / "_frames10"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cshift = 0

    for f in range(N_FRAMES):
        if f < C_PLAYER:
            # ---------------- 1. オープニング：照明点灯
            lf = f - C_OPEN
            z = 1.0 + 0.05 * prog(f, 0, C_PLAYER)
            arr = stadium_np * (0.10 + 0.32 * ease_out_cubic(prog(f, 4, 20)))
            arr = arr * (1 + 0 * z)
            for i, (lx, on) in enumerate([(0.12, 4), (0.88, 10), (0.5, 16)]):
                p = prog(f, on, 3)
                if p > 0:
                    flick = 1.0 if f > on + 4 else (0.4 + 0.6 * ((f - on) % 2))
                    arr = arr + flare(W * lx, H * 0.08, W * 0.25, 0.9 * p * flick, (0.95, 0.9, 1.0))
                    arr = arr + god_rays(W * lx, H * 0.06, 0.22 * p * flick, f + i * 10)
            base = to_img(arr).convert("RGBA")
            pt = prog(f, 14, 10)
            if pt > 0:
                paste_t(base, t_taikai_big, W / 2, H * 0.40, alpha=pt, scale=1.1 - 0.1 * ease_out_cubic(pt))
            pm = prog(f, 28, 6)
            if pm > 0:
                paste_t(base, t_matchday, W / 2, H * 0.50, scale=(2.2 - 1.2 * ease_out_back(pm, 1.5)),
                        alpha=min(1, pm * 2.5))
            pe = prog(f, 32, 8)
            if pe > 0:
                paste_t(base, t_school_en, W / 2, H * 0.605, alpha=pe)
            arr = to_np(base)
            arr = arr + light_sweep(f, 33, 12, angle=0.3, width=0.05, strength=0.6)
            arr = arr + flash(f, 28, 5, 0.6, (1.0, 0.9, 0.7))
            arr = arr + sparks.render(f, 0.25 + 0.5 * prog(f, 10, 30))
            arr = arr * ease_out_cubic(prog(f, 0, 4) * 0.7 + 0.3)
            cshift = 8 * math.exp(-max(0, f - 28) / 3) if f >= 28 else 0
        elif f < C_BEAT:
            # ---------------- 2. 選手アップ + キャッチコピー
            lf = f - C_PLAYER
            t = prog(f, C_PLAYER, C_BEAT - C_PLAYER)
            img = cover(player, W, H, zoom=1.22 - 0.18 * ease_out_cubic(t), cx=0.52, cy=0.46 - 0.05 * t)
            arr = to_np(img)
            fade_bot = np.clip((yy / H - 0.55) / 0.45, 0, 1)[..., None]
            arr = arr * (1 - 0.75 * fade_bot) + fade_bot * np.array([0.18, 0.05, 0.32]) * 0.75
            base = to_img(arr).convert("RGBA")
            col_x = [W * 0.86, W * 0.73]
            k = 0
            for li, (line, chars) in enumerate(copy_layers):
                for ci, ch in enumerate(chars):
                    p = prog(lf, 4 + k * 2.1, 6)
                    if p > 0:
                        cy_ = H * 0.12 + ci * 84 - (40 if line[ci] in "、。" else 0)
                        paste_t(base, ch, col_x[li] + (1 - p) * 30, cy_,
                                scale=1.0 + 0.6 * (1 - ease_out_cubic(p)), alpha=p)
                    k += 1
            arr = to_np(base)
            arr = arr + light_sweep(f, C_PLAYER + 6, 30, angle=-0.5, width=0.06, strength=0.5)
            arr = arr + flash(f, C_PLAYER, 8, 1.0)
            arr = arr + sparks.render(f, 0.6)
            cshift = 10 * math.exp(-lf / 3)
        elif f < C_VS:
            # ---------------- 3. ビートカット 3連
            lf = f - C_BEAT
            idx, sf = lf // 15, lf % 15
            crop = beat_crops[idx]
            t = sf / 15
            img = cover(crop, W, H, zoom=1.04 + 0.08 * t, cx=0.5 + (0.04 - 0.08 * t) * (1 if idx % 2 else -1), cy=0.42)
            arr = to_np(img)
            # 下部を暗く・紫に
            fade_bot = np.clip((yy / H - 0.45) / 0.55, 0, 1)[..., None]
            arr = arr * (1 - 0.7 * fade_bot) + fade_bot * np.array([0.16, 0.04, 0.30]) * 0.7
            arr = whip(arr, sf, 1 if idx % 2 == 0 else -1)
            base = to_img(arr).convert("RGBA")
            word, en = beat_words[idx]
            pw = prog(sf, 1, 4)
            if pw > 0:
                paste_t(base, word, W / 2, H * 0.70, scale=(1.9 - 0.9 * ease_out_back(pw, 1.6)) * 0.85,
                        alpha=min(1, pw * 3), rot=(1 - pw) * (8 if idx % 2 else -8))
            pe = prog(sf, 4, 5)
            if pe > 0:
                paste_t(base, en, W / 2 + (1 - ease_out_cubic(pe)) * 160 * (1 if idx % 2 else -1),
                        H * 0.875, alpha=pe, scale=1.0)
            arr = to_np(base)
            arr = arr + flash(f, C_BEAT + idx * 15, 4, 0.9, (1.0, 0.92, 0.8))
            arr = arr + sparks.render(f, 0.6)
            cshift = 14 * math.exp(-sf / 2.5)
        elif f < C_TITLE:
            # ---------------- 4. エンブレム激突 VS
            lf = f - C_VS
            arr = bg_dark.copy()
            pulse = 0.5 + 0.5 * ease_out_cubic(prog(f, VS_HIT, 8))
            arr = arr + flare(W / 2, H * 0.42, W * 0.7, 0.35 * pulse, (0.55, 0.25, 1.0))
            base = to_img(arr).convert("RGBA")
            em_y = H * 0.42
            p = ease_out_back(prog(f, C_VS, 12), 1.2)
            gap = W * 0.25
            esc = 1.15 * (1.0 + 0.08 * math.exp(-max(0, f - VS_HIT) / 4.0) * (f >= VS_HIT))
            sep = 0 if f < VS_HIT else 18 * ease_out_cubic(prog(f, VS_HIT, 10))
            # 衝突後はゆっくり寄る
            drift = 1 + 0.04 * prog(f, VS_HIT, C_TITLE - VS_HIT)
            lx = -W * 0.4 + (W / 2 - gap - sep + W * 0.4) * p
            rx = W * 1.4 - (W * 1.4 - (W / 2 + gap + sep)) * p
            paste_t(base, ozu, W / 2 + (lx - W / 2) * drift, em_y, scale=esc * drift, rot=(1 - p) * 25)
            paste_t(base, kok, W / 2 + (rx - W / 2) * drift, em_y, scale=esc * 1.02 * drift, rot=-(1 - p) * 25)
            pv = prog(f, VS_HIT - 1, 9)
            if pv > 0:
                paste_t(base, t_vs, W / 2, em_y + 10, scale=(2.6 - 1.6 * ease_out_back(pv, 1.6)) * 0.62 * drift,
                        alpha=min(1, pv * 3))
            pn = prog(f, VS_HIT + 4, 8)
            if pn > 0:
                ny = em_y + 300
                paste_t(base, t_ozu, W / 2 - (1 - ease_out_cubic(pn)) * W * 0.6, ny, alpha=pn, scale=0.95)
                paste_t(base, t_kok, W / 2 + (1 - ease_out_cubic(pn)) * W * 0.6, ny + 130, alpha=pn, scale=0.95)
                vs_small = prog(f, VS_HIT + 6, 6)
                if vs_small > 0:
                    paste_t(base, t_vs, W / 2, ny + 66, scale=0.18, alpha=vs_small)
            pt = prog(f, C_VS + 4, 8)
            if pt > 0:
                paste_t(base, t_taikai, W / 2, em_y - 300, alpha=pt, scale=0.9 + 0.1 * ease_out_cubic(pt))
            arr = to_np(base)
            if f < VS_HIT:
                streak = np.exp(-((yy - em_y) / (H * 0.13)) ** 2)[..., None] * 0.12 * (lf / 12)
                arr = arr + streak * np.array([0.7, 0.5, 1.0])
            arr = arr + flash(f, VS_HIT, 5, 0.9, (1.0, 0.95, 0.85))
            arr = arr + flash(f, C_VS, 4, 0.6)
            arr = arr + light_sweep(f, VS_HIT + 14, 16, angle=0.3, width=0.05, strength=0.4)
            arr = arr + sparks.render(f, 0.5)
            cshift = 14 * math.exp(-max(0, f - VS_HIT) / 3) if f >= VS_HIT else 0
        elif f < C_END:
            # ---------------- 5. チーム集合 + タイトル
            t = prog(f, C_TITLE, C_END - C_TITLE)
            img = cover(team, W, H, zoom=1.0 + 0.07 * t, cx=0.54 - 0.08 * t, cy=0.55)
            arr = to_np(img)
            topd = np.clip(1 - yy / (H * 0.62), 0, 1)[..., None]
            arr = arr * (1 - 0.65 * topd) + topd * np.array([0.12, 0.03, 0.22]) * 0.65
            arr = arr + flare(W * (0.85 - 0.3 * t), H * 0.12, W * 0.45, 0.45, (0.9, 0.75, 1.0))
            base = to_img(arr).convert("RGBA")
            ty = H * 0.27
            pm = prog(f, TITLE_HIT, 5)
            if pm > 0:
                paste_t(base, t_murasaki, W * 0.19, ty, scale=0.9 * (2.4 - 1.4 * ease_out_back(pm, 1.4)),
                        alpha=min(1, pm * 2.5), rot=(1 - pm) * -12)
            pr = prog(f, TITLE_HIT + 5, 6)
            if pr > 0:
                paste_t(base, t_rest, W * 0.615 + (1 - ease_out_cubic(pr)) * 120, ty + 40, scale=0.84, alpha=pr)
            pk = prog(f, TITLE_HIT + 10, 6)
            if pk > 0:
                paste_t(base, t_rest2, W * 0.58, ty + 290, scale=0.9 * (1.5 - 0.5 * ease_out_back(pk, 1.6)),
                        alpha=min(1, pk * 2))
            pl = prog(f, TITLE_HIT + 12, 8)
            if pl > 0:
                ld = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(ld)
                x_end = W * 0.08 + (W * 0.84) * ease_out_cubic(pl)
                yl = ty + 450
                for i in range(7):
                    off = (i - 3) * 3
                    dd.line((W * 0.08, yl + off + i * 0.6, x_end, yl - 18 + off),
                            fill=(*GOLD, 210 - abs(i - 3) * 50), width=5)
                ld = Image.alpha_composite(ld.filter(ImageFilter.GaussianBlur(6)), ld)
                base.alpha_composite(ld)
            arr = to_np(base)
            arr = arr + flash(f, C_TITLE, 5, 0.8)
            arr = arr + flash(f, TITLE_HIT, 5, 0.7, (1.0, 0.85, 0.5))
            arr = arr + light_sweep(f, TITLE_HIT + 18, 16, angle=0.25, width=0.05, strength=0.5)
            arr = arr + sparks.render(f, 0.8)
            cshift = 10 * math.exp(-max(0, f - TITLE_HIT) / 3) if f >= TITLE_HIT else 0
        else:
            # ---------------- 6. ポスター + 試合情報
            lf = f - C_END
            arr = bg_dark.copy() * 0.9
            arr = arr + flare(W / 2, H * 0.36, W * 0.8, 0.18 + 0.05 * math.sin(lf * 0.2), (0.5, 0.25, 0.9))
            base = to_img(arr).convert("RGBA")
            pz = ease_out_cubic(prog(f, C_END, 14))
            psc = W * 0.80 / PW * (1.12 - 0.12 * pz) * (1 + 0.025 * prog(f, C_END + 14, N_FRAMES - C_END - 14))
            py = H * 0.355
            shadow = Image.new("RGBA", (int(PW * psc) + 80, int(PH * psc) + 80), (0, 0, 0, 0))
            ImageDraw.Draw(shadow).rectangle((40, 40, shadow.width - 40, shadow.height - 40), fill=(10, 0, 25, 200))
            shadow = shadow.filter(ImageFilter.GaussianBlur(24))
            paste_t(base, shadow, W / 2, py + 16, alpha=pz)
            frame = Image.new("RGBA", (PW + 16, PH + 16), (*GOLD, 255))
            frame.paste(poster.convert("RGBA"), (8, 8))
            paste_t(base, frame, W / 2, py, scale=psc, alpha=min(1, pz * 1.4))
            ph_disp = PH * psc
            paste_t(base, t_school_en, W / 2, py - ph_disp / 2 - 70, scale=0.9, alpha=prog(f, C_END + 2, 8))
            info_y = py + ph_disp / 2 + 115
            pd = prog(f, C_END + 8, 7)
            if pd > 0:
                e = ease_out_back(pd, 1.5)
                paste_t(base, t_date, W / 2 - 140, info_y, scale=0.9 * (1.6 - 0.6 * e), alpha=min(1, pd * 2))
                paste_t(base, t_sun, W / 2 + 140, info_y + 22, scale=0.9 * (1.6 - 0.6 * e), alpha=min(1, pd * 2))
            pko = prog(f, C_END + 14, 7)
            if pko > 0:
                paste_t(base, t_kick, W / 2 + (1 - ease_out_cubic(pko)) * W * 0.5, info_y + 160, scale=0.9, alpha=pko)
            pv_ = prog(f, C_END + 20, 7)
            if pv_ > 0:
                paste_t(base, t_venue, W / 2, info_y + 270, scale=0.95, alpha=pv_)
            pc = prog(f, C_END + 28, 8)
            if pc > 0:
                paste_t(base, t_cheer, W / 2, info_y + 360,
                        scale=1 + 0.03 * math.sin(lf * 0.4), alpha=pc * (0.8 + 0.2 * math.sin(lf * 0.4)))
            arr = to_np(base)
            arr = arr + light_sweep(f, C_END + 16, 20, angle=0.4, width=0.05, strength=0.45)
            arr = arr + light_sweep(f, C_END + 50, 18, angle=0.4, width=0.04, strength=0.35)
            arr = arr + flash(f, C_END, 7, 1.1, (1.0, 0.92, 0.8))
            arr = arr + sparks.render(f, 0.9)
            arr *= 1 - 0.35 * ease_in_out(prog(f, N_FRAMES - 5, 5))
            cshift = 8 * math.exp(-lf / 3)

        arr = chroma(arr, cshift)
        dx, dy = shake(f, HITS, amp=24)
        if dx or dy:
            arr = np.roll(arr, (dy, dx), axis=(0, 1))
        arr = arr * vign
        g = np.asarray(Image.fromarray(((grain[f % 6] + 0.5) * 255).clip(0, 255).astype(np.uint8)).resize((W, H)),
                       np.float32) / 255 - 0.5
        arr = np.clip(arr + g[..., None] * 0.6, 0, 1) ** 0.95
        to_img(arr).save(frames_dir / f"{f:04d}.png", compress_level=1)
        if f % 30 == 0:
            print(f"frame {f}/{N_FRAMES}", flush=True)

    audio = out_path.parent / "_sfx10.wav"
    make_audio(audio)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(frames_dir / "%04d.png"), "-i", str(audio),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out_path)
    ], check=True)
    print("written", out_path)


# -------------------------------------------------------------------- audio
def make_audio(path, sr=44100):
    n = int(sr * DURATION)
    out = np.zeros((n, 2), np.float64)
    rng = np.random.default_rng(11)
    hit = lambda fr: fr / FPS

    def add(i0, sig, pan=0.0):
        m = min(n - i0, len(sig))
        if m <= 0:
            return
        out[i0:i0 + m, 0] += sig[:m] * (1 - pan) ** 0.5
        out[i0:i0 + m, 1] += sig[:m] * (1 + pan) ** 0.5

    def boom(t0, gain=1.0, f_hi=120, f_lo=38, dur=1.1):
        m = int(dur * sr)
        tt = np.arange(m) / sr
        freq = f_lo + (f_hi - f_lo) * np.exp(-tt * 9)
        body = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-tt * 3.2)
        click = rng.normal(0, 1, m) * np.exp(-tt * 60) * 0.5
        crack = rng.normal(0, 1, m) * np.exp(-tt * 9) * 0.18
        add(int(t0 * sr), gain * (body + click + crack))

    def lowpass_noise(m, k_from, k_to):
        noise = rng.normal(0, 1, m)
        k = np.linspace(k_from, k_to, m).astype(int)
        cs = np.cumsum(np.concatenate([[0], noise]))
        idx = np.arange(m)
        lo = np.maximum(idx - k, 0)
        return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)

    def whoosh(t0, t1, gain=0.4, pan=0.0):
        m = int((t1 - t0) * sr)
        add(int(t0 * sr), gain * lowpass_noise(m, 40, 4) * np.linspace(0, 1, m) ** 2.2 * 6, pan)

    def riser(t0, t1, gain=0.1, base=220):
        m = int((t1 - t0) * sr)
        tt = np.arange(m) / sr
        L = t1 - t0
        ph = 2 * np.pi * np.cumsum(base * (2 ** (tt / L * 2))) / sr
        add(int(t0 * sr), gain * (np.sin(ph) + 0.5 * np.sin(ph * 1.5) + 0.3 * np.sin(ph * 2.01)) * (tt / L) ** 1.5)

    def shimmer(t0, dur, gain=0.08):
        m = int(dur * sr)
        tt = np.arange(m) / sr
        sig = np.zeros(m)
        for fr in [1318.5, 1567.98, 1975.5, 2637.0, 3135.9]:
            sig += np.sin(2 * np.pi * fr * tt + rng.uniform(0, 6)) * (0.5 + 0.5 * np.sin(2 * np.pi * 9 * tt))
        add(int(t0 * sr), gain * sig * np.exp(-tt * 1.6))

    def clap(t0, gain=0.25):
        m = int(0.25 * sr)
        tt = np.arange(m) / sr
        env = np.exp(-tt * 28) + 0.6 * np.exp(-np.maximum(tt - 0.012, 0) * 30) * (tt > 0.012)
        sig = (rng.normal(0, 1, m) - lowpass_noise(m, 12, 12)) * env
        add(int(t0 * sr), gain * sig)

    def pad(t0, t1, freqs, gain=0.05):
        m = int((t1 - t0) * sr)
        tt = np.arange(m) / sr
        sig = np.zeros(m)
        for fr in freqs:
            for det in (-0.6, 0.6):
                for h in range(1, 6):
                    sig += np.sin(2 * np.pi * fr * h * (1 + det / 1200 * 8) * tt) / h ** 1.6
        env = np.clip(tt / 0.6, 0, 1) * np.clip((t1 - t0 - tt) / 0.3, 0, 1)
        add(int(t0 * sr), gain * sig * env / len(freqs))

    def bass(t0, fr, dur, gain=0.22):
        m = int(dur * sr)
        tt = np.arange(m) / sr
        sig = np.sin(2 * np.pi * fr * tt) + 0.35 * np.sin(2 * np.pi * fr * 2 * tt)
        add(int(t0 * sr), gain * sig * np.exp(-tt * 3.5) * np.clip(tt / 0.005, 0, 1))

    beat = 60 / 128
    hits_s = [hit(h) for h in HITS]
    # ビート：オープニング後半から入り、選手カット以降はフル
    bt = beat * 2
    i = 0
    while bt < DURATION - 0.8:
        near_hit = any(abs(bt - h) < 0.1 for h in hits_s)
        if not near_hit:
            boom(bt, 0.36, f_hi=150, f_lo=48, dur=0.35)
        if bt > hit(C_PLAYER) and i % 2 == 1:
            clap(bt, 0.18)
        bt += beat
        i += 1
    # ベースライン（D マイナー）
    roots = [73.42, 73.42, 58.27, 65.41]  # D, D, Bb, C
    bt = hit(C_PLAYER)
    j = 0
    while bt < hit(C_END) + beat * 7:
        bass(bt, roots[(j // 4) % 4], beat * 0.9)
        bt += beat
        j += 1
    # パッド
    pad(0.0, hit(C_VS), [146.83, 174.61, 220.0], 0.05)          # Dm
    pad(hit(C_VS), hit(C_END), [116.54, 146.83, 174.61], 0.05)  # Bb
    pad(hit(C_END), DURATION, [146.83, 220.0, 293.66, 349.23], 0.06)  # Dm 広め
    # 効果音
    riser(0.1, hit(C_PLAYER) - 0.02, 0.08, 180)
    riser(hit(C_BEAT + 30), hit(VS_HIT) - 0.02, 0.10, 260)
    boom(hit(4), 0.35, 90, 40, 0.6)
    boom(hit(10), 0.35, 90, 40, 0.6)
    boom(hit(16), 0.45, 90, 40, 0.8)
    boom(hit(28), 0.6)
    whoosh(hit(C_PLAYER) - 0.4, hit(C_PLAYER), 0.4)
    for k, h in enumerate(HITS[1:4]):
        whoosh(hit(h) - 0.25, hit(h), 0.35, -0.6 if k % 2 == 0 else 0.6)
    whoosh(hit(VS_HIT) - 0.45, hit(VS_HIT), 0.5)
    whoosh(hit(TITLE_HIT) - 0.4, hit(TITLE_HIT), 0.4)
    whoosh(hit(C_END) - 0.4, hit(C_END), 0.4)
    for h, g in zip(HITS, [0.8, 0.7, 0.75, 0.85, 1.0, 0.9, 1.0]):
        boom(hit(h), g, dur=1.6 if h == C_END else 1.1)
    shimmer(hit(28) + 0.05, 1.0, 0.05)
    shimmer(hit(TITLE_HIT) + 0.1, 0.9, 0.05)
    shimmer(hit(C_END) + 0.1, 2.0, 0.08)
    fo = int(0.5 * sr)
    out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    out = np.tanh(out * 1.3) * 0.85
    pcm = (out * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
