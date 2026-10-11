"""CM 用の動かせるレイヤー（人物の切り抜き + 人物を消した背景）を作る.

背景除去モデル（ISNet / @imgly/background-removal-node 同梱の ONNX）で人物を切り抜き、
人物がいた部分は OpenCV で穴埋めして背景だけの画像を作る。

使い方:
    python3 make_layers.py <model.onnx> <assets_dir>
      → <assets_dir>/layers/ に key_fg.png, key_bg.png, group_fg.png, group_bg.png, char0-3.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

sess = None


def mask(img):
    x = np.asarray(img.convert("RGB").resize((1024, 1024), Image.BILINEAR), np.float32)
    x = ((x - 128) / 256).transpose(2, 0, 1)[None]
    m = sess.run(["output"], {"input": x})[0][0, 0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    return Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)


def inpaint(img, m, grow=25):
    """人物部分（少し広げる）を周囲の色で埋めた背景を返す."""
    a = (np.asarray(m) > 40).astype(np.uint8) * 255
    a = cv2.dilate(a, np.ones((grow, grow), np.uint8))
    rgb = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    # 縮小して塗り、拡大して戻す（大きな穴でも滑らかに）
    s = 0.5
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sm = cv2.resize(a, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, sm, 12, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (0, 0), 6)
    filled = cv2.resize(filled, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(a, (0, 0), 6)[..., None] / 255.0
    out = rgb * (1 - soft) + filled * soft
    return Image.fromarray(cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB))


def cutout(img, m):
    rgba = img.convert("RGBA")
    # 半透明の残りかすを消し、輪郭は少しだけぼかす
    m = m.point(lambda v: max(0, min(255, int((v - 70) * 255 / 130))))
    rgba.putalpha(m.filter(ImageFilter.GaussianBlur(0.8)))
    return rgba


def main():
    global sess
    sess = ort.InferenceSession(sys.argv[1])
    assets = Path(sys.argv[2])
    out = assets / "layers"
    out.mkdir(exist_ok=True)

    # キーアート（縁側の主人公と柴犬）: 全体で1回 + 柴犬まわりでもう1回推論して合成
    key = Image.open(assets / "keyart.png").convert("RGB").crop((0, 330, 805, 783))
    m = np.asarray(mask(key), np.float32)
    dog_box = (560, 200, 805, 453)
    md = np.asarray(mask(key.crop(dog_box)), np.float32)
    m[dog_box[1]:dog_box[3], dog_box[0]:dog_box[2]] = np.maximum(
        m[dog_box[1]:dog_box[3], dog_box[0]:dog_box[2]], md)
    # 右端の柱と縦書きコピーは背景に残す（人物と犬だけ前景）
    m[:, :300] = 0
    m[:150, 700:] = 0
    km = Image.fromarray(m.astype(np.uint8))
    cutout(key, km).save(out / "key_fg.png")
    inpaint(key, km).save(out / "key_bg.png")

    group = Image.open(assets / "group.png").convert("RGB")
    gm = mask(group)
    cutout(group, gm).save(out / "group_fg.png")
    inpaint(group, gm, grow=31).save(out / "group_bg.png")

    chars = Image.open(assets / "characters.png").convert("RGB")
    for i, (x0, x1) in enumerate([(0, 386), (390, 768), (772, 1152), (1156, 1536)]):
        p = chars.crop((x0, 0, x1, 1024))
        pm = np.asarray(mask(p)).copy()
        # 名札・吹き出しの残りかすを消す：大きな連結成分だけ残す
        n, lab, stats, _ = cv2.connectedComponentsWithStats((pm > 60).astype(np.uint8))
        big = stats[1:, cv2.CC_STAT_AREA].max()
        keep = np.isin(lab, [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > big * 0.05])
        pm = (pm * cv2.dilate(keep.astype(np.uint8), np.ones((9, 9), np.uint8))).astype(np.uint8)
        c = cutout(p, Image.fromarray(pm))
        c.crop(c.getbbox()).save(out / f"char{i}.png")
        print("char", i, c.getbbox())


if __name__ == "__main__":
    main()
