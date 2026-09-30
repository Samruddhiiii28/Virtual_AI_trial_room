"""Product seed data + auto-generation of transparent PNG garment images (no downloads needed)."""
import os
import cv2, numpy as np
from PIL import Image, ImageDraw

COLORS = {'black': '#1f1f1f', 'white': '#f4f4f4', 'navy': '#1f2f5c', 'blue': '#3b6fd4', 'red': '#d23a3a',
          'grey': '#8d8d92', 'beige': '#d8c3a0', 'olive': '#6b7a3a', 'brown': '#7b5234', 'pink': '#ec8fb0',
          'maroon': '#7a1f2e'}
# name, category, color, style, occasion, price, gender, shape
PRODUCTS = [
 ("Classic White Tee", "top", "white", "minimal", "casual", 499, "unisex", "tshirt"),
 ("Black Graphic Tee", "top", "black", "streetwear", "casual", 599, "unisex", "tshirt"),
 ("Navy Polo Tee", "top", "navy", "classic", "casual", 799, "male", "tshirt"),
 ("Olive Sport Tee", "top", "olive", "sporty", "sports", 649, "unisex", "tshirt"),
 ("Pink Casual Top", "top", "pink", "elegant", "party", 699, "female", "tshirt"),
 ("White Formal Shirt", "top", "white", "classic", "formal", 1299, "unisex", "shirt"),
 ("Blue Oxford Shirt", "top", "blue", "classic", "formal", 1499, "male", "shirt"),
 ("Maroon Party Shirt", "top", "maroon", "elegant", "party", 1399, "unisex", "shirt"),
 ("Black Bomber Jacket", "jacket", "black", "streetwear", "casual", 2499, "unisex", "jacket"),
 ("Beige Blazer", "jacket", "beige", "elegant", "formal", 3299, "unisex", "jacket"),
 ("Grey Zip Jacket", "jacket", "grey", "sporty", "sports", 1999, "unisex", "jacket"),
 ("Blue Slim Jeans", "bottom", "blue", "classic", "casual", 1799, "unisex", "pants"),
 ("Black Formal Trousers", "bottom", "black", "classic", "formal", 1699, "unisex", "pants"),
 ("Beige Chinos", "bottom", "beige", "minimal", "casual", 1599, "unisex", "pants"),
 ("Grey Track Pants", "bottom", "grey", "sporty", "sports", 999, "unisex", "pants"),
 ("Navy Formal Trousers", "bottom", "navy", "elegant", "formal", 1799, "unisex", "pants"),
 ("Brown Cargo Pants", "bottom", "brown", "streetwear", "casual", 1899, "unisex", "pants"),
 ("Black Sport Shorts", "bottom", "black", "sporty", "sports", 699, "unisex", "shorts"),
 ("Red Party Dress", "dress", "red", "elegant", "party", 2799, "female", "dress"),
 ("Navy Formal Dress", "dress", "navy", "classic", "formal", 2999, "female", "dress"),
 ("Pink Summer Dress", "dress", "pink", "minimal", "casual", 1799, "female", "dress"),
 ("Black Cap", "hat", "black", "streetwear", "casual", 399, "unisex", "hat"),
 ("Beige Sun Hat", "hat", "beige", "minimal", "casual", 549, "female", "hat"),
 ("Red Cap", "hat", "red", "sporty", "sports", 399, "unisex", "hat"),
]
SHAPES = {
 'tshirt': [(120,50),(165,50),(200,75),(235,50),(280,50),(370,130),(335,180),(290,150),(290,450),(110,450),(110,150),(65,180),(30,130)],
 'shirt': [(120,50),(165,50),(200,75),(235,50),(280,50),(385,320),(345,335),(290,170),(290,450),(110,450),(110,170),(55,335),(15,320)],
 'jacket': [(115,45),(165,45),(200,70),(235,45),(285,45),(390,330),(340,345),(295,170),(295,460),(105,460),(105,170),(60,345),(10,330)],
 'pants': [(100,0),(300,0),(325,500),(215,500),(200,140),(185,500),(75,500)],
 'shorts': [(100,0),(300,0),(325,230),(215,230),(200,90),(185,230),(75,230)],
 'dress': [(140,0),(165,0),(200,30),(235,0),(260,0),(285,150),(350,500),(50,500),(115,150)],
}

def _shade(h, f):
    r, g, b = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return tuple(max(0, min(255, int(c * f))) for c in (r, g, b)) + (255,)

def _finish(im, shape, hexc):
    """Fabric look: soft folds, lighting gradient, edge darkening, weave / denim texture."""
    a = np.array(im).astype(np.float32); h, w = a.shape[:2]; alpha = a[:, :, 3]
    xx = np.linspace(-1, 1, w)[None, :]; yy = np.linspace(0, 1, h)[:, None]
    shade = 1 - .10 * xx - .07 * yy + .045 * np.sin(xx * 9 + yy * 2.5) + .035 * np.sin(xx * 17 - yy * 4) + .03 * np.sin(yy * 11 + xx * 3)
    d = cv2.distanceTransform((alpha > 128).astype(np.uint8), cv2.DIST_L2, 3)
    shade *= .80 + .20 * np.clip(d / 14, 0, 1)
    rng = np.random.default_rng(3); grain = rng.normal(0, 3.5, (h, w, 1))
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    weave = 3.0 * (np.sin(gx * 1.9) * np.sin(gy * 1.9))[:, :, None]
    if hexc == COLORS['blue'] and shape in ('pants', 'shorts'): weave = 6.0 * np.sin(gx * 1.3 + gy * 1.3)[:, :, None]
    a[:, :, :3] = np.clip(a[:, :, :3] * shade[:, :, None] + grain + weave, 0, 255)
    return Image.fromarray(a.astype(np.uint8), 'RGBA')

def draw(shape, hexc, path):
    im = Image.new("RGBA", (400, 500), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    base, dark, light = _shade(hexc, 1), _shade(hexc, .68), _shade(hexc, 1.18)
    if shape == 'hat':
        d.ellipse((20, 130, 380, 200), fill=dark)
        d.pieslice((70, 20, 330, 280), 180, 360, fill=base, outline=dark, width=4)
        d.line((200, 20, 200, 150), fill=dark, width=3)
    else:
        pts = SHAPES[shape]
        d.polygon(pts, fill=base); d.line(pts + [pts[0]], fill=dark, width=4)
        if shape in ('tshirt', 'shirt', 'jacket'):
            d.pieslice((160, 20, 240, 110), 0, 180, fill=dark)
        if shape == 'shirt':
            d.polygon([(165, 50), (200, 100), (158, 108)], fill=light, outline=dark)
            d.polygon([(235, 50), (200, 100), (242, 108)], fill=light, outline=dark)
            d.line((200, 100, 200, 450), fill=dark, width=3)
            for y in range(150, 440, 60): d.ellipse((194, y, 206, y + 12), fill=light, outline=dark)
        if shape == 'jacket':
            d.line((200, 70, 200, 460), fill=dark, width=5)
            d.rectangle((105, 430, 295, 460), fill=dark)
            d.rectangle((125, 300, 175, 350), outline=dark, width=3); d.rectangle((225, 300, 275, 350), outline=dark, width=3)
        if shape in ('pants', 'shorts'):
            d.rectangle((100, 0, 300, 28), fill=dark)
        if shape == 'dress':
            d.line((115, 150, 285, 150), fill=dark, width=6)
    im = _finish(im.crop(im.getbbox()), shape, hexc); im.save(path)

def ensure_assets():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'clothes')
    os.makedirs(folder, exist_ok=True)
    marker = os.path.join(folder, '.v2'); force = not os.path.exists(marker)
    for i, p in enumerate(PRODUCTS, 1):
        f = os.path.join(folder, f'{i}.png')
        if force or not os.path.exists(f): draw(p[7], COLORS[p[2]], f)
    if force: open(marker, 'w').write('v2')
