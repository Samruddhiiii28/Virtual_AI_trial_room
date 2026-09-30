"""Turn a product photo (plain background) into a transparent PNG. Uses rembg if installed, else OpenCV GrabCut."""
import cv2, numpy as np

def _grabcut(bgr):
    h, w = bgr.shape[:2]
    border = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]]).astype(np.float32)
    diff = np.linalg.norm(bgr.astype(np.float32) - np.median(border, axis=0), axis=2)
    mask = np.where(diff > 28, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[diff > 80] = cv2.GC_FGD
    mask[:3, :] = mask[-3:, :] = mask[:, :3] = mask[:, -3:] = cv2.GC_BGD
    try: cv2.grabCut(bgr, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error: pass
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1: fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    inv = ((1 - fg) * 255).astype(np.uint8); cv2.floodFill(inv, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0)
    fg[inv > 0] = 1                                            # fill holes
    alpha = cv2.GaussianBlur(fg * 255, (5, 5), 0)
    return np.dstack([bgr, alpha])

def remove_background(data, use_ai=False):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None: raise ValueError('not a valid image')
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4 and (img[:, :, 3] < 250).mean() > .05: out = img          # already transparent
    else:
        bgr = img[:, :, :3]; s = 1000 / max(bgr.shape[:2])
        if s < 1: bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out = None
        if use_ai:
            try:
                from rembg import remove
                out = cv2.imdecode(np.frombuffer(remove(cv2.imencode('.png', bgr)[1].tobytes()), np.uint8), cv2.IMREAD_UNCHANGED)
            except Exception as e: print('[rembg unavailable]', e)
        if out is None: out = _grabcut(bgr)
    return cv2.imencode('.png', out)[1].tobytes()
