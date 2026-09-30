"""Virtual try-on engine.
MediaPipe Pose -> body landmarks (shoulders, elbows, wrists, hips, knees, ankles)
OpenCV + thin-plate-spline warp -> garment is bent to follow arms / legs / torso in real time."""
import os, math, threading
import cv2, numpy as np
from scipy.interpolate import RBFInterpolator
try:
    import mediapipe as mp
    _ = mp.solutions.pose
except Exception as e:  # pragma: no cover
    print('[WARN] MediaPipe unavailable:', e); mp = None

N = lambda p: np.asarray(p, float)

def prep(path, maxside=700):
    """Load RGBA garment, crop to content, premultiply alpha."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim < 3 or im.shape[2] < 4: return None
    s = maxside / max(im.shape[:2])
    if s < 1: im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ys, xs = np.where(im[:, :, 3] > 20)
    if not len(ys): return None
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = im[:, :, 3:4].astype(np.float32) / 255
    return np.dstack([im[:, :, :3].astype(np.float32) * a, a * 255]).astype(np.uint8)

def anchors(pm, cat):
    """Find shoulder / hem / sleeve-tip / leg points automatically from the garment's alpha mask."""
    m = pm[:, :, 3] > 40; H, W = m.shape; band = max(3, int(H * .04))
    def ext(y0, y1):
        c = np.where(m[y0:y1].any(axis=0))[0]
        return (float(c.min()), float(c.max())) if len(c) else (0., W - 1.)
    a = {'W': W, 'H': H}
    if cat == 'hat': return a
    l, r = ext(0, band); a['SL'] = (l, band / 2); a['SR'] = (r, band / 2)
    hl, hr = ext(H - band, H); a['HL'] = (hl, H - 1.); a['HR'] = (hr, H - 1.)
    a['NC'] = ((l + r) / 2, 0.); a['BC'] = ((hl + hr) / 2, H - 1.)
    ys, xs = np.where(m)
    if cat in ('top', 'jacket'):
        for sd in 'LR':
            s = N(a['S' + sd])
            sel = (xs < hl - .04 * W) if sd == 'L' else (xs > hr + .04 * W)
            sel &= ys > band
            if sel.sum() > 50:
                px, py = xs[sel], ys[sel]; d = np.hypot(px - s[0], py - s[1])
                if d.max() > .12 * H:
                    k = d > .75 * d.max(); a['P' + sd] = (float(px[k].mean()), float(py[k].mean()))
    if cat == 'dress':
        wl, wr = ext(int(H * .33), int(H * .37) + 1); a['WL'] = (wl, H * .35); a['WR'] = (wr, H * .35)
    if cat == 'bottom':
        a['WL'], a['WR'] = a['SL'], a['SR']; cx = (l + r) / 2
        rows = np.where(m[:, int(cx)])[0]; cy = float(rows.max()) if len(rows) else H * .4
        if cy > .5 * H: cy = .4 * H
        a['CR'] = (cx, cy)
        bx = np.where(m[H - band:].any(axis=0))[0]; lx, rx = bx[bx < cx], bx[bx >= cx]
        a['AL'] = (float(lx.mean()) if len(lx) else (l + cx) / 2, H - 1.)
        a['AR'] = (float(rx.mean()) if len(rx) else (r + cx) / 2, H - 1.)
        ym = int(cy + (H - cy) * .5)
        for sd in 'LR':
            xr = np.where(m[ym])[0]; xr = xr[xr < cx] if sd == 'L' else xr[xr >= cx]
            a['K' + sd] = (float(xr.mean()) if len(xr) else a['A' + sd][0], float(ym))
    return a

def split_parts(pm, a, cat):
    """Split a top/jacket into torso + two sleeves (each in two hinged segments) so sleeves can follow the arms."""
    if cat not in ('top', 'jacket') or 'PL' not in a or 'PR' not in a: return None
    H, W = pm.shape[:2]
    poly = np.array([a['SL'], a['SR'], a['HR'], a['HL']], np.int32)
    inner = np.zeros((H, W), np.uint8); cv2.fillPoly(inner, [poly], 255)
    tor = cv2.dilate(inner, np.ones((9, 9), np.uint8))
    parts = {'torso': pm * (tor > 0)[:, :, None].astype(np.uint8)}
    yy, xx = np.mgrid[0:H, 0:W]; cx = a['NC'][0]
    for sd in 'LR':
        out = (tor == 0) & ((xx < cx) if sd == 'L' else (xx >= cx)) & (pm[:, :, 3] > 40)
        n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8))       # keep only the sleeve blob
        if n > 1: out = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        S0, P = N(a['S' + sd]), N(a['P' + sd]); v = P - S0
        t = ((xx - S0[0]) * v[0] + (yy - S0[1]) * v[1]) / max(v @ v, 1e-6)
        root = out & (t < .15)
        Q = np.array([xx[root].mean(), yy[root].mean()]) if root.sum() > 20 else S0
        v = P - Q; t = ((xx - Q[0]) * v[0] + (yy - Q[1]) * v[1]) / max(v @ v, 1e-6)
        parts['Q' + sd], parts['M' + sd] = Q, (Q + P) / 2
        parts[sd + '1'] = pm * (out & (t < .56))[:, :, None].astype(np.uint8)
        parts[sd + '2'] = pm * (out & (t >= .44))[:, :, None].astype(np.uint8)
    return parts

def poly_point(pts, d):
    for p, q in zip(pts[:-1], pts[1:]):
        seg = np.linalg.norm(q - p)
        if seg < 1e-6: continue
        if d <= seg: return p + (q - p) * d / seg
        d -= seg
    p, q = pts[-2], pts[-1]; return q + (q - p) / max(np.linalg.norm(q - p), 1e-6) * d

def rot(fg, angle):
    h, w = fg.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(fg, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def blend(bg, fg, x, y, gain=1.0):
    fh, fw = fg.shape[:2]; H, W = bg.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + fw, W), min(y + fh, H)
    if x0 >= x1 or y0 >= y1: return
    f = fg[y0 - y:y1 - y, x0 - x:x1 - x]; a = f[:, :, 3:4].astype(np.float32) / 255
    bg[y0:y1, x0:x1] = np.clip(f[:, :, :3] * gain + bg[y0:y1, x0:x1] * (1 - a), 0, 255).astype(np.uint8)

class TryOnEngine:
    ORDER = ['bottom', 'dress', 'top', 'jacket', 'hat']

    def __init__(self, folder):
        self.folder, self.cache, self.lock, self.pose = folder, {}, threading.Lock(), None
        self.prev, self.prev_size = None, None
        if mp: self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

    def _asset(self, it):
        key = (it['id'], it['category'])
        if key not in self.cache:
            pm = prep(os.path.join(self.folder, f"{it['id']}.png"))
            if pm is None: self.cache[key] = None
            else:
                a = anchors(pm, it['category']); self.cache[key] = (pm, a, split_parts(pm, a, it['category']))
        return self.cache[key]

    @staticmethod
    def _msg(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

    # ---- control points: garment anchor -> body position ----
    def _points(self, cat, name, a, b, sleeves=True):
        sL, sR, hL, hR = b['sL'], b['sR'], b['hL'], b['hR']
        c, hm = (sL + sR) / 2, (hL + hR) / 2; T = max(np.linalg.norm(hm - c), 1.); up = (c - hm) / T
        A = lambda k: N(a[k]); S, D = [], []; info = {}
        def add(k, p): S.append(A(k)); D.append(N(p))
        def addp(sp, p): S.append(N(sp)); D.append(N(p))
        sLo, sRo = c + (sL - c) * 1.08, c + (sR - c) * 1.08
        NCd = c + up * .1 * T
        if cat in ('top', 'jacket'):
            ext = .22 if cat == 'top' else .32
            HLd, HRd = sLo + (hL - sL) * (1 + ext), sRo + (hR - sR) * (1 + ext)
            hc = (HLd + HRd) / 2; HLd, HRd = hc + (HLd - hc) * 1.12, hc + (HRd - hc) * 1.12
            add('SL', sLo); add('SR', sRo); add('NC', NCd); add('HL', HLd); add('HR', HRd); add('BC', hc)
            addp((A('SL') + A('HL')) / 2, (sLo + HLd) / 2); addp((A('SR') + A('HR')) / 2, (sRo + HRd) / 2)
            addp((A('NC') + A('BC')) / 2, (NCd + hc) / 2)
            gt, bt = np.linalg.norm(A('BC') - A('NC')), np.linalg.norm(hc - NCd); info['gt'], info['bt'] = gt, bt
            for sd, sh, el, wr in (('L', sLo, b['eL'], b['wL']), ('R', sRo, b['eR'], b['wR'])):
                if sleeves and 'P' + sd in a:           # sleeve follows the real arm (shoulder -> elbow -> wrist)
                    P, S0 = A('P' + sd), A('S' + sd); ln = np.linalg.norm(P - S0) / gt * bt
                    poly = [sh, el, wr, wr + (wr - el)]
                    addp((S0 + P) / 2, poly_point(poly, ln / 2)); addp(P, poly_point(poly, ln))
        elif cat == 'bottom':
            short = 'short' in name.lower()
            WLd, WRd = hL + (hL - hm) * .3 + up * .12 * T, hR + (hR - hm) * .3 + up * .12 * T
            if short: ALd, ARd = b['kL'] + (b['kL'] - hL) * .08, b['kR'] + (b['kR'] - hR) * .08; KLd, KRd = (hL + b['kL']) / 2, (hR + b['kR']) / 2
            else: ALd, ARd = b['aL'] + (b['aL'] - b['kL']) * .12, b['aR'] + (b['aR'] - b['kR']) * .12; KLd, KRd = b['kL'], b['kR']
            add('WL', WLd); add('WR', WRd); add('CR', hm - up * .22 * T); add('AL', ALd); add('AR', ARd); add('KL', KLd); add('KR', KRd)
        elif cat == 'dress':
            side = (sR - sL) / max(np.linalg.norm(sR - sL), 1e-6); kc = (b['kL'] + b['kR']) / 2
            center = kc + (kc - hm) * .2; hw = np.linalg.norm(hL - hR)
            half = max(hw * 1.25, np.linalg.norm(b['kL'] - b['kR']) * .7 + hw * .4)
            add('SL', sLo); add('SR', sRo); add('NC', NCd)
            add('WL', sLo + (hL - sL) * .72); add('WR', sRo + (hR - sR) * .72)
            add('HL', center - side * half); add('HR', center + side * half); add('BC', center)
            addp((A('NC') + A('BC')) / 2, (NCd + center) / 2)
        return np.array(S), np.array(D), info

    def _warp(self, img, pm, S, D, gain):
        H, W = img.shape[:2]; span = max(np.ptp(D[:, 0]), np.ptp(D[:, 1]), 20); mg = .3 * span
        x0, y0 = int(max(0, D[:, 0].min() - mg)), int(max(0, D[:, 1].min() - mg))
        x1, y1 = int(min(W, D[:, 0].max() + mg)), int(min(H, D[:, 1].max() + mg))
        if x1 - x0 < 8 or y1 - y0 < 8: return
        st = 6; gx = np.linspace(x0, x1 - 1, (x1 - x0) // st + 2); gy = np.linspace(y0, y1 - 1, (y1 - y0) // st + 2)
        mx, my = np.meshgrid(gx, gy)
        rbf = RBFInterpolator(D, S, kernel='thin_plate_spline', smoothing=1.0)
        src = rbf(np.c_[mx.ravel(), my.ravel()]).reshape(len(gy), len(gx), 2).astype(np.float32)
        mapx = cv2.resize(src[:, :, 0], (x1 - x0, y1 - y0)); mapy = cv2.resize(src[:, :, 1], (x1 - x0, y1 - y0))
        warped = cv2.remap(pm, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        blend(img, warped, x0, y0, gain)

    def _seg(self, img, part, a0, b0, a1, b1, gain):
        """Rigid (rotate + scale + move) placement of one sleeve segment: src a0->b0 maps to dst a1->b1."""
        z = complex(*(b1 - a1)) / complex(*(b0 - a0)); re, im = z.real, z.imag
        M = np.array([[re, -im, a1[0] - (re * a0[0] - im * a0[1])], [im, re, a1[1] - (im * a0[0] + re * a0[1])]])
        h, w = part.shape[:2]; H, W = img.shape[:2]
        tc = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) @ M[:, :2].T + M[:, 2]
        x0, y0 = int(max(0, tc[:, 0].min())), int(max(0, tc[:, 1].min()))
        x1, y1 = int(min(W, tc[:, 0].max())) + 1, int(min(H, tc[:, 1].max())) + 1
        if x1 <= x0 or y1 <= y0: return
        M[0, 2] -= x0; M[1, 2] -= y0
        blend(img, cv2.warpAffine(part, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0)), x0, y0, gain)

    def _sleeves(self, img, parts, a, b, info, tf, gain):
        c = (b['sL'] + b['sR']) / 2
        for sd, sh, el, wr in (('L', b['sL'], b['eL'], b['wL']), ('R', b['sR'], b['eR'], b['wR'])):
            Q, M_, P = parts['Q' + sd], parts['M' + sd], N(a['P' + sd])
            ln = np.linalg.norm(P - Q) / info['gt'] * info['bt']
            poly = [sh, el, wr, wr + (wr - el)]
            d0, d1, d2 = tf(sh + (sh - c) * .08), tf(poly_point(poly, ln / 2)), tf(poly_point(poly, ln))
            self._seg(img, parts[sd + '1'], Q, M_, d0, d1, gain)
            self._seg(img, parts[sd + '2'], M_, P, d1, d2, gain)

    def _render(self, img, b, items, scale=1.0, dy=0.0, gain=1.0):
        c, hm = (b['sL'] + b['sR']) / 2, (b['hL'] + b['hR']) / 2; T = max(np.linalg.norm(hm - c), 1.)
        sw = np.linalg.norm(b['sL'] - b['sR'])
        for it in sorted(items, key=lambda i: self.ORDER.index(i['category'])):
            asset = self._asset(it)
            if asset is None: continue
            pm, a, parts = asset; cat = it['category']
            if cat == 'hat':
                w = int(b['head_w'] * 1.9 * scale); h = int(w * pm.shape[0] / pm.shape[1])
                if w < 8 or h < 8: continue
                fg = rot(cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA), b['angle'])
                cy = b['eye_y'] - .02 * sw - h * .35 + dy * T
                blend(img, fg, int(b['nose'][0] - fg.shape[1] / 2), int(cy - fg.shape[0] / 2), gain); continue
            try:
                use = parts is not None
                S, D, info = self._points(cat, it['name'], a, b, sleeves=not use)
                ctr = D.mean(0); tf = lambda p: ctr + (N(p) - ctr) * scale + np.array([0, dy * T])
                D = np.array([tf(p) for p in D])
                self._warp(img, parts['torso'] if use else pm, S, D, gain)
                if use: self._sleeves(img, parts, a, b, info, tf, gain)
            except Exception as e:
                print('[warp]', it['name'], e)
        return img

    def process(self, bgr, items, mirror=True, scale=1.0, dy=0.0, height_cm=170.0, smooth=True):
        if mirror: bgr = cv2.flip(bgr, 1)
        out = bgr.copy(); H, W = out.shape[:2]
        if self.pose is None: return self._msg(out, 'MediaPipe not installed'), None
        with self.lock:
            res = self.pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks: self.prev = None; return self._msg(out, 'No body detected - step back'), None
            pts = np.array([[l.x * W, l.y * H, l.visibility] for l in res.pose_landmarks.landmark], float)
            if smooth and self.prev is not None and self.prev_size == (W, H):
                pts[:, :2] = .65 * pts[:, :2] + .35 * self.prev[:, :2]      # temporal smoothing = less jitter
            self.prev, self.prev_size = (pts.copy() if smooth else None), (W, H)
        P = lambda i: pts[i, :2]; V = lambda i: pts[i, 2]
        if min(V(11), V(12), V(23), V(24)) < .4: return self._msg(out, 'Show shoulders and hips'), None
        ids = ((11, 13, 15, 23, 25, 27, 7, 2), (12, 14, 16, 24, 26, 28, 8, 5))
        if P(11)[0] > P(12)[0]: ids = ids[::-1]          # order by image position (left/right on screen)
        b = {'nose': P(0)}; ank_ok = min(V(27), V(28)) > .5
        for nm, t in zip('LR', ids):
            s, h = P(t[0]), P(t[3]); d = h - s
            arm = min(V(t[1]), V(t[2])) > .4
            b['s' + nm], b['h' + nm] = s, h
            b['e' + nm], b['w' + nm] = (P(t[1]), P(t[2])) if arm else (s + d * .5, s + d * .95)
            b['k' + nm] = P(t[4]) if V(t[4]) > .5 else h + d * .9
            b['a' + nm] = P(t[5]) if V(t[5]) > .5 else h + d * 1.8
        sw = np.linalg.norm(b['sL'] - b['sR']); c, hm = (b['sL'] + b['sR']) / 2, (b['hL'] + b['hR']) / 2
        torso = hm[1] - c[1]
        if sw < 15 or np.linalg.norm(hm - c) < 15: return self._msg(out, 'Move closer / stand upright'), None
        e1, e2 = sorted([P(2), P(5)], key=lambda p: p[0])
        b['eye_y'] = (e1[1] + e2[1]) / 2; b['angle'] = math.degrees(math.atan2(e2[1] - e1[1], e2[0] - e1[0]))
        b['head_w'] = max(np.linalg.norm(P(7) - P(8)) if min(V(7), V(8)) > .5 else sw * .55, sw * .4)
        gain = float(np.clip(cv2.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))[0] / 135, .82, 1.15))
        self._render(out, b, items, scale, dy, gain)
        body_px = ((b['aL'][1] + b['aR'][1]) / 2 - P(0)[1]) / .87 if ank_ok else np.linalg.norm(hm - c) / .30
        cm = float(sw / max(body_px, 1) * height_cm)
        size = next((s for t, s in zip((35, 38, 41, 44, 47), ('XS', 'S', 'M', 'L', 'XL')) if cm < t), 'XXL')
        return out, {'shoulder_cm': round(cm, 1), 'size': size}
