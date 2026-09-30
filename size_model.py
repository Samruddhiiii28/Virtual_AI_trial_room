"""ML size prediction (RandomForest). Trained automatically on synthetic measurement data on first use."""
import os
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier

SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'size_model.joblib')
_model = None

def _label(h, w, g):
    w_adj = w if g == 1 else w - 6
    idx = sum(w_adj >= t for t in (50, 60, 72, 84, 98))
    idx += 1 if h > 183 else (-1 if h < 155 else 0)
    return int(np.clip(idx, 0, 5))

def train():
    rng = np.random.default_rng(42); n = 5000
    h = np.clip(rng.normal(168, 10, n), 145, 200)
    bmi = np.clip(rng.normal(23.5, 4, n), 15, 40)
    w = bmi * (h / 100) ** 2; g = rng.integers(0, 2, n)
    y = [_label(a, b, c) for a, b, c in zip(h, w, g)]
    m = RandomForestClassifier(n_estimators=120, random_state=1).fit(np.c_[h, w, g], y)
    joblib.dump(m, PATH); return m

def predict_size(height, weight, gender=1):
    """gender: 1 = male, 0 = female"""
    global _model
    if _model is None:
        _model = joblib.load(PATH) if os.path.exists(PATH) else train()
    x = np.array([[height, weight, gender]])
    p = _model.predict_proba(x)[0]
    order = np.argsort(p)[::-1]
    cls = _model.classes_
    best = SIZES[int(cls[order[0]])]
    alt = SIZES[int(cls[order[1]])] if len(order) > 1 and p[order[1]] > 0.05 else None
    return {'size': best, 'confidence': round(float(p[order[0]]) * 100), 'alternative': alt}
