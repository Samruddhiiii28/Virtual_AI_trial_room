"""Outfit recommendation: occasion / style / colour compatibility / budget / personal history."""
from itertools import combinations

NEUTRAL = {'black', 'white', 'grey', 'beige', 'navy'}
PAIRS = {frozenset(p) for p in [('blue', 'beige'), ('navy', 'white'), ('olive', 'beige'), ('red', 'black'), ('red', 'white'),
         ('maroon', 'beige'), ('pink', 'grey'), ('brown', 'beige'), ('olive', 'black'), ('blue', 'white'),
         ('blue', 'grey'), ('pink', 'navy'), ('maroon', 'grey'), ('brown', 'white'), ('blue', 'brown')]}

def color_score(a, b):
    if a == b: return 0.5
    if frozenset((a, b)) in PAIRS: return 3
    if a in NEUTRAL or b in NEUTRAL: return 2
    return 0

def item_score(p, pr, hist):
    s = 0
    if p['occasion'] == pr.get('occasion'): s += 3
    if p['style'] == pr.get('style'): s += 2
    if p['color'] == pr.get('color'): s += 1.5
    return s + 0.5 * min(2, hist.get(p['style'], 0))

def _total(items): return sum(i['price'] for i in items)

def _pool(products, pr):
    g = pr.get('gender', 'male')
    return [p for p in products if p['gender'] in (g, 'unisex') and p['stock'] > 0]

def recommend_outfits(products, pr, hist, limit=6):
    budget = float(pr.get('budget') or 0)
    ps = _pool(products, pr)
    by = lambda c: [p for p in ps if p['category'] == c]
    tops, bots, dresses, jackets, hats = by('top'), by('bottom'), by('dress'), by('jacket'), by('hat')
    bases = [[t, b] for t in tops for b in bots] + [[d] for d in dresses]
    outs = []
    for relax in (False, True):
        for base in bases:
            if budget and not relax and _total(base) > budget: continue
            items = list(base)
            for extra in (jackets, hats):
                if not extra: continue
                e = max(extra, key=lambda e: item_score(e, pr, hist) + color_score(e['color'], base[0]['color']))
                if item_score(e, pr, hist) >= 3 and (not budget or relax or _total(items) + e['price'] <= budget):
                    items.append(e)
            avg_item = sum(item_score(i, pr, hist) for i in items) / len(items)
            pair = sum(color_score(a['color'], b['color']) for a, b in combinations(items, 2)) / max(1, len(items) * (len(items) - 1) / 2)
            score = avg_item + 2 * pair
            reasons = []
            if all(i['occasion'] == pr.get('occasion') for i in items[:2]): reasons.append(f"Great for {pr.get('occasion')} occasions")
            if any(i['style'] == pr.get('style') for i in items): reasons.append(f"{pr.get('style')} style match")
            if len(items) > 1 and pair >= 2: reasons.append(f"{items[0]['color']} + {items[1]['color']} colours work well together")
            if budget and _total(items) <= budget: reasons.append("Within your budget")
            outs.append(dict(items=items, ids=','.join(str(i['id']) for i in items), total=_total(items),
                             score=score, pct=min(99, int(100 * score / 13.5)), reasons=reasons or ["Balanced combination"]))
        if outs: break
    outs.sort(key=lambda o: -o['score'])
    chosen, seen = [], {}
    for o in outs:
        k = o['items'][0]['id']
        if seen.get(k, 0) >= 2: continue
        seen[k] = seen.get(k, 0) + 1; chosen.append(o)
        if len(chosen) == limit: break
    return chosen, bool(budget and chosen and chosen[0]['total'] > budget)

def best_picks(products, pr, hist, n=4):
    budget = float(pr.get('budget') or 0)
    ps = [p for p in _pool(products, pr) if not budget or p['price'] <= budget]
    return sorted(ps, key=lambda p: -item_score(p, pr, hist))[:n]
