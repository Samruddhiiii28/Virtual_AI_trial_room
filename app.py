"""Virtual AI Trial Room - Flask backend (routing, auth, products, cart, recommendation, size & try-on APIs)."""
import os, base64, sqlite3, datetime
from functools import wraps
import cv2, numpy as np
from flask import Flask, g, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from assets import ensure_assets, PRODUCTS, COLORS
from size_model import predict_size
from tryon import TryOnEngine
from bgremove import remove_background
from recommender import recommend_outfits, best_picks

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, 'trial_room.db')
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')
STYLES = ['classic', 'minimal', 'streetwear', 'sporty', 'elegant']
OCCASIONS = ['casual', 'formal', 'party', 'sports']

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, password TEXT,
  gender TEXT DEFAULT 'male', height REAL DEFAULT 170, weight REAL DEFAULT 65, style TEXT DEFAULT 'classic',
  color TEXT DEFAULT 'black', occasion TEXT DEFAULT 'casual', budget REAL DEFAULT 5000);
CREATE TABLE IF NOT EXISTS products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, color TEXT, style TEXT,
  occasion TEXT, price REAL, gender TEXT, image TEXT, stock INTEGER DEFAULT 50, description TEXT);
CREATE TABLE IF NOT EXISTS cart(id INTEGER PRIMARY KEY, user_id INT, product_id INT, size TEXT, qty INT);
CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, user_id INT, total REAL, created TEXT, address TEXT);
CREATE TABLE IF NOT EXISTS order_items(id INTEGER PRIMARY KEY, order_id INT, product_id INT, size TEXT, qty INT, price REAL);
CREATE TABLE IF NOT EXISTS saved_outfits(id INTEGER PRIMARY KEY, user_id INT, name TEXT, product_ids TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS wishlist(user_id INT, product_id INT, UNIQUE(user_id, product_id));
CREATE TABLE IF NOT EXISTS views(user_id INT, product_id INT);
"""

def init_db():
    ensure_assets()
    con = sqlite3.connect(DB); con.executescript(SCHEMA)
    if not con.execute('SELECT COUNT(*) FROM products').fetchone()[0]:
        for i, (n, c, col, st, oc, pr, ge, sh) in enumerate(PRODUCTS, 1):
            con.execute('INSERT INTO products VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                        (i, n, c, col, st, oc, pr, ge, f'clothes/{i}.png', 50, f'{n}: a {st} {col} {c} suited for {oc} wear.'))
        con.execute('INSERT INTO users(name,email,password,gender,height,weight) VALUES(?,?,?,?,?,?)',
                    ('Demo User', 'demo@trialroom.com', generate_password_hash('demo123'), 'female', 165, 58))
    con.commit(); con.close()

def db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db

def q(sql, args=(), one=False):
    rows = [dict(r) for r in db().execute(sql, args).fetchall()]
    return (rows[0] if rows else None) if one else rows

def ex(sql, args=()):
    cur = db().execute(sql, args); db().commit(); return cur.lastrowid

@app.teardown_appcontext
def close_db(e=None):
    d = g.pop('db', None)
    if d: d.close()

def cur_user():
    if 'me' not in g:
        g.me = q('SELECT * FROM users WHERE id=?', (session['uid'],), one=True) if 'uid' in session else None
    return g.me

def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if not cur_user():
            if request.is_json: return jsonify(ok=False, msg='Please log in first'), 401
            flash('Please log in first.', 'warning'); return redirect(url_for('login', next=request.path))
        return f(*a, **k)
    return w

@app.context_processor
def inject():
    u = cur_user(); n = 0
    if u: n = q('SELECT COALESCE(SUM(qty),0) n FROM cart WHERE user_id=?', (u['id'],), one=True)['n']
    return dict(me=u, cart_count=n, STYLES=STYLES, OCCASIONS=OCCASIONS, COLOR_NAMES=list(COLORS))

def gnum(gender): return 1 if gender == 'male' else 0
def my_size(u): return predict_size(u['height'], u['weight'], gnum(u['gender']))['size'] if u else 'M'
def ids_of(v): return [int(x) for x in str(v).replace('[', '').replace(']', '').split(',') if x.strip().isdigit()]
def payload(): return request.get_json(silent=True) or request.form.to_dict(flat=True)
def reply(msg, ok=True, to='cart'):
    if request.is_json: return jsonify(ok=ok, msg=msg), (200 if ok else 400)
    flash(msg, 'success' if ok else 'danger'); return redirect(request.referrer or url_for(to))

def history(u):
    if not u: return {}
    rows = q('SELECT p.style s, COUNT(*) c FROM products p JOIN (SELECT product_id FROM views WHERE user_id=? UNION ALL '
             'SELECT product_id FROM wishlist WHERE user_id=? UNION ALL SELECT product_id FROM cart WHERE user_id=?) h '
             'ON h.product_id=p.id GROUP BY p.style', (u['id'],) * 3)
    return {r['s']: r['c'] for r in rows}

def add_to_cart(uid, pid, size, qty=1):
    r = q('SELECT id FROM cart WHERE user_id=? AND product_id=? AND size=?', (uid, pid, size), one=True)
    if r: ex('UPDATE cart SET qty=qty+? WHERE id=?', (qty, r['id']))
    else: ex('INSERT INTO cart(user_id,product_id,size,qty) VALUES(?,?,?,?)', (uid, pid, size, qty))

# ---------------- pages ----------------
@app.route('/')
def index():
    return render_template('index.html', featured=q('SELECT * FROM products ORDER BY RANDOM() LIMIT 8'))

@app.route('/catalog')
def catalog():
    a = request.args; sql, args = 'SELECT * FROM products WHERE 1=1', []
    for f in ('category', 'occasion', 'color', 'style'):
        if a.get(f): sql += f' AND {f}=?'; args.append(a[f])
    if a.get('q'): sql += ' AND name LIKE ?'; args.append(f"%{a['q']}%")
    if a.get('max_price'): sql += ' AND price<=?'; args.append(float(a['max_price']))
    return render_template('catalog.html', products=q(sql + ' ORDER BY id', args), a=a)

@app.route('/product/<int:pid>')
def product(pid):
    p = q('SELECT * FROM products WHERE id=?', (pid,), one=True)
    if not p: return redirect(url_for('catalog'))
    u = cur_user(); wished = False
    if u:
        ex('INSERT INTO views VALUES(?,?)', (u['id'], pid))
        wished = bool(q('SELECT 1 FROM wishlist WHERE user_id=? AND product_id=?', (u['id'], pid)))
    sim = q('SELECT * FROM products WHERE (occasion=? OR style=?) AND id!=? LIMIT 4', (p['occasion'], p['style'], pid))
    return render_template('product.html', p=p, similar=sim, wished=wished, rec_size=my_size(u))

@app.route('/tryon')
def tryon():
    u = cur_user()
    return render_template('tryon.html', products=q('SELECT * FROM products ORDER BY category,id'),
                           initial=ids_of(request.args.get('ids', '')), user=u)

@app.route('/recommend')
def recommend():
    u, a = cur_user(), request.args
    d = dict(gender=(u or {}).get('gender', 'female'), occasion=(u or {}).get('occasion', 'casual'),
             style=(u or {}).get('style', 'classic'), color=(u or {}).get('color', 'black'),
             budget=(u or {}).get('budget', 5000), height=(u or {}).get('height', 165), weight=(u or {}).get('weight', 60))
    pr = {k: a.get(k) or v for k, v in d.items()}
    outfits = picks = size = None; over = False
    if a.get('go'):
        prods, hist = q('SELECT * FROM products'), history(u)
        outfits, over = recommend_outfits(prods, pr, hist)
        picks = best_picks(prods, pr, hist)
        size = predict_size(float(pr['height']), float(pr['weight']), gnum(pr['gender']))
    return render_template('recommend.html', pr=pr, outfits=outfits, picks=picks, size=size, over=over)

# ---------------- auth ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        f = request.form
        if q('SELECT 1 FROM users WHERE email=?', (f['email'].lower(),)):
            flash('Email already registered.', 'danger')
        else:
            uid = ex('INSERT INTO users(name,email,password,gender,height,weight) VALUES(?,?,?,?,?,?)',
                     (f['name'], f['email'].lower(), generate_password_hash(f['password']), f['gender'],
                      float(f['height'] or 170), float(f['weight'] or 65)))
            session['uid'] = uid; flash('Welcome! Set your style preferences in your profile.', 'success')
            return redirect(url_for('profile'))
    return render_template('auth.html', mode='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = q('SELECT * FROM users WHERE email=?', (request.form['email'].lower(),), one=True)
        if u and check_password_hash(u['password'], request.form['password']):
            session['uid'] = u['id']; return redirect(request.args.get('next') or url_for('index'))
        flash('Invalid email or password.', 'danger')
    return render_template('auth.html', mode='login')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    u = cur_user()
    if request.method == 'POST':
        f = request.form
        ex('UPDATE users SET name=?,gender=?,height=?,weight=?,style=?,color=?,occasion=?,budget=? WHERE id=?',
           (f['name'], f['gender'], float(f['height']), float(f['weight']), f['style'], f['color'], f['occasion'],
            float(f['budget']), u['id']))
        flash('Profile updated.', 'success'); return redirect(url_for('profile'))
    outfits = q('SELECT * FROM saved_outfits WHERE user_id=? ORDER BY id DESC', (u['id'],))
    for o in outfits: o['ids'] = ids_of(o['product_ids'])
    orders = q('SELECT * FROM orders WHERE user_id=? ORDER BY id DESC', (u['id'],))
    for o in orders:
        o['items'] = q('SELECT p.name, oi.size, oi.qty, oi.price FROM order_items oi JOIN products p ON p.id=oi.product_id WHERE order_id=?', (o['id'],))
    wish = q('SELECT p.* FROM products p JOIN wishlist w ON w.product_id=p.id WHERE w.user_id=?', (u['id'],))
    return render_template('profile.html', outfits=outfits, orders=orders, wish=wish, size=my_size(u))

# ---------------- wishlist / cart / outfits ----------------
@app.post('/wishlist/<int:pid>')
@login_required
def wishlist(pid):
    u = cur_user()
    if q('SELECT 1 FROM wishlist WHERE user_id=? AND product_id=?', (u['id'], pid)):
        ex('DELETE FROM wishlist WHERE user_id=? AND product_id=?', (u['id'], pid))
    else: ex('INSERT INTO wishlist VALUES(?,?)', (u['id'], pid))
    return redirect(request.referrer or url_for('catalog'))

@app.post('/cart/add')
@login_required
def cart_add():
    p = payload(); pid = int(p['product_id'])
    add_to_cart(cur_user()['id'], pid, p.get('size') or my_size(cur_user()), int(p.get('qty', 1)))
    return reply('Added to cart.', to='catalog')

@app.post('/outfit/cart')
@login_required
def outfit_cart():
    u = cur_user(); size = my_size(u)
    for pid in ids_of(payload().get('ids', '')): add_to_cart(u['id'], pid, size)
    if request.is_json: return jsonify(ok=True, msg='Outfit added to cart.')
    flash('Outfit added to cart.', 'success'); return redirect(url_for('cart'))

@app.post('/outfit/save')
@login_required
def outfit_save():
    p = payload(); ids = ids_of(p.get('ids', ''))
    if not ids: return reply('Select at least one item first.', False)
    ex('INSERT INTO saved_outfits(user_id,name,product_ids,created) VALUES(?,?,?,?)',
       (cur_user()['id'], p.get('name') or 'My outfit', ','.join(map(str, ids)), datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    return reply('Outfit saved to your profile.', to='profile')

@app.route('/cart')
@login_required
def cart():
    items = q('SELECT c.id cid, c.size, c.qty, p.* FROM cart c JOIN products p ON p.id=c.product_id WHERE c.user_id=?', (cur_user()['id'],))
    return render_template('cart.html', items=items, total=sum(i['price'] * i['qty'] for i in items))

@app.post('/cart/remove/<int:cid>')
@login_required
def cart_remove(cid):
    ex('DELETE FROM cart WHERE id=? AND user_id=?', (cid, cur_user()['id'])); return redirect(url_for('cart'))

@app.post('/checkout')
@login_required
def checkout():
    u = cur_user()
    items = q('SELECT c.size, c.qty, p.id pid, p.price FROM cart c JOIN products p ON p.id=c.product_id WHERE c.user_id=?', (u['id'],))
    if not items: flash('Your cart is empty.', 'warning'); return redirect(url_for('cart'))
    oid = ex('INSERT INTO orders(user_id,total,created,address) VALUES(?,?,?,?)',
             (u['id'], sum(i['price'] * i['qty'] for i in items), datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), request.form.get('address', '')))
    for i in items:
        ex('INSERT INTO order_items(order_id,product_id,size,qty,price) VALUES(?,?,?,?,?)', (oid, i['pid'], i['size'], i['qty'], i['price']))
        ex('UPDATE products SET stock=MAX(stock-?,0) WHERE id=?', (i['qty'], i['pid']))
    ex('DELETE FROM cart WHERE user_id=?', (u['id'],))
    flash(f'Order #{oid} placed successfully (demo checkout).', 'success'); return redirect(url_for('profile'))

@app.route('/add-product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        f, file = request.form, request.files.get('photo')
        if not file or not file.filename: flash('Choose a photo.', 'danger'); return redirect(request.url)
        try: png = remove_background(file.read(), use_ai=bool(f.get('ai')))
        except Exception as e: flash(f'Could not process image: {e}', 'danger'); return redirect(request.url)
        pid = ex('INSERT INTO products(name,category,color,style,occasion,price,gender,image,stock,description) VALUES(?,?,?,?,?,?,?,?,?,?)',
                 (f['name'], f['category'], f['color'], f['style'], f['occasion'], float(f['price']), f['gender'], '', 50, f.get('description') or f['name']))
        open(os.path.join(BASE, 'static', 'clothes', f'{pid}.png'), 'wb').write(png)
        ex('UPDATE products SET image=? WHERE id=?', (f'clothes/{pid}.png', pid))
        if engine: engine.cache.clear()
        flash('Product added - try it on!', 'success'); return redirect(url_for('tryon', ids=pid))
    return render_template('add_product.html')

# ---------------- APIs ----------------
@app.post('/api/size')
def api_size():
    p = request.get_json(force=True)
    return jsonify(predict_size(float(p['height']), float(p['weight']), gnum(p.get('gender', 'male'))))

engine = None
@app.post('/api/tryon')
def api_tryon():
    global engine
    if engine is None: engine = TryOnEngine(os.path.join(BASE, 'static', 'clothes'))
    p = request.get_json(force=True)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(p['image'].split(',', 1)[-1]), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return jsonify(error='bad image'), 400
    h, w = img.shape[:2]
    if max(h, w) > 900: img = cv2.resize(img, (int(w * 900 / max(h, w)), int(h * 900 / max(h, w))))
    ids = ids_of(p.get('ids', []))
    items = q(f"SELECT * FROM products WHERE id IN ({','.join('?' * len(ids))})", ids) if ids else []
    out, meas = engine.process(img, items, bool(p.get('mirror', True)), float(p.get('scale', 1)),
                               float(p.get('dy', 0)), float(p.get('height', 170)), smooth=bool(p.get('mirror', True)))
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return jsonify(image='data:image/jpeg;base64,' + base64.b64encode(buf).decode(), detected=meas is not None, measurements=meas)

init_db()
if __name__ == '__main__':
    predict_size(170, 65, 1)  # trains / loads size model at start-up
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
