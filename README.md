# Virtual AI Trial Room – An Intelligent Clothing Try-On System

## Run (Windows / macOS / Linux)
1. Install **Python 3.10, 3.11 or 3.12** (mediapipe 0.10.14 does not support 3.13+).
2. Open a terminal in this folder and run:
```
python -m venv venv
venv\Scripts\activate          # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
3. Open **http://127.0.0.1:5000** in Chrome/Edge (use 127.0.0.1 or localhost so the webcam is allowed).
4. Demo login: `demo@trialroom.com` / `demo123` (or register a new account).

The database, garment PNG images and the ML size model are created automatically on first run.

## Project structure
| File | Purpose (architecture block) |
|---|---|
| app.py | Flask backend: routing, auth, cart/orders, recommendation/size/try-on APIs |
| tryon.py | MediaPipe Pose + OpenCV overlay engine (AI / Computer Vision module) |
| recommender.py | Outfit engine: colour matching, occasion, style, budget, history (ML/Recommendation module) |
| size_model.py | RandomForest size prediction (XS-XXL) |
| assets.py | Product data + auto-generated transparent clothing PNGs |
| templates/, static/ | Frontend (Bootstrap, HTML, CSS, JavaScript) |
| trial_room.db | SQLite (created automatically): users, products, cart, orders, wishlist, saved outfits |

## Using your own clothing images
Replace `static/clothes/<product id>.png` with transparent PNGs (front view, tightly cropped).
Add products by editing the `PRODUCTS` list in `assets.py` and deleting `trial_room.db`.

## Tips for good try-on
Good lighting, stand 1.5-2 m from the camera so shoulders, hips and legs are visible. Use the *size* and *vertical shift* sliders to fine-tune.

## v2 changes
- Garments are warped to the body: sleeves rotate/bend with your arms (arms-up works), pants follow legs, dresses flare with knees; landmark smoothing reduces jitter.
- **+ Add Product** page: upload your own product photos (flat-lay / ghost-mannequin, plain background). Background is removed automatically (optional AI: `pip install rembg onnxruntime`, then tick the box).
- Catalog category tabs (Tops / Bottoms / Dresses / Jackets / Caps) and a category filter inside the Try-On page.
- More fabric-like built-in garment images (auto-regenerated on first start).
