# HairRemoval_of_DermoscopicImages
A compact, CV-based pipeline for removing hair artifacts from dermoscopic images using classic image processing and inpainting techniques.

---

## 📌 Project status (short)
- Core hair-removal pipeline implemented and tested in `experiments/experiments.ipynb` (Top‑Hat → Brightening → Flat‑Field Correction → Thresholding → Morphological Cleaning → Iterative Inpainting + Verification).
- Robustness improvements added: diagnostic helpers, an **auto-fix** parameter sweep, multi-channel processing, edge fusion, and iterative inpainting that escalates radius/method if residual hair persists.
- Engine refactor: `src/remove.py` exposes `remove_hairs_from_rgb(img_rgb, progress_callback=None)` which returns `(hair_free_image, final_mask, stats)` and supports progress callbacks (for UI integration).
- Streamlit UI: `app.py` (in-memory processing, no disk writes by default) — upload, run, view before/after, preview mask, and download the clean image.
- CLI helper: `scripts/hair_removal_auto_fix.py` for batch runs and diagnostic dumps.

---

## 🚀 Quick start
1. Install dependencies (recommended from the repo environment):

```bash
pip install -r requirements.txt
# or minimal:
# pip install streamlit opencv-python numpy pillow
```

2. Run the Streamlit UI (recommended for interactive use):

```bash
streamlit run app.py
```

3. Use the CLI for batch/diagnostic runs:

```bash
python scripts/hair_removal_auto_fix.py --input path/to/image.jpg --output out.jpg
```

Note: The Streamlit UI calls `remove_hairs_from_rgb` and keeps processed images in memory for immediate download (no files are left on disk unless you explicitly save them).

---

## 🧠 How it works (high level)
Stages in the pipeline:
1. Load & resize
2. Hair enhancement (Black top-hat on each channel)
3. Local brightening
4. Flat‑field correction (optional; can be reduced/skipped in the auto-fix)
5. Thresholding (Otsu by default; adaptive/manual available in auto-fix)
6. Morphological cleaning + connected-component filtering
7. Iterative inpainting (Telea / Navier‑Stokes; escalate radius / method if hair persists)
8. Selective detail-preserving enhancement (denoising, CLAHE, light sharpening)

The notebook `experiments/experiments.ipynb` contains diagnostic visualizations and the **auto-fix** runner that tries progressively stronger parameters until the mask/inpainting verification is satisfactory.

---

## 🧩 Engine API
Use the engine programmatically:

```python
from src.remove import remove_hairs_from_rgb

hair_free, mask, stats = remove_hairs_from_rgb(img_rgb, progress_callback=my_callback)
```

- `img_rgb`: H×W×3 uint8 RGB image (numpy array)
- `progress_callback(status_string)`: optional; used by the Streamlit UI to show progress (strings include `enhancement_done`, `thresholding_done`, `cleaning_done`, `inpainting`, `complete`)
- Returns:
  - `hair_free`: cleaned RGB image (uint8)
  - `mask`: final binary mask (uint8) where white pixels indicate detected hair
  - `stats`: dictionary with performance metrics (initial/final coverage, PSNR, etc.)

---

## 🖼️ Adding UI screenshots or GIFs to this README
To include images or animated GIFs (recommended for showing the processing animation), do the following:

1. Create a folder for visuals, e.g., `docs/screenshots` or `assets/` and add images there (commit them to the repo).
2. Reference them in Markdown with relative paths. Example:

```markdown
![Streamlit UI screenshot](docs/screenshots/ui.png)
![Processing demo GIF](docs/screenshots/demo.gif)
```

Tips:
- Use PNG for screenshots and GIF for short animated demos of progress/animation.
- Keep files small (resize or compress) so GitHub renders them quickly.
- You can drag-and-drop images into GitHub when editing README on the website — GitHub will upload and insert the right Markdown link.

---

## ✅ Tests, QA & next steps
- Current: Diagnostic routines and auto-fix candidates implemented; the engine is callable and integrated into Streamlit for visual testing.
- Next: Add a diagnostics panel to the UI (optional) to reveal intermediate images (Top‑Hat, enhanced, threshold), add unit/integration tests for the core engine, and run a final QA sweep across the dataset to ensure the “no residual hair” requirement is met.

---

## 📂 Important files
- `src/remove.py` — main in-memory engine (callable)
- `experiments/experiments.ipynb` — exploratory notebook with diagnostics and parameter sweeps
- `scripts/hair_removal_auto_fix.py` — CLI for batch processing / diagnostics
- `app.py` — Streamlit UI

---

## 💬 Contributing
- Open issues for failing images or to propose algorithm improvements.
- PRs welcome — please include tests or examples demonstrating improvements.

---

## 📜 License
See `LICENSE`.

---

If you'd like, I can add a small **diagnostics toggle** to the Streamlit UI next (showing Top-Hat, enhanced, Otsu mask and intermediate stats). Want me to add that now?