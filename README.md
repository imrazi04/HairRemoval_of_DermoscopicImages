# HairRemoval_of_DermoscopicImages
A compact computer‑vision pipeline for removing hair artifacts from dermoscopic images using classic image processing and inpainting.

---

## Status (short)
- Core pipeline implemented and demonstrated in `experiments/experiments.ipynb` (Top‑Hat → Brightening → Flat‑Field Correction → Thresholding → Morphological Cleaning → Iterative Inpainting + Verification).
- Robustness improvements added: diagnostic helpers, an **auto-fix** parameter sweep, multi‑channel processing, edge fusion, and iterative inpainting that escalates radius/method if residual hair persists.
- Engine refactor: `src/remove.py` exposes `remove_hairs_from_rgb(img_rgb, progress_callback=None)` returning `(hair_free_image, final_mask, stats)` and supports progress callbacks for integration.
- Streamlit UI: `app.py` performs in‑memory processing (no disk writes unless explicitly requested), displays before/after, mask preview, and provides a download button.
- CLI helper: `scripts/hair_removal_auto_fix.py` for batch processing and diagnostics.

---

## Quick start
Dependencies can be installed from the included requirements file:

```bash
pip install -r requirements.txt
# or (minimal): pip install streamlit opencv-python numpy pillow
```

Run the interactive UI:

```bash
streamlit run app.py
```

A CLI mode is available for scripted runs:

```bash
python scripts/hair_removal_auto_fix.py --input path/to/image.jpg --output out.jpg
```

The UI keeps the processed images in memory for immediate download; files are not saved to disk by default.

---

## How it works (brief)
Pipeline stages:
1. Load & resize
2. Hair enhancement (black top‑hat per channel)
3. Local brightening
4. Flat‑field correction (optional in auto‑fix)
5. Thresholding (Otsu by default; adaptive/manual available)
6. Morphological cleaning + connected component filtering
7. Iterative inpainting (Telea / Navier‑Stokes; escalate if needed)
8. Selective detail‑preserving enhancement (denoise, CLAHE, light sharpening)

The notebook includes diagnostic visualizations and an **auto‑fix** routine that tries progressively stronger parameter sets until verification passes.

---

## Engine API
The engine is callable from Python:

```python
from src.remove import remove_hairs_from_rgb

hair_free, mask, stats = remove_hairs_from_rgb(img_rgb, progress_callback=my_callback)
```

- `img_rgb`: H×W×3 uint8 RGB numpy array
- `progress_callback(status_string)`: optional; status strings include `enhancement_done`, `thresholding_done`, `cleaning_done`, `inpainting`, `complete`
- Returns:
  - `hair_free`: cleaned RGB image (uint8)
  - `mask`: final binary mask (uint8)
  - `stats`: dict with coverage and quality metrics (initial/final coverage, PSNR, etc.)

---

## Screenshots
Repository assets include UI screenshots used below (files found in `assets/`).

![App main screen](assets/asset01.png)
_App main screen — upload, run, and view results._

![Before vs After](assets/asset02.png)
_Before/after comparison and mask._

![Mask preview](assets/asset03.png)
_Detected hair mask._

![Processing progress GIF](assets/asset04.png)
_Processing progress and status indicators (short animation recommended as GIF)._ 

---

## Adding images to the README
Images can be placed in a repository folder (for example `assets/` or `docs/screenshots/`) and referenced via relative paths:

```markdown
![caption](assets/asset01.png)
```

Notes:
- Use PNG for screenshots and GIF for short animations.
- Keep images reasonably sized so GitHub renders them quickly.
- When editing README on GitHub, images can be dragged into the editor — GitHub uploads them and inserts the Markdown link automatically.

---

## Tests & next steps
- Diagnostics and auto‑fix functionality are implemented and available in the notebook and CLI.
- Planned: add a diagnostics panel to the UI (intermediate images and aggressive auto‑fix toggle), add automated tests for key pipeline functions, and perform a final QA sweep across the dataset.

---

## Key files
- `src/remove.py` — core, in‑memory engine
- `experiments/experiments.ipynb` — exploratory notebooks and diagnostics
- `scripts/hair_removal_auto_fix.py` — CLI batch/diagnostics
- `app.py` — Streamlit UI

---

## Contributing
Issues and pull requests are welcome. Include tests or example images when proposing algorithmic changes.

---

## License
See `LICENSE`.

---
