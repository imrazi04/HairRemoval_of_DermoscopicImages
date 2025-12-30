# HairRemoval_of_DermoscopicImages
A computer vision–based project for automatic hair removal from dermoscopic images using image processing filters and inpainting techniques. It provides an interactive UI where users can upload images, view before–after comparisons, and download the hair-removed results for further analysis.

## Streamlit App (UI)
A simple Streamlit-based user interface is provided in `app.py`.

Requirements:
- Python 3.8+
- Install dependencies: `pip install streamlit opencv-python numpy pillow`

Run:

```
streamlit run app.py
```

Notes:
- The UI calls the in-memory engine `remove_hairs_from_rgb` (no processed images are saved to disk by default).
- Upload an image or pick a sample from `data/`, click **Remove Hairs**, and download the cleaned image directly from the browser.
