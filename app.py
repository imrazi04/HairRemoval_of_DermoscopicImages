import io
import os
from PIL import Image
import numpy as np
import streamlit as st

from src.remove import remove_hairs_from_rgb

st.set_page_config(page_title='Dermoscopic Hair Removal', layout='centered')

st.markdown("""
# 🩺 Dermoscopic Hair Removal
Upload a dermoscopic image and remove hair artifacts using the existing pipeline.
- Uses the project's original algorithm (Top-hat → Brightening → FFC → Otsu → Clean → Inpaint)
- No images are stored on disk during processing (in-memory only)
- You can download the clean image after processing
""")

# Sidebar: sample images
st.sidebar.header("Options")
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
examples = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))] if os.path.exists(DATA_DIR) else []
example_choice = st.sidebar.selectbox('Choose example image (optional)', ['-- none --'] + examples)

uploaded_file = st.file_uploader('Upload dermoscopic image (jpg, png)', type=['jpg', 'jpeg', 'png'])

img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
elif example_choice != '-- none --':
    img = Image.open(os.path.join(DATA_DIR, example_choice)).convert('RGB')

if img is not None:
    st.subheader('Original Image')
    st.image(img, use_column_width=True)

    cols = st.columns([1, 1])

    with cols[0]:
        st.markdown('---')
        st.write('Processing controls')
        run_button = st.button('Remove Hairs', key='run')
        st.write('Tip: Use small to medium resolution images for faster processing.')

    with cols[1]:
        st.markdown('---')
        st.write('Download')

    if run_button:
        # placeholders
        status = st.empty()
        progress = st.progress(0)

        # mapping statuses to progress
        status_map = {
            'enhancement_done': (10, 'Enhancement done'),
            'thresholding_done': (30, 'Thresholding done'),
            'cleaning_done': (50, 'Cleaning done'),
            'inpainting': (75, 'Inpainting...'),
            'selective_enhancement': (90, 'Final enhancement...'),
            'complete': (100, 'Complete')
        }

        def callback(s):
            p, msg = status_map.get(s, (0, s))
            status.text(msg)
            progress.progress(p)

        # Convert PIL to numpy RGB
        img_np = np.array(img)

        with st.spinner('Running hair removal pipeline...'):
            hair_free, mask, stats = remove_hairs_from_rgb(img_np, progress_callback=callback)

        status.text('Complete')
        progress.progress(100)

        # Show results
        st.subheader('Results')
        c1, c2 = st.columns(2)
        c1.image(img_np, caption='Original', use_column_width=True)
        c2.image(hair_free, caption='Hair-Removed', use_column_width=True)

        # Mask preview
        st.subheader('Detected Hair Mask')
        st.image(mask, caption='Hair mask (white=hair)', use_column_width=True, clamp=True)

        # Stats & download
        st.write('**Summary**')
        st.write(f"Initial hair coverage (estimated): {stats['initial_hair_coverage']:.2f}%")
        st.write(f"Final hair coverage: {stats['final_hair_coverage']:.2f}%")
        st.write(f"PSNR: {stats['psnr']:.2f} dB")

        # Prepare download (in-memory, no disk writes)
        out_pil = Image.fromarray(hair_free)
        buf = io.BytesIO()
        out_pil.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        st.download_button('Download clean image (JPEG)', data=byte_im, file_name='hair_free.jpg', mime='image/jpeg')

else:
    st.info('Upload an image or choose an example to get started.')
