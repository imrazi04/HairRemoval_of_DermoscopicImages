import cv2
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================
# CONFIGURABLE PARAMETERS
# ============================
IMAGE_PATH = '../data/BCC_2.jpg'
OUTPUT_PATH = '../data/BCC_1_hair_removed_careful.jpg'
TARGET_SIZE = (720, 720)

# Enhancement
TOPHAT_RADIUS = 18
BRIGHTENING_FACTOR = 0.7

# Flat Field Correction
FFC_SIGMA = 30

# Thresholding
USE_OTSU = True
ADAPTIVE_BLOCK_SIZE = 35

# Multi-channel processing
USE_MULTI_CHANNEL = True

# Inpainting
INPAINTING_RADIUS = 3
USE_CAREFUL_INPAINTING = True

# Quality preservation
PRESERVE_DETAILS = True


# ============================
# STAGE 1–4: CHANNEL PROCESSING
# ============================
def process_channel(channel, channel_name):
    print(f"\n--- Processing {channel_name} Channel ---")

    # Stage 2: Black Top-Hat
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * TOPHAT_RADIUS + 1, 2 * TOPHAT_RADIUS + 1)
    )
    closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
    tophat = cv2.subtract(closed, channel)

    potential_hair = np.sum(tophat > 5) / tophat.size * 100
    print(f"Top-Hat non-zero (>5) coverage: {potential_hair:.2f}%")

    # Stage 3: Local Brightening
    brightened = tophat.astype(np.float32)
    brightened += (255 - brightened) * BRIGHTENING_FACTOR
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)

    # Stage 4: Flat-Field Correction
    img_float = brightened.astype(np.float32)
    offset = cv2.GaussianBlur(img_float, (0, 0), FFC_SIGMA)
    mean_offset = np.mean(offset)

    ffc = (img_float / (offset + 1e-6)) * mean_offset
    ffc = np.clip(ffc, 0, 255).astype(np.uint8)

    print(f"FFC mean intensity: {ffc.mean():.2f}")

    return ffc


# ============================
# MAIN PIPELINE
# ============================
def remove_hairs_from_rgb(img_rgb, progress_callback=None):
    """Engine function: run full pipeline on an RGB numpy image and return (image, mask, stats)."""
    # Resize
    img_resized = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # STAGE 1–4: Hair Enhancement
    if USE_MULTI_CHANNEL:
        channels = [
            (img_resized[:, :, 0], "Red"),
            (img_resized[:, :, 1], "Green"),
            (img_resized[:, :, 2], "Blue")
        ]
        processed = [process_channel(ch, name) for ch, name in channels]
        enhanced = np.maximum.reduce(processed)
    else:
        enhanced = process_channel(img_resized[:, :, 0], "Red")

    if progress_callback is not None:
        progress_callback('enhancement_done')

    # STAGE 5: Thresholding
    if USE_OTSU:
        thresh_val, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, 0)

    if progress_callback is not None:
        progress_callback('thresholding_done')

    # STAGE 6: Morphological Cleaning
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.dilate(clean_mask, dilate_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    final_mask = np.zeros_like(clean_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 10:
            final_mask[labels == i] = 255

    if progress_callback is not None:
        progress_callback('cleaning_done')

    # STAGE 7: Inpainting
    inpaint_mask = (final_mask > 0).astype(np.uint8)
    if USE_CAREFUL_INPAINTING:
        hair_free_1 = cv2.inpaint(img_resized, inpaint_mask, 2, cv2.INPAINT_TELEA)
        hair_free_2 = cv2.inpaint(img_resized, inpaint_mask, 3, cv2.INPAINT_NS)
        hair_free = cv2.addWeighted(hair_free_1, 0.5, hair_free_2, 0.5, 0)
        mask_3ch = cv2.cvtColor(inpaint_mask, cv2.COLOR_GRAY2RGB)
        hair_free_final = np.where(mask_3ch == 1, hair_free, img_resized)
    else:
        hair_free_final = cv2.inpaint(img_resized, inpaint_mask, INPAINTING_RADIUS, cv2.INPAINT_TELEA)

    if progress_callback is not None:
        progress_callback('inpainting')

    # STAGE 8: Selective Enhancement
    enhanced_img = hair_free_final.copy()

    if PRESERVE_DETAILS:
        smooth_mask = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=1)
        denoised = cv2.fastNlMeansDenoisingColored(hair_free_final, None, 3, 3, 7, 21)
        smooth_mask_3ch = cv2.cvtColor(smooth_mask, cv2.COLOR_GRAY2RGB) / 255.0
        enhanced_img = (hair_free_final * (1 - smooth_mask_3ch) + denoised * smooth_mask_3ch).astype(np.uint8)

    lab = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    hair_free_final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    gaussian = cv2.GaussianBlur(hair_free_final, (0, 0), 1.0)
    hair_free_final = cv2.addWeighted(hair_free_final, 1.3, gaussian, -0.3, 0)

    # Final stats
    hair_coverage = np.sum(mask == 255) / mask.size * 100
    final_coverage = np.sum(final_mask == 255) / final_mask.size * 100
    mse = np.mean((img_resized.astype(float) - hair_free_final.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    stats_out = {
        'initial_hair_coverage': float(hair_coverage),
        'final_hair_coverage': float(final_coverage),
        'final_hair_pixels': int(np.sum(final_mask == 255)),
        'psnr': float(psnr)
    }

    if progress_callback is not None:
        progress_callback('complete')

    return hair_free_final.astype(np.uint8), final_mask.astype(np.uint8), stats_out


def main(save_output=False):
    # ============================
    # STAGE 0: Load & Resize
    # ============================
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # ============================
    # STAGE 1–4: Hair Enhancement
    # ============================
    if USE_MULTI_CHANNEL:
        channels = [
            (img_resized[:, :, 0], "Red"),
            (img_resized[:, :, 1], "Green"),
            (img_resized[:, :, 2], "Blue")
        ]
        processed = [process_channel(ch, name) for ch, name in channels]
        enhanced = np.maximum.reduce(processed)
        print("\nMulti-channel max projection applied.")
    else:
        enhanced = process_channel(img_resized[:, :, 0], "Red")

    # ============================
    # STAGE 5: Thresholding
    # ============================
    if USE_OTSU:
        thresh_val, mask = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"Otsu threshold: {thresh_val:.1f}")
    else:
        mask = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE, 0
        )

    hair_coverage = np.sum(mask == 255) / mask.size * 100
    print(f"\nINITIAL HAIR COVERAGE: {hair_coverage:.2f}%")

    # ============================
    # STAGE 6: Morphological Cleaning
    # ============================
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.dilate(clean_mask, dilate_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    final_mask = np.zeros_like(clean_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 10:
            final_mask[labels == i] = 255

    final_coverage = np.sum(final_mask == 255) / final_mask.size * 100
    print(f"FINAL HAIR MASK COVERAGE: {final_coverage:.2f}%")

    # ============================
    # STAGE 7: Inpainting
    # ============================
    inpaint_mask = (final_mask > 0).astype(np.uint8)

    if USE_CAREFUL_INPAINTING:
        hair_free_1 = cv2.inpaint(img_resized, inpaint_mask, 2, cv2.INPAINT_TELEA)
        hair_free_2 = cv2.inpaint(img_resized, inpaint_mask, 3, cv2.INPAINT_NS)
        hair_free = cv2.addWeighted(hair_free_1, 0.5, hair_free_2, 0.5, 0)

        mask_3ch = cv2.cvtColor(inpaint_mask, cv2.COLOR_GRAY2RGB)
        hair_free_final = np.where(mask_3ch == 1, hair_free, img_resized)
    else:
        hair_free_final = cv2.inpaint(
            img_resized, inpaint_mask, INPAINTING_RADIUS, cv2.INPAINT_TELEA
        )

    # ============================
    # STAGE 8: Selective Enhancement
    # ============================
    enhanced_img = hair_free_final.copy()

    if PRESERVE_DETAILS:
        smooth_mask = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=1)
        denoised = cv2.fastNlMeansDenoisingColored(
            hair_free_final, None, 3, 3, 7, 21
        )
        smooth_mask_3ch = cv2.cvtColor(smooth_mask, cv2.COLOR_GRAY2RGB) / 255.0
        enhanced_img = (
            hair_free_final * (1 - smooth_mask_3ch) +
            denoised * smooth_mask_3ch
        ).astype(np.uint8)

    lab = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    hair_free_final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    gaussian = cv2.GaussianBlur(hair_free_final, (0, 0), 1.0)
    hair_free_final = cv2.addWeighted(hair_free_final, 1.3, gaussian, -0.3, 0)

    # ============================
    # SAVE RESULT (optional)
    # ============================
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print(f"Hair coverage detected: {final_coverage:.2f}%")
    if save_output:
        cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(hair_free_final, cv2.COLOR_RGB2BGR))
        print(f"Saved output to: {OUTPUT_PATH}")
    else:
        print("Output not saved to disk (save_output=False)")
    print("=" * 50)


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    main()
