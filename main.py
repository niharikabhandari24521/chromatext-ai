"""
========================================================================================
CHROMATEXT AI • NATURAL LANGUAGE COLOR REPLACEMENT ENGINE (SINGLE-FILE EDITION)
========================================================================================
All-in-one Python implementation containing:
1. NLP Parser (Command extraction, Color Dictionary, Circular Hue mapping, Fuzzy fallback,
   Sequential Multi-color scanner, and Single-color Auto-detection)
2. Computer Vision Engine (OpenCV HSV & CIELAB, Morphology, Edge Feathering,
   Specular Highlight Preservation, Lighting-Preserved Transfer)
3. Interactive Gradio Web UI (Side-by-side previews, Mask visualization, Diagnostics)

Usage:
    pip install opencv-python gradio numpy
    python main.py
========================================================================================
"""

from __future__ import annotations
import re
import difflib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import gradio as gr


# ======================================================================================
# SECTION 1: NLP COLOR PARSING & MATHEMATICAL HSV DICTIONARY
# ======================================================================================

@dataclass
class ColorSpec:
    """
    Mathematical color specification across HSV and RGB color spaces.
    OpenCV HSV space:
      - Hue (H): [0, 179] -> representing [0°, 360°] / 2
      - Saturation (S): [0, 255] -> color purity
      - Value (V): [0, 255] -> brightness / luminance
    """
    name: str
    hsv_ranges: List[Tuple[np.ndarray, np.ndarray]]  # Handles Red circular wrap-around!
    target_hsv: Tuple[int, int, int]
    target_rgb: Tuple[int, int, int]
    hex_code: str


# Comprehensive dictionary of colors mapped to OpenCV HSV ranges
COLOR_PALETTE: Dict[str, ColorSpec] = {
    # Red family (Circular wrap-around: two intervals required!)
    "red": ColorSpec("red", [
        (np.array([0, 70, 50], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        (np.array([170, 70, 50], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    ], (0, 240, 230), (230, 20, 20), "#E61414"),
    "crimson": ColorSpec("crimson", [
        (np.array([0, 110, 60], dtype=np.uint8), np.array([8, 255, 230], dtype=np.uint8)),
        (np.array([172, 110, 60], dtype=np.uint8), np.array([179, 255, 230], dtype=np.uint8))
    ], (175, 220, 180), (180, 20, 50), "#B41432"),
    "maroon": ColorSpec("maroon", [
        (np.array([0, 100, 20], dtype=np.uint8), np.array([10, 255, 120], dtype=np.uint8)),
        (np.array([170, 100, 20], dtype=np.uint8), np.array([179, 255, 120], dtype=np.uint8))
    ], (176, 210, 110), (110, 15, 30), "#6E0F1E"),

    # Orange & Yellow family
    "orange": ColorSpec("orange", [(np.array([11, 80, 70], dtype=np.uint8), np.array([22, 255, 255], dtype=np.uint8))], (16, 230, 240), (245, 130, 20), "#F58214"),
    "coral": ColorSpec("coral", [(np.array([6, 70, 110], dtype=np.uint8), np.array([18, 210, 255], dtype=np.uint8))], (12, 160, 240), (245, 125, 105), "#F57D69"),
    "amber": ColorSpec("amber", [(np.array([18, 130, 100], dtype=np.uint8), np.array([26, 255, 255], dtype=np.uint8))], (22, 240, 245), (245, 180, 15), "#F5B40F"),
    "yellow": ColorSpec("yellow", [(np.array([23, 70, 80], dtype=np.uint8), np.array([34, 255, 255], dtype=np.uint8))], (29, 230, 245), (245, 230, 25), "#F5E619"),
    "gold": ColorSpec("gold", [(np.array([22, 110, 90], dtype=np.uint8), np.array([32, 255, 240], dtype=np.uint8))], (26, 210, 220), (225, 185, 30), "#E1B91E"),

    # Green family
    "green": ColorSpec("green", [(np.array([35, 60, 40], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))], (60, 220, 200), (30, 200, 50), "#1EC832"),
    "emerald green": ColorSpec("emerald green", [(np.array([65, 90, 50], dtype=np.uint8), np.array([85, 255, 240], dtype=np.uint8))], (75, 215, 185), (20, 185, 95), "#14B95F"),
    "emerald": ColorSpec("emerald", [(np.array([65, 90, 50], dtype=np.uint8), np.array([85, 255, 240], dtype=np.uint8))], (75, 215, 185), (20, 185, 95), "#14B95F"),
    "lime": ColorSpec("lime", [(np.array([36, 100, 70], dtype=np.uint8), np.array([55, 255, 255], dtype=np.uint8))], (46, 240, 240), (135, 235, 25), "#87EB19"),
    "olive": ColorSpec("olive", [(np.array([28, 45, 30], dtype=np.uint8), np.array([45, 200, 140], dtype=np.uint8))], (36, 140, 115), (105, 115, 40), "#697328"),
    "mint": ColorSpec("mint", [(np.array([60, 35, 120], dtype=np.uint8), np.array([85, 160, 255], dtype=np.uint8))], (72, 100, 240), (150, 240, 190), "#96F0BE"),

    # Cyan & Blue family
    "cyan": ColorSpec("cyan", [(np.array([86, 70, 70], dtype=np.uint8), np.array([99, 255, 255], dtype=np.uint8))], (90, 240, 240), (20, 230, 235), "#14E6EB"),
    "teal": ColorSpec("teal", [(np.array([82, 80, 35], dtype=np.uint8), np.array([96, 255, 160], dtype=np.uint8))], (89, 210, 130), (20, 125, 130), "#147D82"),
    "turquoise": ColorSpec("turquoise", [(np.array([83, 70, 80], dtype=np.uint8), np.array([96, 255, 255], dtype=np.uint8))], (88, 190, 225), (45, 215, 205), "#2DD7CD"),
    "blue": ColorSpec("blue", [(np.array([100, 65, 45], dtype=np.uint8), np.array([132, 255, 255], dtype=np.uint8))], (115, 230, 235), (30, 95, 235), "#1E5FEB"),
    "royal blue": ColorSpec("royal blue", [(np.array([110, 110, 60], dtype=np.uint8), np.array([126, 255, 255], dtype=np.uint8))], (118, 235, 220), (35, 75, 225), "#234BE1"),
    "navy": ColorSpec("navy", [(np.array([105, 90, 20], dtype=np.uint8), np.array([130, 255, 115], dtype=np.uint8))], (116, 240, 95), (10, 25, 95), "#0A195F"),
    "sky blue": ColorSpec("sky blue", [(np.array([98, 45, 120], dtype=np.uint8), np.array([118, 180, 255], dtype=np.uint8))], (103, 130, 245), (115, 190, 245), "#73BEF5"),

    # Purple & Pink family
    "purple": ColorSpec("purple", [(np.array([130, 60, 40], dtype=np.uint8), np.array([155, 255, 255], dtype=np.uint8))], (142, 220, 190), (140, 30, 190), "#8C1EBE"),
    "violet": ColorSpec("violet", [(np.array([135, 70, 50], dtype=np.uint8), np.array([158, 255, 255], dtype=np.uint8))], (146, 210, 215), (160, 45, 215), "#A02DD7"),
    "lavender": ColorSpec("lavender", [(np.array([125, 25, 110], dtype=np.uint8), np.array([150, 140, 255], dtype=np.uint8))], (138, 80, 235), (195, 165, 235), "#C3A5EB"),
    "pink": ColorSpec("pink", [(np.array([150, 40, 100], dtype=np.uint8), np.array([172, 200, 255], dtype=np.uint8))], (165, 130, 245), (245, 130, 185), "#F582B9"),
    "magenta": ColorSpec("magenta", [(np.array([148, 100, 70], dtype=np.uint8), np.array([168, 255, 255], dtype=np.uint8))], (155, 240, 235), (235, 25, 185), "#EB19B9"),
}

# Synonyms & descriptive aliases
SYNONYMS: Dict[str, str] = {
    "crimson red": "crimson", "dark red": "maroon", "deep red": "crimson", "bright red": "red",
    "dark blue": "navy", "light blue": "sky blue", "bright blue": "royal blue", "cobalt": "royal blue",
    "dark green": "olive", "light green": "lime", "bright green": "green", "jade": "emerald green",
    "golden": "gold", "golden yellow": "gold", "aqua": "cyan", "fuchsia": "magenta",
}

# Regex intent extraction rules
EXTRACTION_RULES = [
    re.compile(r"replace\s+(?P<src>[\w\s]+?)\s+(?:with|by)\s+(?P<dst>[\w\s]+?)(?:$|\.|\!)", re.I),
    re.compile(r"change\s+(?:the\s+)?(?P<src>[\w\s]+?)\s+to\s+(?P<dst>[\w\s]+?)(?:$|\.|\!)", re.I),
    re.compile(r"turn\s+(?:the\s+)?(?P<src>[\w\s]+?)\s+into\s+(?P<dst>[\w\s]+?)(?:$|\.|\!)", re.I),
    re.compile(r"swap\s+(?:the\s+)?(?P<src>[\w\s]+?)\s+for\s+(?P<dst>[\w\s]+?)(?:$|\.|\!)", re.I),
    re.compile(r"convert\s+(?:the\s+)?(?P<src>[\w\s]+?)\s+to\s+(?P<dst>[\w\s]+?)(?:$|\.|\!)", re.I),
    re.compile(r"^(?:from\s+)?(?P<src>[\w\s]+?)\s+to\s+(?P<dst>[\w\s]+?)(?:$|\.|\!)", re.I),
]


def resolve_color(term: str) -> Optional[ColorSpec]:
    """Resolve natural color term via: 1) Exact match, 2) Synonyms, 3) Fuzzy matching."""
    cleaned = re.sub(r"\b(color|colour|shade|the|a|an)\b", "", term.strip().lower()).strip()
    if not cleaned:
        return None
    if cleaned in COLOR_PALETTE:
        return COLOR_PALETTE[cleaned]
    if cleaned in SYNONYMS:
        return COLOR_PALETTE[SYNONYMS[cleaned]]

    # Fuzzy match fallback using Levenshtein / SequenceMatcher
    all_names = list(COLOR_PALETTE.keys()) + list(SYNONYMS.keys())
    matches = difflib.get_close_matches(cleaned, all_names, n=1, cutoff=0.55)
    if matches:
        match_key = matches[0]
        canonical = SYNONYMS.get(match_key, match_key)
        return COLOR_PALETTE.get(canonical)
    return None


def scan_colors_in_text(text: str) -> List[ColorSpec]:
    """
    Scans prompt text sequentially to extract recognized color names.
    Checks multi-word n-grams (e.g. 'royal blue', 'emerald green') before single words.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    found_specs: List[ColorSpec] = []
    i = 0
    all_known_names = set(COLOR_PALETTE.keys()).union(set(SYNONYMS.keys()))

    while i < len(tokens):
        matched_spec = None
        matched_len = 0
        # Check 3-word, 2-word, then 1-word color phrases
        for length in (3, 2, 1):
            if i + length <= len(tokens):
                phrase = " ".join(tokens[i:i + length])
                clean_phrase = re.sub(r"\b(color|colour|shade|the|a|an)\b", "", phrase).strip()
                if not clean_phrase:
                    continue

                if clean_phrase in all_known_names:
                    spec = resolve_color(clean_phrase)
                    if spec:
                        matched_spec = spec
                        matched_len = length
                        break
                else:
                    # High-confidence fuzzy match for typos
                    close = difflib.get_close_matches(clean_phrase, list(all_known_names), n=1, cutoff=0.7)
                    if close:
                        spec = resolve_color(close[0])
                        if spec:
                            matched_spec = spec
                            matched_len = length
                            break

        if matched_spec:
            found_specs.append(matched_spec)
            i += matched_len
        else:
            i += 1

    return found_specs


def auto_detect_source_color(image_rgb: np.ndarray) -> Optional[ColorSpec]:
    """
    Auto-detects the dominant source color from the input image:
    Computes the circular median Hue of all non-neutral pixels (Saturation > 40, Value > 40)
    and matches it to the nearest color in COLOR_PALETTE.
    """
    if image_rgb is None or image_rgb.size == 0:
        return None

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Mask for non-neutral, vivid pixels
    non_neutral = (sat > 40) & (val > 40)
    if not np.any(non_neutral):
        return None

    hues = hsv[:, :, 0][non_neutral]
    median_hue = float(np.median(hues))

    # Match median_hue to the closest ColorSpec using circular hue distance on [0, 179]
    best_spec = None
    min_dist = float("inf")

    for spec in COLOR_PALETTE.values():
        target_h = float(spec.target_hsv[0])
        dist = abs(median_hue - target_h)
        circ_dist = min(dist, 180.0 - dist)
        if circ_dist < min_dist:
            min_dist = circ_dist
            best_spec = spec

    return best_spec


def parse_prompt(
    prompt: str,
    image_rgb: Optional[np.ndarray] = None
) -> Tuple[Optional[ColorSpec], Optional[ColorSpec], Optional[str]]:
    """
    Parse text prompt into (source_color_spec, target_color_spec, error_message).

    Features:
      1. Regex pattern matching ('replace [src] with [dst]', 'change [src] to [dst]', etc.)
      2. Flexible sequential scan fallback if regex misses: takes first 2 recognized colors.
      3. Single-color command support: auto-detects source color from image median hue.
    """
    if not prompt or not prompt.strip():
        return None, None, "Please enter a command like 'replace crimson red with emerald green'."

    src_spec, dst_spec = None, None

    # Step 1: Attempt structured regex extraction
    for rule in EXTRACTION_RULES:
        m = rule.search(prompt.strip())
        if m:
            src_t = m.group("src")
            dst_t = m.group("dst")
            s = resolve_color(src_t)
            d = resolve_color(dst_t)
            if s and d:
                src_spec, dst_spec = s, d
                break

    # Step 2: Flexible Sequential Scanner Fallback (if regex did not yield both colors)
    if not src_spec or not dst_spec:
        found_colors = scan_colors_in_text(prompt)

        if len(found_colors) >= 2:
            # First recognized color is source, second is target
            src_spec = found_colors[0]
            dst_spec = found_colors[1]

        elif len(found_colors) == 1:
            # Step 3: Single-color command support
            dst_spec = found_colors[0]
            if image_rgb is not None:
                src_spec = auto_detect_source_color(image_rgb)
                if not src_spec:
                    return None, None, (
                        f"Target color '{dst_spec.name}' detected, but could not auto-detect a source color "
                        "from the image (image appears neutral/monochrome). Please specify: 'replace [Color] with [Color]'."
                    )
            else:
                return None, None, (
                    f"Target color '{dst_spec.name}' recognized. Please upload an image so the source color can be auto-detected."
                )

    if not src_spec or not dst_spec:
        return None, None, (
            f"Could not parse valid color command from '{prompt}'.\n"
            "Try formats like:\n"
            "  • 'replace crimson with emerald green'\n"
            "  • 'change red to royal blue'\n"
            "  • 'make it gold' (auto-detects source color)"
        )

    return src_spec, dst_spec, None


# ======================================================================================
# SECTION 2: COMPUTER VISION COLOR SEGMENTATION & RECOLORING ENGINE
# ======================================================================================

def process_color_replacement(
    image_rgb: np.ndarray,
    source_spec: ColorSpec,
    target_spec: ColorSpec,
    morph_kernel_size: int = 5,
    feather_radius: int = 5,
    sat_min: int = 40,
    transfer_mode: str = "HSV (Shading-Preserved)",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Core CV Pipeline:
      1. Converts RGB -> HSV.
      2. Thresholds image using multi-interval cv2.inRange (handles red wrap-around).
      3. Applies morphological Opening (removes noise) and Closing (fills holes).
      4. Feathers binary edges using Gaussian blur into a continuous alpha matte [0.0, 1.0].
      5. Performs lighting-preserving color transfer (locks original Value/L* channel).
      6. Specular Highlight Preservation: restores shiny reflections (Value > 240, Sat < 30).
    """
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # 1. Multi-interval thresholding (Red circular wrap-around handling)
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    for lower_bound, upper_bound in source_spec.hsv_ranges:
        lower = lower_bound.copy()
        upper = upper_bound.copy()
        lower[1] = max(lower[1], sat_min)  # Exclude neutral whites/grays
        interval_mask = cv2.inRange(hsv, lower, upper)
        raw_mask = cv2.bitwise_or(raw_mask, interval_mask)

    # 2. Morphological Cleaning with Elliptical kernel (no square directional bias)
    k_size = morph_kernel_size if morph_kernel_size % 2 != 0 else morph_kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    # Opening: removes false-positive background noise; Closing: fills internal pinholes
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Sub-pixel Gaussian Edge Feathering
    f_size = feather_radius if feather_radius % 2 != 0 else feather_radius + 1
    mask_float = clean_mask.astype(np.float32) / 255.0
    alpha_matte = cv2.GaussianBlur(mask_float, (f_size, f_size), 0)
    alpha_matte = np.clip(alpha_matte, 0.0, 1.0)
    alpha_3d = np.expand_dims(alpha_matte, axis=2)

    # 4. Color Transformation
    if "LAB" in transfer_mode.upper():
        # CIELAB Mode: Keep original perceptual lightness L*, transfer target chrominance (a*, b*)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_rgb_px = np.array([[[*target_spec.target_rgb]]], dtype=np.uint8)
        _, target_a, target_b = cv2.cvtColor(target_rgb_px, cv2.COLOR_RGB2LAB)[0, 0]

        recolored_lab = lab.copy()
        recolored_lab[:, :, 1] = float(target_a)
        recolored_lab[:, :, 2] = float(target_b)
        recolored_rgb = cv2.cvtColor(np.clip(recolored_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        # HSV Mode: Keep original Value (shading & folds), set target Hue, scale Saturation
        h_orig, s_orig, v_orig = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        target_h, target_s, _ = target_spec.target_hsv

        h_new = np.full_like(h_orig, fill_value=target_h)
        s_scale = target_s / 255.0
        s_new = np.clip(s_orig.astype(np.float32) * s_scale * 1.15, 0.0, 255.0).astype(np.uint8)
        v_new = v_orig  # Lock original brightness!

        recolored_hsv = np.stack([h_new, s_new, v_new], axis=2)
        recolored_rgb = cv2.cvtColor(recolored_hsv, cv2.COLOR_HSV2RGB)

    # 5. Alpha blending: Result = (1 - alpha) * Original + alpha * Recolored
    composite = (1.0 - alpha_3d) * image_rgb.astype(np.float32) + alpha_3d * recolored_rgb.astype(np.float32)
    output_image = np.clip(composite, 0, 255).astype(np.uint8)

    # 6. Specular Highlight Preservation:
    # Bright glossy/specular pixels (Value > 240, Saturation < 30, and alpha_matte > 0.1)
    # Re-overlay the original RGB pixels over those exact spots to keep reflections natural and metallic.
    specular_mask = (hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 30) & (alpha_matte > 0.1)
    output_image[specular_mask] = image_rgb[specular_mask]

    # Telemetry metrics
    masked_pixels = int(np.count_nonzero(clean_mask))
    coverage = round((masked_pixels / float(h * w)) * 100.0, 2)
    mask_viz = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB)

    stats = {
        "source": source_spec.name,
        "target": target_spec.name,
        "source_hex": source_spec.hex_code,
        "target_hex": target_spec.hex_code,
        "coverage": coverage,
        "pixels": masked_pixels,
    }
    return output_image, mask_viz, stats


# ======================================================================================
# SECTION 3: PROCEDURAL SAMPLE ASSET GENERATOR (Zero external dependencies)
# ======================================================================================

def create_synthetic_test_sphere() -> np.ndarray:
    """Generate a realistic 3D crimson sphere with ambient, diffuse, and specular lighting."""
    h, w = 480, 640
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    for y in range(h):
        img[y, :, :] = np.clip(245 - (y * 30 // h), 210, 255)

    cx, cy, r = 320, 240, 135
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    mask = dist <= r

    light = np.array([-0.5, -0.6, 0.6], dtype=np.float32)
    light /= np.linalg.norm(light)

    nx = (x_coords - cx) / float(r)
    ny = (y_coords - cy) / float(r)
    nz = np.sqrt(np.maximum(0.0, 1.0 - (nx**2 + ny**2)))
    diffuse = np.maximum(0.0, nx * light[0] + ny * light[1] + nz * light[2])
    lighting = np.clip(0.25 + 0.75 * diffuse, 0.0, 1.0)

    # Specular glint
    half = (light + np.array([0, 0, 1.0], dtype=np.float32))
    half /= np.linalg.norm(half)
    specular = (np.maximum(0.0, nx * half[0] + ny * half[1] + nz * half[2]) ** 32) * 0.75

    base_r, base_g, base_b = 220.0, 20.0, 55.0  # Crimson
    for c, base in enumerate([base_r, base_g, base_b]):
        sphere_ch = base * lighting + 255.0 * specular
        img[:, :, c] = np.where(mask, np.clip(sphere_ch, 0, 255).astype(np.uint8), img[:, :, c])

    return img


# ======================================================================================
# SECTION 4: GRADIO WEB APPLICATION INTERFACE
# ======================================================================================

def run_app_pipeline(image, prompt, morph_k, feather_r, sat_min, mode):
    if image is None:
        raise gr.Error("Please upload an image or click one of the quick test presets.")

    # Pass image to enable Single-Color auto-detection
    src_spec, dst_spec, err = parse_prompt(prompt, image_rgb=image)
    if err:
        error_card = f"""
        <div style="background: #2D1A1E; border-left: 4px solid #E5484D; padding: 12px; border-radius: 6px;">
            <strong style="color: #FF858D;">NLP Error:</strong>
            <span style="color: #F8D7DA; margin-left: 6px;">{err}</span>
        </div>
        """
        return image, np.zeros_like(image), error_card

    output_img, mask_viz, stats = process_color_replacement(
        image_rgb=image,
        source_spec=src_spec,
        target_spec=dst_spec,
        morph_kernel_size=int(morph_k),
        feather_radius=int(feather_r),
        sat_min=int(sat_min),
        transfer_mode=mode,
    )

    telemetry_html = f"""
    <div style="background: #161920; border: 1px solid #2B303C; border-radius: 8px; padding: 14px; font-family: sans-serif;">
        <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; border-radius: 50%; background: {stats['source_hex']}; display: inline-block;"></span>
                <span style="color: #F1F5F9; font-weight: 600; text-transform: capitalize;">{stats['source']}</span>
            </div>
            <span style="color: #64748B;">➔</span>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 14px; height: 14px; border-radius: 50%; background: {stats['target_hex']}; display: inline-block;"></span>
                <span style="color: #F1F5F9; font-weight: 600; text-transform: capitalize;">{stats['target']}</span>
            </div>
        </div>
        <div style="display: flex; gap: 16px; color: #94A3B8; font-size: 13px;">
            <span>Coverage: <strong style="color: #38BDF8;">{stats['coverage']}%</strong></span>
            <span>Pixels Changed: <strong style="color: #F1F5F9;">{stats['pixels']:,}</strong></span>
            <span>Filter: <strong style="color: #A78BFA;">{int(morph_k)}px</strong></span>
            <span>Feather: <strong style="color: #F472B6;">{int(feather_r)}px</strong></span>
        </div>
    </div>
    """
    return output_img, mask_viz, telemetry_html


def build_app():
    sample_img = create_synthetic_test_sphere()

    with gr.Blocks(title="Natural Language Color Replacement Engine") as demo:
        gr.Markdown(
            """
            # 🎨 Natural Language Color Replacement Tool
            ### *Single-File Production Implementation (OpenCV + NLP + Gradio)*
            Type commands like **"replace crimson red with emerald green"**, **"red to royal blue"**, or single-color commands like **"make it gold"**!
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Input Image", value=sample_img, type="numpy")
                prompt_in = gr.Textbox(
                    label="Command Prompt",
                    value="replace crimson red with emerald green",
                    placeholder="e.g. replace crimson with royal blue OR make it gold",
                )

                gr.Markdown("**Quick Prompts:**")
                with gr.Row():
                    btn1 = gr.Button("🔴 ➔ 🟢 Crimson to Emerald", size="sm")
                    btn2 = gr.Button("🔴 ➔ 🔵 Red to Royal Blue", size="sm")
                    btn3 = gr.Button("✨ Auto ➔ 🟡 Make it Gold", size="sm")

                with gr.Accordion("⚙️ Computer Vision Fine-Tuning", open=False):
                    mode_in = gr.Radio(
                        ["HSV (Shading-Preserved)", "CIELAB (Perceptual Lightness)"],
                        value="HSV (Shading-Preserved)",
                        label="Transfer Color Space",
                    )
                    k_slider = gr.Slider(1, 15, value=5, step=2, label="Morphology Kernel (Noise Cleaning)")
                    f_slider = gr.Slider(1, 21, value=5, step=2, label="Gaussian Edge Feathering (Alpha Matte)")
                    sat_slider = gr.Slider(0, 100, value=40, step=5, label="Min Saturation (Isolates Colors)")

                submit_btn = gr.Button("✨ Transform Color", variant="primary", size="lg")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🖼️ Recolored Image"):
                        img_out = gr.Image(label="Output Image", interactive=False)
                    with gr.TabItem("🎭 Segmentation Mask"):
                        mask_out = gr.Image(label="Binary Mask", interactive=False)
                info_out = gr.HTML("<div style='color: #64748B;'>Click 'Transform Color' to process image.</div>")

        # Event triggers
        btn1.click(lambda: "replace crimson red with emerald green", outputs=prompt_in)
        btn2.click(lambda: "change red to royal blue", outputs=prompt_in)
        btn3.click(lambda: "make it gold", outputs=prompt_in)

        submit_btn.click(
            fn=run_app_pipeline,
            inputs=[img_in, prompt_in, k_slider, f_slider, sat_slider, mode_in],
            outputs=[img_out, mask_out, info_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
