# ChromaText AI

A lightweight computer vision tool that recolors objects in images using natural language text prompts (e.g. *"replace crimson red with emerald green"*, *"change red to royal blue"*, or *"make it gold"*).

Built with OpenCV and Gradio, it segments target colors in HSV and CIELAB color spaces and preserves underlying lighting, shadows, and reflections so edits look natural rather than flat.

---

## Features

- **Natural Language Input:** Parse color commands like "replace [color] with [color]", "change [color] to [color]", or single-color targets like "make it [color]".
- **Auto Source Detection:** When only a target color is specified, the tool computes the median hue of the image's non-neutral pixels to automatically identify what to recolor.
- **Hue Wrap-Around Support:** Red hues wrap around 0°/360° on the color wheel; dual-interval thresholding prevents missed boundaries.
- **Clean Masking & Feathering:** Uses morphological opening and closing to remove noise and fill pinholes, followed by Gaussian edge feathering for smooth transitions.
- **Reflection Preservation:** Detects bright specular highlights and restores original pixels over them to keep glossy reflections intact.
- **Interactive UI:** Runs locally or on cloud spaces via Gradio with side-by-side previews, mask visualizer, and fine-tuning sliders.

---

## Example Commands

You can type commands in plain English or click the built-in quick preset buttons inside the app:

| Command Type | Example Prompt | Behavior |
| :--- | :--- | :--- |
| **Two-Color Replacement** | `replace crimson red with emerald green` | Detects crimson areas and shifts hue/saturation to emerald green. |
| **Direct Transition** | `change red to royal blue` | Replaces red pixels with royal blue while locking surface shadows. |
| **Single-Color (Auto)** | `make it gold` | Automatically detects the dominant object color in the image and converts it to gold. |
| **Casual Phrasing** | `turn it into royal blue` | Identifies target color and recolors the segmented object. |

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/niharikabhandari24521/chromatext-ai.git
   cd chromatext-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   python main.py
   ```
   Open `http://127.0.0.1:7860` in your web browser. A pre-rendered 3D sphere is loaded automatically so you can test with one click.

---

## How It Works

1. **Text Parsing:** Extracts source and destination colors using regex patterns, with sequential token scanning and fuzzy matching fallbacks for typos or compound color names (*e.g., "royal blue", "forest green"*).
2. **Color Segmentation:** Converts the image to HSV and applies OpenCV thresholding (`cv2.inRange`) across mapped color boundaries.
3. **Morphological Post-Processing:** Applies elliptical opening and closing filters to remove false-positive noise and bridge gaps.
4. **Sub-Pixel Blending:** Creates a continuous alpha matte with Gaussian blur to prevent jagged borders.
5. **Color Transfer:** Replaces hue while locking the original Value (in HSV) or Lightness $L^*$ (in CIELAB) to preserve shading and contours, followed by specular highlight re-overlay.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
