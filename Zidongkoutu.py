import os
import sys
import time
import cv2
import numpy as np

try:
    import msvcrt
except ImportError:
    msvcrt = None


def ensure_bgra(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _sample_positions(length, step):
    positions = list(range(0, length, step))
    if not positions:
        return [0]
    if positions[-1] != length - 1:
        positions.append(length - 1)
    return positions


def _build_border_seeds(w, h, step):
    seed_points = set()
    for x in _sample_positions(w, step):
        seed_points.add((x, 0))
        seed_points.add((x, h - 1))
    for y in _sample_positions(h, step):
        seed_points.add((0, y))
        seed_points.add((w - 1, y))
    return seed_points


def read_image_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def write_image_unicode(path, img):
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        encoded.tofile(path)
        return True
    except Exception:
        return False


def get_white_trigger_with_timeout(default_value=235, timeout_seconds=3):
    prompt = (
        f"Input white threshold (default {default_value}, auto use default in "
        f"{timeout_seconds}s): "
    )

    if msvcrt is None:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return default_value
        if not raw:
            return default_value
        try:
            val = int(raw)
            return max(0, min(255, val))
        except ValueError:
            return default_value

    sys.stdout.write(prompt)
    sys.stdout.flush()

    start = time.time()
    buffer = ""

    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()

            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                if not buffer:
                    return default_value
                try:
                    val = int(buffer)
                    return max(0, min(255, val))
                except ValueError:
                    return default_value

            if ch == "\b":
                if buffer:
                    buffer = buffer[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if ch.isdigit():
                buffer += ch
                sys.stdout.write(ch)
                sys.stdout.flush()

        if time.time() - start >= timeout_seconds:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return default_value

        time.sleep(0.02)


def wait_for_next_action():
    if msvcrt is None:
        return "exit"

    print("Press Enter to process again, or Esc to exit.")
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "rerun"
        if ord(ch) == 27:
            return "exit"


def remove_white_edges(input_path, output_path, white_trigger=235):
    """Remove white borders by clearing border-connected near-white regions."""
    img = read_image_unicode(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Unable to load image {input_path}")
        return False

    img = ensure_bgra(img)
    h, w = img.shape[:2]

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    near_white_mask = np.where(
        ((bgr[:, :, 0] > white_trigger)
         & (bgr[:, :, 1] > white_trigger)
         & (bgr[:, :, 2] > white_trigger))
        & (alpha > 0),
        255,
        0,
    ).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        near_white_mask, connectivity=8
    )
    background_mask = np.zeros((h, w), np.uint8)
    small_component_area = max(2000, int(h * w * 0.05))

    for label in range(1, num_labels):
        comp = labels == label
        area = stats[label, cv2.CC_STAT_AREA]

        touches_left = np.any(comp[:, 0])
        touches_right = np.any(comp[:, w - 1])
        touches_top = np.any(comp[0, :])
        touches_bottom = np.any(comp[h - 1, :])
        edge_touch_count = sum([touches_left, touches_right, touches_top, touches_bottom])

        if edge_touch_count == 0:
            continue

        if edge_touch_count >= 2 or area <= small_component_area:
            background_mask[comp] = 255

    img[background_mask == 255] = [0, 0, 0, 0]

    ok = write_image_unicode(output_path, img)
    if ok:
        print(f"White edges removed. Saved to: {output_path}")
    else:
        print(f"Failed to save image: {output_path}")
    return ok


def process_directory(input_dir, output_dir, white_trigger=235, on_progress=None):
    os.makedirs(output_dir, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    input_files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in image_exts
    )

    if not input_files:
        print(f"No image files found in: {input_dir}")
        return 0, 0

    success_count = 0
    total_count = len(input_files)

    for index, file_name in enumerate(input_files):
        input_path = os.path.join(input_dir, file_name)
        output_name = f"output{index + 1}.png"
        output_path = os.path.join(output_dir, output_name)
        ok = remove_white_edges(input_path, output_path, white_trigger=white_trigger)
        if ok:
            success_count += 1
        if on_progress is not None:
            on_progress(index + 1, total_count, input_path, output_path, ok)

    print(f"Processed {success_count}/{total_count} images.")
    return success_count, total_count


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(
        sys.executable if getattr(sys, "frozen", False) else __file__
    ))
    input_dir = os.path.join(base_dir, "Input")
    output_dir = os.path.join(base_dir, "OutPut")

    while True:
        white_trigger = get_white_trigger_with_timeout(default_value=235, timeout_seconds=3)
        print(f"Using white threshold: {white_trigger}")
        process_directory(input_dir, output_dir, white_trigger=white_trigger)

        if not getattr(sys, "frozen", False):
            break

        action = wait_for_next_action()
        if action == "rerun":
            continue
        break
