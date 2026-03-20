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


def _clear_selected_points(img, selected_points, color_tolerance):
    if not selected_points:
        return

    h, w = img.shape[:2]
    bgr = np.ascontiguousarray(img[:, :, :3])
    alpha = img[:, :, 3]
    lo_diff = (color_tolerance, color_tolerance, color_tolerance)
    up_diff = (color_tolerance, color_tolerance, color_tolerance)
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    for x, y in selected_points:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if alpha[y, x] == 0:
            continue
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(bgr, mask, seedPoint=(x, y), newVal=(0, 0, 0), loDiff=lo_diff, upDiff=up_diff, flags=flags)
        filled = mask[1:-1, 1:-1] == 255
        img[filled] = [0, 0, 0, 0]
def resize_to_fit(img, max_w, max_h):
    if max_w <= 0 or max_h <= 0:
        return img

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if new_w == w and new_h == h:
        return img

    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _remove_white_edges_from_image(img, white_trigger=235, selected_points=None, color_tolerance=5):
    img = ensure_bgra(img)
    _clear_selected_points(img, selected_points, color_tolerance)
    h, w = img.shape[:2]

    bgr = np.ascontiguousarray(img[:, :, :3])
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
    return img


def remove_white_edges(input_path, output_path, white_trigger=235, selected_points=None, color_tolerance=5):
    """Remove white borders by clearing border-connected near-white regions."""
    img = read_image_unicode(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Unable to load image {input_path}")
        return False

    img = _remove_white_edges_from_image(
        img,
        white_trigger=white_trigger,
        selected_points=selected_points,
        color_tolerance=color_tolerance,
    )

    ok = write_image_unicode(output_path, img)
    if ok:
        print(f"White edges removed. Saved to: {output_path}")
    else:
        print(f"Failed to save image: {output_path}")
    return ok


def process_directory(
    input_dir,
    output_dir,
    white_trigger=235,
    selected_points_map=None,
    color_tolerance=5,
    do_remove_white=True,
    do_remove_points=True,
    do_resize=False,
    target_width=0,
    target_height=0,
    keep_original_name=False,
    on_progress=None,
):
    os.makedirs(output_dir, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    input_files = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in image_exts:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, input_dir)
                input_files.append((rel_path, abs_path))

    if not input_files:
        print(f"No image files found in: {input_dir}")
        return 0, 0

    success_count = 0
    total_count = len(input_files)

    selected_points_map = selected_points_map or {}

    for index, (rel_path, input_path) in enumerate(input_files):
        if keep_original_name:
            output_rel = rel_path
        else:
            rel_dir = os.path.dirname(rel_path)
            output_rel = os.path.join(rel_dir, f"output{index + 1}.png") if rel_dir else f"output{index + 1}.png"
        output_path = os.path.join(output_dir, output_rel)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        selected_points = selected_points_map.get(os.path.normpath(input_path))
        ok = True

        img = read_image_unicode(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            ok = False
        else:
            img = ensure_bgra(img)

            if do_remove_white:
                img = _remove_white_edges_from_image(
                    img,
                    white_trigger=white_trigger,
                    selected_points=selected_points,
                    color_tolerance=color_tolerance,
                )

            if do_remove_points:
                _clear_selected_points(img, selected_points, color_tolerance)

            if do_resize:
                img = resize_to_fit(img, target_width, target_height)

            ok = write_image_unicode(output_path, img)
            if ok:
                print(f"Saved to: {output_path}")
            else:
                print(f"Failed to save image: {output_path}")

        if ok:
            success_count += 1
        if on_progress is not None:
            on_progress(index + 1, total_count, input_path, output_path, ok)

    print(f"Processed {success_count}/{total_count} images.")
    return success_count, total_count


def _get_next_frame_subdir(output_dir):
    """获取下一个可用的 frameN 子目录名"""
    index = 1
    while True:
        subdir_name = f"frame{index}"
        subdir_path = os.path.join(output_dir, subdir_name)
        if not os.path.exists(subdir_path):
            return subdir_path, subdir_name
        index += 1


def extract_video_frames(
    video_path,
    output_dir,
    interval_seconds=1.0,
    target_frame_count=None,
    output_format="png",
    on_progress=None,
):
    """从视频中按间隔抽帧并保存为图片

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval_seconds: 抽帧间隔（秒），默认1秒，当 target_frame_count 设置时优先使用目标帧数
        target_frame_count: 目标抽帧数量，若设置则按此数量均匀抽帧
        output_format: 输出图片格式，默认png
        on_progress: 进度回调函数 (current, total, input_path, output_path, ok)

    Returns:
        (saved_count, total_expected, frame_subdir): 实际保存帧数, 预期总帧数, 使用的子目录名
    """
    if not os.path.isfile(video_path):
        print(f"视频文件不存在: {video_path}")
        return 0, 0, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return 0, 0, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # 计算帧间隔
    if target_frame_count and target_frame_count > 0:
        # 按目标帧数均匀抽帧
        total_expected = target_frame_count
        # 计算间隔帧数，确保能抽到目标数量
        frame_interval = max(1, int(total_frames / target_frame_count))
    else:
        # 按时间间隔抽帧
        frame_interval = int(fps * interval_seconds)
        if frame_interval < 1:
            frame_interval = 1
        total_expected = int(duration / interval_seconds) + 1 if interval_seconds > 0 else 0

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 获取并创建 frameN 子目录
    frame_subdir_path, frame_subdir_name = _get_next_frame_subdir(output_dir)
    os.makedirs(frame_subdir_path, exist_ok=True)

    saved_count = 0

    # 获取视频文件名（不含扩展名）作为前缀
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    def _save_frame(frame, idx):
        output_name = f"{video_name}_frame_{idx:04d}.{output_format}"
        output_path = os.path.join(frame_subdir_path, output_name)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        ok = write_image_unicode(output_path, frame)
        if ok:
            print(f"保存帧: {output_path}")
        else:
            print(f"保存失败: {output_path}")
        return ok, output_path

    if target_frame_count and target_frame_count > 0:
        # count 模式：精准 seek，只读目标帧数次，release 时无残留缓冲
        for i in range(target_frame_count):
            frame_pos = i * frame_interval
            if frame_pos >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            success, frame = cap.read()
            if not success:
                break
            ok, output_path = _save_frame(frame, saved_count)
            if ok:
                saved_count += 1
            if on_progress is not None:
                on_progress(saved_count, total_expected, video_path, output_path, ok)
    else:
        # interval 模式：顺序读帧，自然跑到 EOF，release 无阻塞
        read_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if read_count % frame_interval == 0:
                ok, output_path = _save_frame(frame, saved_count)
                if ok:
                    saved_count += 1
                if on_progress is not None:
                    on_progress(saved_count, total_expected, video_path, output_path, ok)
            read_count += 1

    cap.release()
    print(f"视频抽帧完成: 共保存 {saved_count} 帧到 {frame_subdir_path}")
    return saved_count, total_expected, frame_subdir_name


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
