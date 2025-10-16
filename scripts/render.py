import numpy as np
from typing import Tuple, Optional

# --------------------------
# Rendering primitives (RGB)
# --------------------------

def render_circle(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    thickness: float,
    color: Optional[Tuple[float, float, float]] = None,
    in_place: bool = False,
) -> np.ndarray:
    """
    Draw a circle outline (ring) on a color image.

    Args:
        image: (H, W, 3) array. dtype can be uint8, float32, etc.
        center: (cy, cx) in pixel coordinates (row, col), floats allowed.
        radius: Circle radius in pixels.
        thickness: Ring thickness in pixels (>= 0). If 0, nothing is drawn.
        color: Optional (R, G, B). If None, uses max value for integer dtypes
               (e.g., 255) or 1.0 for float images.
        in_place: If True, modify the input image; otherwise operate on a copy.

    Returns:
        The image with the circle drawn (same dtype as input).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if thickness < 0:
        raise ValueError("thickness must be non-negative")

    img = image if in_place else image.copy()
    H, W, _ = img.shape
    cy, cx = center

    # Choose default color if none provided.
    if color is None:
        if np.issubdtype(img.dtype, np.integer):
            vmax = np.iinfo(img.dtype).max
            draw_color = np.array([vmax, vmax, vmax], dtype=img.dtype)
        else:
            draw_color = np.array([1.0, 1.0, 1.0], dtype=img.dtype)
    else:
        draw_color = np.asarray(color, dtype=img.dtype)
        if draw_color.shape != (3,):
            raise ValueError("color must be a 3-tuple (R, G, B)")

    if thickness == 0 or radius == 0:
        return img

    # Compute distance-squared grid to the (float) center.
    ys = np.arange(H, dtype=np.float32)[:, None]  # shape (H, 1)
    xs = np.arange(W, dtype=np.float32)[None, :]  # shape (1, W)
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2

    # Ring band [r_inner, r_outer] with half-thickness on each side.
    half_t = thickness / 2.0
    r_inner = max(0.0, radius - half_t)
    r_outer = radius + half_t

    mask = (dist2 >= r_inner * r_inner) & (dist2 <= r_outer * r_outer)

    # Assign color on the ring. mask is (H, W); img[mask] is (N, 3) and broadcasts draw_color.
    img[mask] = draw_color
    return img

def render_dot(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    color: Optional[Tuple[float, float, float]] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw a circle outline (ring) on a color image.

    Args:
        image: (H, W, 3) array. dtype can be uint8, float32, etc.
        center: (cy, cx) in pixel coordinates (row, col), floats allowed.
        radius: Circle radius in pixels.
        thickness: Ring thickness in pixels (>= 0). If 0, nothing is drawn.
        color: Optional (R, G, B). If None, uses max value for integer dtypes
               (e.g., 255) or 1.0 for float images.
        in_place: If True, modify the input image; otherwise operate on a copy.

    Returns:
        The image with the circle drawn (same dtype as input).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    img = image.copy()
    H, W, _ = img.shape
    cy, cx = center

    # Choose default color if none provided.
    if color is None:
        if np.issubdtype(img.dtype, np.integer):
            vmax = np.iinfo(img.dtype).max
            draw_color = np.array([vmax, vmax, vmax], dtype=img.dtype)
        else:
            draw_color = np.array([1.0, 1.0, 1.0], dtype=img.dtype)
    else:
        draw_color = np.asarray(color, dtype=img.dtype)
        if draw_color.shape != (3,):
            raise ValueError("color must be a 3-tuple (R, G, B)")

    if radius == 0:
        return img

    # Compute distance-squared grid to the (float) center.
    ys = np.arange(H, dtype=np.float32)[:, None]  # shape (H, 1)
    xs = np.arange(W, dtype=np.float32)[None, :]  # shape (1, W)
    dist2 = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    mask = dist2 <= radius

    # Assign color on the ring. mask is (H, W); img[mask] is (N, 3) and broadcasts draw_color.
    img[mask] = draw_color
    return img

def render_line(
    image: np.ndarray,
    point: Tuple[float, float],
    angle: float,
    thickness: float,
    color: Optional[Tuple[float, float, float]] = None,
    in_place: bool = False,
) -> np.ndarray:
    """
    Draw an infinite straight line on a color image as a solid band of given thickness.

    Args:
        image: (H, W, 3) array, any numeric dtype (e.g., uint8, float32).
        point: (py, px) = (row, col) of a point on the line (floats allowed).
        angle: Line direction angle. Measured from +x axis (rightwards) toward +y (downwards).
               Use radians by default; set `degrees=True` to interpret in degrees.
        thickness: Line thickness in pixels (>= 0). If 0, nothing is drawn.
        color: Optional (R, G, B). If None, uses max value for integer dtype or 1.0 for float.
        in_place: If True, modify input image; otherwise operate on a copy.
        degrees: If True, `angle` is in degrees; otherwise radians.

    Returns:
        Image with the line drawn (same dtype as input).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if thickness < 0:
        raise ValueError("thickness must be non-negative")

    img = image if in_place else image.copy()
    H, W, _ = img.shape
    py, px = point

    # Default color based on dtype
    if color is None:
        if np.issubdtype(img.dtype, np.integer):
            vmax = np.iinfo(img.dtype).max
            draw_color = np.array([vmax, vmax, vmax], dtype=img.dtype)
        else:
            draw_color = np.array([1.0, 1.0, 1.0], dtype=img.dtype)
    else:
        draw_color = np.asarray(color, dtype=img.dtype)
        if draw_color.shape != (3,):
            raise ValueError("color must be a 3-tuple (R, G, B)")

    if thickness == 0:
        return img

    # Angle handling (image coords: x->cols (right), y->rows (down))
    theta = float(angle)
    c, s = np.cos(theta), -np.sin(theta) # Rico: negative because it's up-down flipped.

    # Unit normal vector to the line (in (x,y) ordering)
    # Line direction d = (c, s); normal n = (-s, c).
    nx, ny = -s, c

    # Pixel-center grid (X for columns, Y for rows)
    X = np.arange(W, dtype=np.float32)[None, :]   # shape (1, W)
    Y = np.arange(H, dtype=np.float32)[:, None]   # shape (H, 1)

    # Signed perpendicular distance to the infinite line through (px, py)
    dist = (nx * (X - px)) + (ny * (Y - py))

    # Longitudinal (along-ray) projection length t of each pixel center onto d = (c, s)
    # Keep only points where t >= 0 to form a ray starting at (px, py).
    t = (c * (X - px)) + (s * (Y - py))

    half_t = thickness / 2.0
    mask = (np.abs(dist) <= half_t) & (t >= 0.0)

    # Apply color to all pixels within the band
    img[mask] = draw_color
    return img


# -------------------------------------------
# Utilities to stack frames (NumPy-only path)
# -------------------------------------------

def stack_frames(frames_list):
    """
    Stack a list of (H, W, 3) frames into (T, H, W, 3) uint8 array.

    All frames must share the same shape and dtype.
    """
    if not frames_list:
        raise ValueError("frames_list is empty.")
    arr = np.stack(frames_list, axis=0)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


# -------------------------------------------------
# Optional: write an mp4 using a NumPy-friendly lib
# -------------------------------------------------

def write_video_mp4(frames, fps=30, path="out.mp4",
                    codec="libx264", crf=18, pix_fmt="yuv420p"):
    """
    Write (T, H, W, 3) uint8 frames to MP4 using imageio==2.9.0 and
    imageio-ffmpeg==0.4.3.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), dtype uint8.
    fps : int
        Frames per second.
    path : str
        Output file path (e.g., 'out.mp4').
    codec : str
        FFMPEG codec (e.g., 'libx264', 'libx265', 'mpeg4').
    crf : int
        Constant Rate Factor for libx264/libx265 (lower = higher quality; ~18–23 good).
    pix_fmt : str
        Pixel format. 'yuv420p' is widely compatible.

    Returns
    -------
    path : str
        The output path that was written.
    """
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be shape (T, H, W, 3)")

    try:
        import imageio  # v2 API
    except Exception as e:
        raise RuntimeError(
            "imageio==2.9.0 is required for this function."
        ) from e

    # macro_block_size=None avoids implicit resizing to multiples of 16
    # ffmpeg_params allows CRF and pixel format control
    writer = imageio.get_writer(
        uri=path,
        fps=fps,
        codec=codec,
        format='FFMPEG',
        macro_block_size=None,
        ffmpeg_params=['-crf', str(crf), '-pix_fmt', pix_fmt],
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()

    return path


# -----------------
# Example generation
# -----------------
if __name__ == "__main__":
    H, W = 240, 320
    T = 60  # number of frames
    center = (H / 2, W / 2)
    radius = 80

    frames = []
    for t in range(T):
        # Animate the dot around the circle
        theta = 2 * np.pi * (t / T)
        dot_center = (
            center[0] + 0.6 * radius * np.sin(theta),
            center[1] + 0.6 * radius * np.cos(theta),
        )
        frame = render_circle_and_dot_rgb(
            shape=(H, W),
            circle_center=center,
            circle_radius=radius,
            circle_thickness=2.0,
            dot_center=dot_center,
            dot_radius=5.0,
            bg_color=(0, 0, 0),
            circle_color=(200, 200, 200),
            dot_color=(255, 80, 80),
            dtype=np.uint8,
        )
        frames.append(frame)

    video_array = stack_frames(frames)  # (T, H, W, 3) uint8

    # --- Option A: NumPy-only persistence (no extra libs) ---
    # Save frames to a .npz; you can encode to video later with ffmpeg if desired.
    # np.savez_compressed("frames.npz", frames=video_array)

    # --- Option B: Write a real MP4 right now (requires imageio + ffmpeg) ---
    # Uncomment to produce an MP4:
    # write_video_mp4(video_array, fps=30, path="demo.mp4")
