import numpy as np

# --------------------------
# Rendering primitives (RGB)
# --------------------------

def render_circle_and_dot_rgb(
    shape,
    circle_center,
    circle_radius,
    circle_thickness=1.0,
    dot_center=None,
    dot_radius=2.0,
    bg_color=(0, 0, 0),
    circle_color=(255, 255, 255),
    dot_color=(255, 0, 0),
    dtype=np.uint8,
):
    """
    Render a circle perimeter (ring) and a filled dot into an RGB image.

    Parameters
    ----------
    shape : (int, int)
        (height, width).
    circle_center : (float, float)
        (cy, cx) center of the circle perimeter in pixel coords.
    circle_radius : float
        Radius (pixels).
    circle_thickness : float, default 1.0
        Ring thickness (pixels).
    dot_center : (float, float) or None
        Center of the filled dot. If None, uses circle_center.
    dot_radius : float, default 2.0
        Dot radius (pixels).
    bg_color, circle_color, dot_color : tuple[int, int, int]
        RGB colors in 0..255.
    dtype : np.dtype
        Output dtype, typically uint8.

    Returns
    -------
    img : (H, W, 3) np.ndarray
        RGB image with rendered shapes.
    """
    H, W = shape
    cy, cx = circle_center
    if dot_center is None:
        dy, dx = cy, cx
    else:
        dy, dx = dot_center

    yy, xx = np.ogrid[:H, :W]
    dist_circle = np.hypot(yy - cy, xx - cx)
    dist_dot = np.hypot(yy - dy, xx - dx)

    half_t = float(circle_thickness) / 2.0
    ring_mask = np.abs(dist_circle - float(circle_radius)) <= half_t
    dot_mask = dist_dot <= float(dot_radius)

    # Initialize RGB canvas
    img = np.empty((H, W, 3), dtype=np.float32)
    img[:] = np.array(bg_color, np.float32)

    # Draw ring (perimeter)
    img[ring_mask] = np.array(circle_color, np.float32)

    # Draw dot (filled). Overwrite ring where overlapping:
    img[dot_mask] = np.array(dot_color, np.float32)

    return img.astype(dtype)


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
