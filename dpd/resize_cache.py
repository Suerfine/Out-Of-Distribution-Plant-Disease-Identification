"""One-off cache of downsized images. The originals go up to 3120 px; decoding them every epoch would make CNN
training CPU-bound, so each image is stored once with its shorter side at `size` px (JPEG, quality 95)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from dpd.data import IMAGE_DIR, PROJECT_ROOT


def resized_dir(size: int) -> Path:
    return PROJECT_ROOT / "runs" / "cache" / f"resized_{size}"


def _resize_one(file: str, size: int, out_dir: Path) -> None:
    dst = out_dir / (Path(file).stem + ".jpg")
    if dst.exists():
        return
    with Image.open(IMAGE_DIR / file) as img:
        img.draft("RGB", (size * 2, size * 2))  # JPEG: decode at reduced scale, much faster than a full decode
        img = img.convert("RGB")
        scale = size / min(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BICUBIC)
        img.save(dst, quality=95)


def ensure_resized(files, size: int, workers: int = 8) -> Path:
    """Create any missing resized copies; returns the folder. Cached file name = original stem + '.jpg'."""
    out_dir = resized_dir(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in files if not (out_dir / (Path(f).stem + ".jpg")).exists()]
    if files:
        with ThreadPoolExecutor(workers) as pool:
            list(tqdm(pool.map(lambda f: _resize_one(f, size, out_dir), files), total=len(files), desc=f"resize to {size}px"))
    return out_dir


def cached_name(file: str) -> str:
    return Path(file).stem + ".jpg"
