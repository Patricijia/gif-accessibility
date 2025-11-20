import csv
import io
from typing import Optional, Iterator, Dict, Any

import requests
import numpy as np
import imageio.v3 as iio
from PIL import Image
from torch.utils.data import IterableDataset


def bytes_to_frames(gif_bytes: bytes, num_frames: int = 8):
    """Convert GIF bytes to a list of uniformly sampled frames (H, W, 3)."""
    arr = iio.imread(io.BytesIO(gif_bytes))
    if arr.ndim == 3:
        arr = arr[None, ...]
    T = arr.shape[0]
    idx = np.linspace(0, T - 1, num_frames, dtype=int)
    return [arr[i] for i in idx]


def make_grid(frames, rows: int = 2, cols: int = 4) -> Image.Image:
    """Create a simple rows×cols grid from a list of frames (H, W, 3)."""
    h, w, _ = frames[0].shape
    grid = Image.new("RGB", (w * cols, h * rows))
    for k, frame in enumerate(frames[: rows * cols]):
        r, c = divmod(k, cols)
        grid.paste(Image.fromarray(frame), (c * w, r * h))
    return grid


class TGIFStreamingDataset(IterableDataset):
    """
    Stream TGIF samples from a TSV *without* IDs or headers.

    Expected row format (like in your screenshot):
        <url>\t<caption>

    If there are extra columns, we take:
        url     = row[0]
        caption = row[-1]
    """

    def __init__(
        self,
        tsv_path: str,
        processor,
        max_samples: Optional[int] = None,
        num_frames: int = 8,
        grid_rows: int = 2,
        grid_cols: int = 4,
        timeout: int = 15,
    ) -> None:
        super().__init__()
        self.tsv_path = tsv_path
        self.processor = processor
        self.max_samples = max_samples
        self.num_frames = num_frames
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.timeout = timeout

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if self.max_samples is not None and i >= self.max_samples:
                    break
                if not row:
                    continue

                gif_url = row[0].strip()
                caption = row[-1].strip()   # last column = sentence
                if not gif_url or not caption:
                    continue

                try:
                    resp = requests.get(gif_url, timeout=self.timeout)
                    resp.raise_for_status()
                    gif_bytes = resp.content

                    frames = bytes_to_frames(gif_bytes, num_frames=self.num_frames)
                    grid_img = make_grid(frames, rows=self.grid_rows, cols=self.grid_cols)

                    user_msg = "<image>\nPlease describe this GIF in one sentence."
                    text = (
                        "<|user|>\n" + user_msg + "<|end|>\n"
                        "<|assistant|>\n" + caption + "<|end|>\n"
                    )

                    inputs = self.processor(
                        text=[text],
                        images=[grid_img],
                        return_tensors="pt",
                    )

                    labels = inputs["input_ids"].clone()
                    pad_id = self.processor.tokenizer.pad_token_id
                    labels[labels == pad_id] = -100

                    batch = {k: v.squeeze(0) for k, v in inputs.items()}
                    batch["labels"] = labels.squeeze(0)

                    yield batch

                except Exception as e:
                    print(f"[TGIFStreamingDataset] Skipping {gif_url}: {e}")
                    continue
