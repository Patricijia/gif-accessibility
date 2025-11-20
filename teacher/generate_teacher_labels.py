import csv
import io
import json

import requests
import numpy as np
import imageio.v3 as iio
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from .tgif_streaming_dataset import bytes_to_frames, make_grid


MODEL_NAME = "./llava_tgif_igvlm_stream"
TSV_PATH = "data/tgif_metadata/tgif-v1.0.tsv"
OUTPUT_PATH = "data/processed/teacher_labels.jsonl"


@torch.inference_mode()
def generate():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()

with open(TSV_PATH, newline="", encoding="utf-8") as f_in, \
        open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    reader = csv.reader(f_in, delimiter="\t")
    for i, row in enumerate(reader):
        if not row:
            continue

        gif_id = i                     # synthetic ID
        gif_url = row[0].strip()
        caption = row[-1].strip()      # last column = sentence

        if not gif_url:
            continue

        try:
            resp = requests.get(gif_url, timeout=15)
            resp.raise_for_status()
            gif_bytes = resp.content

            frames = bytes_to_frames(gif_bytes, num_frames=8)
            grid_img = make_grid(frames, rows=2, cols=4)

            prompt = "<image>\nPlease describe this GIF in one sentence."
            text = "<|user|>\n" + prompt + "<|end|>\n<|assistant|>\n"

            inputs = processor(
                text=[text],
                images=[grid_img],
                return_tensors="pt",
            ).to(model.device)

            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )[0]

            full_text = processor.tokenizer.decode(out_ids, skip_special_tokens=True)
            teacher_caption = full_text.strip()

            record = {
                "id": gif_id,
                "url": gif_url,
                "teacher_caption": teacher_caption,
                "original_caption": caption,
            }
            f_out.write(json.dumps(record) + "\n")

        except Exception as e:
            print(f"[generate_teacher_labels] Skipping {gif_url}: {e}")
            continue


if __name__ == "__main__":
    generate()
