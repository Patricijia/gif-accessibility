from teacher.tgif_streaming_dataset import TGIFStreamingDataset
from transformers import AutoProcessor


def main():
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"  # or your chosen LLaVA model
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    ds = TGIFStreamingDataset(
        tsv_path="data/tgif_metadata/tgif-v1.0.tsv",
        processor=processor,
        max_samples=1,   # just the first GIF
    )

    for sample in ds:
        print("Got sample with keys:", sample.keys())
        break


if __name__ == "__main__":
    main()
