
import torch
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

from .tgif_streaming_dataset import TGIFStreamingDataset


MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
TSV_PATH = "data/tgif_metadata/tgif-v1.0.tsv"


def main():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    train_ds = TGIFStreamingDataset(
        tsv_path=TSV_PATH,
        processor=processor,
        max_samples=None,
        # Adjust url_col / caption_col if your TSV uses different headers
        url_col="gif_url",
        caption_col="sentence",
    )

    training_args = TrainingArguments(
        output_dir="./llava_tgif_igvlm_stream",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        max_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        save_steps=1000,
        logging_steps=50,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.train()
    trainer.save_model("./llava_tgif_igvlm_stream")


if __name__ == "__main__":
    main()
