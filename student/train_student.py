import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .dataset import TeacherLabelDataset
from .model import SimpleStudentCaptioner


def train_student():
    # Placeholder hyperparameters
    vocab_size = 6000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleStudentCaptioner(vocab_size=vocab_size).to(device)
    dataset = TeacherLabelDataset("data/processed/teacher_labels.jsonl")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # NOTE: This is a skeleton; you still need to:
    # - build a tokenizer/vocab
    # - tokenize teacher_caption/original_caption
    # - move tensors to device
    # - compute sequence loss
    for epoch in range(1):
        for batch in loader:
            # TODO: tokenize captions -> input_ids + labels
            # logits = model(input_ids)
            # loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            # loss.backward()
            # optim.step()
            pass


if __name__ == "__main__":
    train_student()
