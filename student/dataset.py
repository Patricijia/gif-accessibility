import json
from typing import List, Dict, Any

from torch.utils.data import Dataset


class TeacherLabelDataset(Dataset):
    """Dataset that loads teacher labels for student training.

    This is a placeholder to be customized once you decide how to handle images
    (e.g., grids on disk or CNN features) and tokenization.
    """

    def __init__(self, labels_path: str):
        self.records: List[Dict[str, Any]] = []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "id": rec.get("id"),
            "teacher_caption": rec.get("teacher_caption", ""),
            "original_caption": rec.get("original_caption", ""),
        }
