import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor


WANTED_WORDS = [
    "visual", "wow", "learn", "backward", "dog", "two", "left", "happy", "nine", "go",
    "up", "bed", "stop", "one", "zero", "tree", "seven", "on", "four", "bird",
    "right", "eight", "no", "six", "forward", "house", "marvin", "sheila", "five", "off",
    "three", "down", "cat", "follow", "yes",
]
def resolve_audio_path(base_dir: str, split_name: str) -> str:
    if os.path.isabs(split_name):
        return split_name
    return os.path.join(base_dir, split_name)


class GscFolderDataset(Dataset):
    def __init__(self, base_dir: str, split_names: Sequence[str], label_to_id: Dict[str, int], target_sr: int = 16000):
        self.base_dir = base_dir
        self.split_names = list(split_names)
        self.label_to_id = label_to_id
        self.target_sr = target_sr
        self.samples: List[Tuple[str, int]] = []

        for split_name in self.split_names:
            split_dir = resolve_audio_path(base_dir, split_name)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            for label in sorted(os.listdir(split_dir)):
                label_dir = os.path.join(split_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                if label == "_background_noise_":
                    continue
                if label not in self.label_to_id:
                    raise ValueError(f"Unknown label folder: {label}")

                wav_files = []
                for root, _, files in os.walk(label_dir):
                    for file_name in files:
                        if file_name.lower().endswith(".wav"):
                            wav_files.append(os.path.join(root, file_name))

                for wav_path in sorted(wav_files):
                    self.samples.append((wav_path, self.label_to_id[label]))

        if not self.samples:
            raise RuntimeError(f"No wav samples found under {self.split_names} in {base_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        wav_path, label_id = self.samples[index]
        speech, sr = sf.read(wav_path)

        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        if sr != self.target_sr:
            speech = torchaudio.functional.resample(
                torch.tensor(speech, dtype=torch.float32),
                orig_freq=sr,
                new_freq=self.target_sr,
            ).numpy()

        speech = speech.astype(np.float32)
        return speech, label_id


class BatchCollator:
    def __init__(self, processor: Wav2Vec2FeatureExtractor):
        self.processor = processor

    def __call__(self, batch: Sequence[Tuple[np.ndarray, int]]) -> Dict[str, torch.Tensor]:
        speeches, labels = zip(*batch)
        inputs = self.processor(
            list(speeches),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class AttentionPooling(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        attn_hidden_size: int = 256,
        num_labels: int = 35,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_size, attn_hidden_size),
            nn.Tanh(),
            nn.Linear(attn_hidden_size, 1)
        )

        self.classifier = nn.Linear(input_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):

        scores = self.attention(hidden_states).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

        logits = self.classifier(pooled)

        if return_features:
            return logits, pooled

        return logits


class FrozenHubertBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = HubertModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def _downsample_attention_mask(self, attention_mask: torch.Tensor, sequence_length: int) -> torch.Tensor:
        input_lengths = attention_mask.sum(dim=1)
        output_lengths = self.backbone._get_feat_extract_output_lengths(input_lengths)
        time_index = torch.arange(sequence_length, device=attention_mask.device).unsqueeze(0)
        return time_index < output_lengths.unsqueeze(1)

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with torch.no_grad():
            outputs = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        hidden_states = outputs.last_hidden_state
        pooled_mask = None
        if attention_mask is not None:
            pooled_mask = self._downsample_attention_mask(attention_mask, hidden_states.size(1))
        return hidden_states, pooled_mask


@dataclass
class TrainConfig:
    data_dir: str = "/data/gsc_splited"
    model_name: str = "facebook/hubert-base-ls960"
    train_splits: Tuple[str, ...] = ("training", "rl_train", "testing")
    batch_size: int = 8
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.01
    attn_hidden_size: int = 256
    num_workers: int = 4
    save_path: str = "wav2vec_attention_pool_final.pt"
    grad_clip: float = 1.0


def build_label_mapping() -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(WANTED_WORDS)}


def train(config: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_id = build_label_mapping()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)
    backbone = FrozenHubertBackbone(config.model_name).to(device)
    attention_pooling = AttentionPooling(
        input_size=backbone.hidden_size,
        attn_hidden_size=config.attn_hidden_size,
        num_labels=len(WANTED_WORDS)
    ).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    train_dataset = GscFolderDataset(config.data_dir, config.train_splits, label_to_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=BatchCollator(processor),
    )

    optimizer = torch.optim.AdamW(
        attention_pooling.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.epochs + 1):
        attention_pooling.train()
        backbone.eval()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in train_bar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                hidden_states, pooled_mask = backbone(input_values=input_values, attention_mask=attention_mask)
            logits = attention_pooling(hidden_states, pooled_mask)
            loss = criterion(logits, labels)
            loss.backward()

            if config.grad_clip is not None and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(attention_pooling.parameters(), config.grad_clip)

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=-1)
            running_correct += (predictions == labels).sum().item()
            running_count += labels.size(0)

            train_bar.set_postfix({
                "loss": f"{running_loss / max(running_count, 1):.4f}",
                "acc": f"{running_correct / max(running_count, 1):.4f}",
            })

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    torch.save(
        {
            "attention_pooling_state_dict": attention_pooling.state_dict(),
            "backbone_model_name": config.model_name,
            "label_to_id": label_to_id,
            "config": config.__dict__,
        },
        config.save_path,
    )
    print(f"Saved final checkpoint to {config.save_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train HuBERT + attention pooling on GSC folders")
    parser.add_argument("--data-dir", type=str, default="/data/gsc_splited")
    parser.add_argument("--model-name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--train-splits", type=str, nargs="+", choices=["training", "rl_train", "validation", "testing"], default=["training", "rl_train"],
                    help="Data splits used to train the attention-pooling module. Multiple splits can be specified, e.g., '--train-splits training rl_train'. The testing and validation splits should not be used for training or model selection.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--attn-hidden-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="wav2vec_attention_pool_final.pt")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        train_splits=tuple(args.train_splits),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        attn_hidden_size=args.attn_hidden_size,
        num_workers=args.num_workers,
        save_path=args.save_path,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    train(parse_args())
