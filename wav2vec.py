import json
import jsonlines
import os

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor


WANTED_WORDS = 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes'.split(',')


def resolve_audio_path(base_dir: str, split_name: str) -> str:
    if os.path.isabs(split_name):
        return split_name
    return os.path.join(base_dir, split_name)


class GscFolderDataset:
    def __init__(self, base_dir: str, split_names, label_to_id, target_sr: int = 16000):
        self.base_dir = base_dir
        self.split_names = list(split_names)
        self.label_to_id = label_to_id
        self.target_sr = target_sr
        self.samples = []

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

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


class AttentionPooling(nn.Module):
    def __init__(self, input_size: int = 768, attn_hidden_size: int = 256, num_labels: int = 35):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attn_hidden_size),
            nn.Tanh(),
            nn.Linear(attn_hidden_size, 1),
        )
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
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

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
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


class FeatureExtractor:
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: str = None,
        attention_pool_ckpt: str = "wav2vec_attention_pool_final.pt",
        attn_hidden_size: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.backbone = FrozenHubertBackbone(model_name).to(self.device)
        self.attention_pool = None
        self.target_sr = 16000

        if attention_pool_ckpt and os.path.exists(attention_pool_ckpt):
            checkpoint = torch.load(attention_pool_ckpt, map_location=self.device)
            ckpt_config = checkpoint.get("config", {})
            attn_hidden_size = ckpt_config.get("attn_hidden_size", attn_hidden_size)

            self.attention_pool = AttentionPooling(
                input_size=self.backbone.hidden_size,
                attn_hidden_size=attn_hidden_size,
                num_labels=len(WANTED_WORDS),
            ).to(self.device)

            state_dict = checkpoint.get("attention_pooling_state_dict", checkpoint)
            self.attention_pool.load_state_dict(state_dict)
            self.attention_pool.eval()
        else:
            raise FileNotFoundError(f"Attention pooling checkpoint not found: {attention_pool_ckpt}")

    def _encode(self, wav_path: str):
        speech, sr = sf.read(wav_path)
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        if sr != self.target_sr:
            speech = torchaudio.functional.resample(
                torch.tensor(speech, dtype=torch.float32), orig_freq=sr, new_freq=self.target_sr
            ).numpy()
            sr = self.target_sr

        speech = speech.astype("float32")

        inputs = self.processor(
            speech,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.inference_mode():
            hidden_states, pooled_mask = self.backbone(input_values, attention_mask=attention_mask)

        return hidden_states, pooled_mask

    def _apply_pooling(self, hidden_states: torch.Tensor, pooled_mask: torch.Tensor, pooling: str):
        if pooling == "mean":
            return hidden_states.mean(dim=1).squeeze(0)
        elif pooling == "max":
            return hidden_states.max(dim=1)[0].squeeze(0)
        elif pooling == "mean_std":
            mean = hidden_states.mean(dim=1)
            std = hidden_states.std(dim=1)
            return torch.cat([mean, std], dim=-1).squeeze(0)
        elif pooling == "mean_max":
            mean = hidden_states.mean(dim=1)
            mx = hidden_states.max(dim=1)[0]
            return torch.cat([mean, mx], dim=-1).squeeze(0)
        elif pooling == "attention_pool":
            if self.attention_pool is None:
                raise RuntimeError(
                    "attention_pooling checkpoint is not loaded; set attention_pool_ckpt to a valid file"
                )
            with torch.inference_mode():
                _, pooled = self.attention_pool(hidden_states, mask=pooled_mask, return_features=True)
            return pooled.squeeze(0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def extract(self, wav_path: str, base_pooling: str = "mean_max"):
        hidden_states, pooled_mask = self._encode(wav_path)
        features = self._apply_pooling(hidden_states, pooled_mask, base_pooling)

        return features

data_dir = '/data/gsc_splited/'
wanted_words = 'visual,wow,learn,backward,dog,two,left,happy,nine,go,up,bed,stop,one,zero,tree,seven,on,four,bird,right,eight,no,six,forward,house,marvin,sheila,five,off,three,down,cat,follow,yes'.split(',')
dirs = ['testing', 'rl_train']

wanted_words_index = {}
for index, wanted_word in enumerate(wanted_words):
    wanted_words_index[wanted_word] = index

data_index = {}
for set_index in dirs:
    dataset = GscFolderDataset(data_dir, [set_index], wanted_words_index)
    data_index[set_index] = [
        {'label': wanted_words[item_label], 'file': wav_path}
        for wav_path, item_label in dataset
    ]

extractor = FeatureExtractor()

output_file = ['/data/gsc_splited/result/gsc_wav2vec_features_rltest.jsonl', '/data/gsc_splited/result/gsc_wav2vec_features_rltrain.jsonl']

for set_index, out_file in zip(dirs, output_file):
    with jsonlines.open(out_file, mode='w') as writer:
        for item in data_index[set_index]:
            label = item['label']
            file_path = item['file']
            feature = extractor.extract(file_path, base_pooling='attention_pool')
            record = {
                'label': label,
                'file': file_path,
                'fused_features': feature.tolist()
            }
            writer.write(record)
            print(f"Processed {file_path}")
    print(f"Finished writing features to {out_file}, total {len(data_index[set_index])} samples.")


