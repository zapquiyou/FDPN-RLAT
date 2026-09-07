import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import jsonlines
import json

import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemorySelector(nn.Module):
    def __init__(self, input_dim, num_flag_repeat=64, hidden_dim=128, mode='random', is_train=False):
        super().__init__()
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.num_flag_repeat = num_flag_repeat
        self.is_train = is_train
        self.memory_flag = torch.tensor([0.0, 1.0, 0.0]).to(device)

    def forward(self, x_candidate, x_memory=None, memory_size=500):
        new_memory_size = None
        log_prob = None
        new_memory = None
        if x_memory is None:
            new_memory_size = min(memory_size, x_candidate.shape[1])
            if self.mode == 'random':
                rand_indices = torch.randperm(x_candidate.shape[1], device=device)[:new_memory_size]
                new_memory = torch.index_select(x_candidate, dim=1, index=rand_indices).to(device)
            else:
                scores = self.encoder(x_candidate).squeeze(-1).squeeze(0)
                probs = F.softmax(scores, dim=0)
                dist = torch.distributions.Categorical(probs=probs)
                if self.is_train:
                    indices = dist.sample((new_memory_size,))
                else:
                    indices = torch.topk(probs, new_memory_size, dim=0).indices
                indices = indices.to(device)
                new_memory = torch.index_select(x_candidate, dim=1, index=indices).to(device)
                log_prob = dist.log_prob(indices).sum().to(device)

        else:
            new_memory_size = min(memory_size, x_candidate.shape[1] + x_memory.shape[1])
            x_total = torch.cat([x_candidate, x_memory], dim=1).to(device)

            if self.mode == 'random':
                rand_indices = torch.randperm(x_total.shape[1], device=device)[:new_memory_size]
                new_memory = torch.index_select(x_total, dim=1, index=rand_indices).to(device)
            else:
                scores = self.encoder(x_total).squeeze(-1).squeeze(0)
                probs = F.softmax(scores, dim=0)
                dist = torch.distributions.Categorical(probs=probs)
                if self.is_train:
                    indices = dist.sample((new_memory_size,))
                else:
                    indices = torch.topk(probs, new_memory_size, dim=0).indices
                indices = indices.to(device)
                new_memory = torch.index_select(x_total, dim=1, index=indices).to(device)
                log_prob = dist.log_prob(indices).sum().to(device)

        if new_memory_size < memory_size:
            padding_size = memory_size - new_memory_size
            if padding_size > 0:
                padding = torch.zeros((1, padding_size, x_candidate.shape[2]), dtype=x_candidate.dtype).to(device)
                new_memory = torch.cat([new_memory, padding], dim=1)

        memory_flags = self.memory_flag.repeat(self.num_flag_repeat).unsqueeze(0).expand(memory_size, -1).to(device)
        new_memory[:, :, -memory_flags.shape[1]:] = memory_flags

        return new_memory, log_prob, new_memory_size

class RLDataLoader:
    def __init__(self, feature_file, result_file, task):
        with open(result_file, 'r') as f:
            results = json.load(f)

        self.data = []

        if task == "stereo":
            with jsonlines.open(feature_file, mode='r') as reader:
                for feature, result in zip(reader, results['result']):
                    self.data.append({
                        "fused_features": torch.tensor(feature["fused_features"][0], dtype=torch.float32).to(device),
                        "result": result,
                    })

        else:
            with jsonlines.open(feature_file, mode='r') as reader:
                result_map = dict(zip(results['path'], results['result']))
                for feature in reader:
                    self.data.append({
                        "fused_features": torch.tensor(feature["fused_features"], dtype=torch.float32).to(device),
                        "result": result_map[feature['file']],
                    })

        self.unread_indices = list(range(len(self.data)))

    def get_random_batch(self, batch_size=1000):
        if len(self.unread_indices) == 0:
            raise ValueError("No unread data available.")

        selected_indices = random.sample(self.unread_indices, min(batch_size, len(self.unread_indices)))

        self.unread_indices = [idx for idx in self.unread_indices if idx not in selected_indices]

        num_false = sum(1 for idx in selected_indices if not self.data[idx]["result"])

        return [self.data[idx] for idx in selected_indices], num_false

    def is_all_data_read(self):
        return len(self.unread_indices) == 0

    def get_num_all_fault(self):
        return sum(1 for item in self.data if not item["result"])

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

class IncrementalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Invalid embed_dim for the number of heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, cache=None, use_cache=True):
        q = self.q_proj(query)

        if use_cache and cache is not None and "k" in cache and "v" in cache:
            prev_len = cache["k"].shape[1]
            new_k = self.k_proj(key[:, prev_len:, :])
            new_v = self.v_proj(value[:, prev_len:, :])
            
            k = torch.cat([cache["k"], new_k], dim=1)
            v = torch.cat([cache["v"], new_v], dim=1)

            cache["k"] = k
            cache["v"] = v
        else:
            k = self.k_proj(key)
            v = self.v_proj(value)
            if use_cache and cache is not None:
                cache["k"] = k
                cache["v"] = v

        def shape(x):
            bsz, seq_len, _ = x.size() 
            return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape(q)  
        k = shape(k)  
        v = shape(v)  

        q = q / (self.head_dim ** 0.5)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        batch_size, _, seq_len, _ = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, cache

class IncrementalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = IncrementalMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, cache=None, use_cache=True):
        attn_output, cache = self.self_attn(src, src, src, attn_mask=src_mask,
                                              key_padding_mask=src_key_padding_mask, cache=cache, use_cache=use_cache)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, cache

class IncrementalTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None, src_key_padding_mask=None, caches=None, use_cache=True):
        output = src
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = caches[i] if caches is not None and i < len(caches) else None
            output, new_cache = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask,
                                        cache=layer_cache, use_cache=use_cache)
            new_caches.append(new_cache)
        return output, new_caches

class PointerNetwork(nn.Module):
    def __init__(self, d_model, candidate_len, variant):
        super().__init__()
        self.candidate_len = candidate_len
        self.variant = variant

        if self.variant == "no_feedback":
            self.main_proj = nn.Linear(d_model, d_model)
            self.scorer = nn.Linear(d_model, 1)
            self.stop_head = nn.Sequential(
                nn.Linear(d_model * 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
            )
            self.stop_final = nn.Linear(candidate_len, 1)

        elif self.variant == "no_gate":
            self.main_proj = nn.Linear(d_model, d_model)
            self.feedback_proj = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model)
            )
            self.scorer = nn.Linear(d_model * 2, 1)
            self.stop_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
            )
            self.stop_final = nn.Linear(candidate_len, 1)

        else:
            self.main_proj = nn.Linear(d_model, d_model)
            self.feedback_proj = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model)
            )
            self.gate_layer = nn.Linear(d_model * 2, d_model)
            self.scorer = nn.Linear(d_model, 1)
            self.stop_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
            )
            self.stop_final = nn.Linear(candidate_len, 1)

    def forward(self, encoder_outputs, feedback_vector):
        if self.variant == "no_feedback":
            main_out = self.main_proj(encoder_outputs)
            item_scores = self.scorer(main_out).squeeze(-1).unsqueeze(0)
            stop_score = self.stop_head(main_out).squeeze(-1)
            stop_score = self.stop_final(stop_score).unsqueeze(0)
            scores = torch.cat([item_scores, stop_score], dim=-1)
            probs = F.softmax(scores, dim=-1)

        elif  self.variant == "no_gate":
            main_out = self.main_proj(encoder_outputs)
            feedback_out = self.feedback_proj(feedback_vector)

            concat = torch.cat([main_out, feedback_out], dim=-1)

            item_scores = self.scorer(concat).squeeze(-1).unsqueeze(0)
            stop_score = self.stop_head(concat).squeeze(-1)
            stop_score = self.stop_final(stop_score).unsqueeze(0)

            scores = torch.cat([item_scores, stop_score], dim=-1)
            probs = F.softmax(scores, dim=-1)

        else:
            main_out = self.main_proj(encoder_outputs)
            feedback_out = self.feedback_proj(feedback_vector)

            concat = torch.cat([main_out, feedback_out], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat))
            fused = gate * main_out + (1 - gate) * feedback_out

            item_scores = self.scorer(fused).squeeze(-1).unsqueeze(0)
            stop_score = self.stop_head(concat).squeeze(-1)
            stop_score = self.stop_final(stop_score).unsqueeze(0)

            scores = torch.cat([item_scores, stop_score], dim=-1)
            probs = F.softmax(scores, dim=-1)
        return probs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class TestCaseModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, candidate_len, variant, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.candidate_len = candidate_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = IncrementalTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = IncrementalTransformerEncoder(encoder_layer, num_layers)
        self.pointer_net = PointerNetwork(d_model, candidate_len, variant)

    def forward(self, x, y, caches=None, use_cache=True):
        if caches is None:
             caches = [{} for _ in range(len(self.encoder.layers))]
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        encoder_output, new_caches = self.encoder(x, caches=caches, use_cache=use_cache)

        candidate_output = encoder_output[:, :self.candidate_len, :]

        pointer_probs = self.pointer_net(candidate_output, y)
        return pointer_probs, new_caches

def get_reward(is_detected_list, num_fault, subdataset_len,
               br=20.0, gamma=0, punish=25, progress_bonus=0, rel_min=0):
    last = is_detected_list[-1]
    t = len(is_detected_list)

    L = max(1.0, float(subdataset_len))
    F = max(1.0, float(num_fault))
    hit = 1.0 if last is True else 0.0

    detected = sum(1 for v in is_detected_list if v)
    is_punish = 1.0 if detected < F else 0.0
    
    rel = max(rel_min, 1.0 - (float(t) - 1.0) / L)
    w = rel ** gamma

    progress = detected / F
    coverage_weight = 1.0 + progress_bonus * progress

    reward = (float(br) / F) * w * hit * coverage_weight - (1 - hit) * is_punish * punish / L

    return reward

def stop_reward(num_action, num_fault, num_undetected_fault, is_detected_list,
                subdataset_len,
                full_detect_bonus=0.0,
                stop_in_time_bonus=10.0,
                undetect_punish=15,
                alpha=1.0,
                beta=1.0, C="t_fin", D=0):
    T = int(num_action)
    L = max(1, int(subdataset_len))
    F = max(0, int(num_fault))
    u = max(0, int(num_undetected_fault))
    B, S, P = float(full_detect_bonus), float(stop_in_time_bonus), float(undetect_punish)

    if F == 0:
        r0 = max(0.0, (L - T + 1) / L)
        return S * (r0 ** float(beta))

    if u <= D:
        last_hit_idx = len(is_detected_list) - 1 - is_detected_list[::-1].index(True) if True in is_detected_list else None
        t_finish = last_hit_idx + 1 if last_hit_idx is not None else T
        if C == "t_fin":
            r = max(0.0, (L - T + 1) / max(1, L - t_finish))
        else:
            r = max(0.0, (L - T + 1) / max(1, L - int(C)))

        return B + S * (r ** float(beta))

    miss_ratio = float(u) / float(F)
    return - P * (miss_ratio ** float(alpha))

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return torch.tensor(discounted_rewards)

def train_rl_testcasemodel(model, optimizer, memory_selector, args, stage="hit"):
    model.train()
    if args.variant != "no_memory":
        memory_selector.eval()

    set_stage(model, stage, args.variant)

    region_flags = {"candidate": torch.tensor([1.0, 0.0, 0.0]),
                    "memory":    torch.tensor([0.0, 1.0, 0.0]),
                    "history":   torch.tensor([0.0, 0.0, 1.0])}
    
    if args.task == "stereo":
        model_under_test = ["IGEV_plusplus", "DEFOM-Stereo", "Monster", "Selective_Stereo"]
        num_train_steps = ["100000", "90000", "80000", "70000"]
    
    else:
        model_under_test = ["inception_resnet", "mobilenet_v2", "svdf_resnet", "tc_resnet"]
        num_train_steps = ["40000", "38000", "36000", "34000"]

    for epoch in range(args.num_epoches):
        model_name = model_under_test[epoch % (len(model_under_test) * len(num_train_steps)) % len(model_under_test)]
        num_steps = num_train_steps[epoch % (len(model_under_test) * len(num_train_steps)) // len(num_train_steps)]

        if args.num_epoches > 1:
            ratio = epoch / (args.num_epoches - 1)
        else:
            ratio = 1.0

        epsilon = args.epsilon_start + ratio * (args.epsilon_end - args.epsilon_start)
        epsilon = max(0.0, min(1.0, float(epsilon)))

        if args.task == "stereo":
            result_file = f"/data/StereoDatasets/results/{model_name}_rltrain_{num_steps}.json"
            feature_file = "/data/StereoDatasets/sceneflow_vit_fusion_features_rltrain.jsonl"
        else:
            result_file = f"/data/gsc_splited/result/{model_name}_{num_steps}_rl_train.json"
            feature_file = "/data/gsc_splited/result/gsc_wav2vec_features_rltrain.jsonl"

        rl_dataset = RLDataLoader(feature_file, result_file, args.task)

        if args.variant != "no_memory":
            memo = torch.zeros(args.memo_seq_len, args.input_dim + args.num_flag_feedback).to(device)
            memory_flags = region_flags["memory"].repeat(args.num_flag_repeat).unsqueeze(0).expand(args.memo_seq_len, -1).to(device)
            memo_with_flag = torch.cat([memo, memory_flags], dim=1)
            memory_len = 0

        for num_subdatasets in range(args.dataset_len // args.candidate_len):
            subdatasets, num_fault = rl_dataset.get_random_batch(args.candidate_len)
            if num_fault == 0 and stage == "hit":
                continue
            flag = region_flags["candidate"].repeat(args.num_flag_repeat).to(device)
            flag = torch.cat([torch.zeros(args.num_flag_feedback).to(device), flag], dim=0)

            feature_list = []
            for item in subdatasets:
                fused = item["fused_features"]
                fused_with_flag = torch.cat([fused, flag], dim=0)
                feature_list.append(fused_with_flag)
            x = torch.stack(feature_list)
            if args.variant != "no_memory":
                x = torch.cat([x, memo_with_flag], dim=0)
            x = x.unsqueeze(0)

            caches = None
            feedback_signals = torch.zeros(args.candidate_len, 1).to(device)
            log_probs, rewards, is_detected_list = [], [], []
            mask = torch.zeros(args.candidate_len + 1, dtype=torch.bool).to(device)
            if stage == "hit":
                mask[-1] = True
            num_action = 0

            for step in range(args.candidate_len):
                pointer_probs, caches = model(x, feedback_signals.unsqueeze(0), caches=caches, use_cache=False)
                num_action += 1

                masked_probs = pointer_probs[0].squeeze(0)

                masked_probs = masked_probs.clone()
                masked_probs[mask] = 0.0

                sum_prob = masked_probs.sum()
                if sum_prob.detach().item() <= 1e-12:
                    masked_probs = pointer_probs[0].squeeze(0)
                    if stage == "hit":
                        masked_probs = masked_probs.clone()
                        masked_probs[-1] = 0.0

                    masked_probs = masked_probs.clone()
                    masked_probs[mask] = 0.0
                    eps = 1e-8
                    masked_probs = masked_probs + eps
                    masked_probs = masked_probs.clone()
                    masked_probs[mask] = 0.0
                    sum_prob = masked_probs.sum()

                masked_probs = masked_probs / (sum_prob + 1e-12)

                valid_mask = ~mask
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

                if valid_indices.numel() == 0:
                    break

                pi_probs = masked_probs

                if stage == "hit" and epsilon > 0:
                    uniform_probs = torch.zeros_like(pi_probs)
                    uniform_probs[valid_indices] = 1.0 / valid_indices.numel()

                    behavior_probs = (1.0 - epsilon) * pi_probs + epsilon * uniform_probs
                    behavior_probs = behavior_probs / (behavior_probs.sum() + 1e-12)
                else:
                    behavior_probs = pi_probs

                dist = torch.distributions.Categorical(probs=behavior_probs)
                action = dist.sample()

                log_prob = dist.log_prob(action)

                log_probs.append(log_prob)
                chosen_idx = action.item()

                if chosen_idx == args.candidate_len:
                    num_detect_fault = int((feedback_signals[:args.candidate_len] > 0).sum().item())
                    num_undetected_fault = max(0, int(num_fault) - num_detect_fault)
                    R_stop = stop_reward(num_action, num_fault, num_undetected_fault,
                                         is_detected_list, args.candidate_len,
                                         full_detect_bonus=args.stop_full_detect_bonus,
                                         stop_in_time_bonus=args.stop_in_time_bonus,
                                         undetect_punish=args.stop_undetect_punish,
                                         alpha=args.stop_alpha, beta=args.stop_beta,
                                         C=args.stop_C, D=args.stop_D)
                    rewards.append(R_stop)
                    num_action -= 1
                    break
                else:
                    mask = mask.clone()
                    mask[chosen_idx] = True

                    feedback_signals = feedback_signals.clone()
                    if subdatasets[chosen_idx]["result"]:
                        feedback_signals[chosen_idx] = -1.0
                        is_detected_list.append(False)
                    else:
                        feedback_signals[chosen_idx] = 1.0
                        is_detected_list.append(True)

                    r_step = get_reward(is_detected_list, num_fault, args.candidate_len,
                                        br=args.hit_reward_base,
                                        gamma=args.hit_reward_gamma,
                                        punish=args.hit_miss_punish,
                                        progress_bonus=args.hit_progress_bonus,
                                        rel_min=args.hit_rel_min)
                    if stage == "stop":
                        r_step = 0
                    rewards.append(r_step)

                    new_feedback = torch.tensor([feedback_signals[chosen_idx].item()] * args.num_flag_feedback).to(device)
                    new_result = torch.cat([subdatasets[chosen_idx]["fused_features"], new_feedback], dim=0)
                    new_result = torch.cat([new_result, region_flags["history"].repeat(args.num_flag_repeat).to(device)], dim=0)
                    new_result = new_result.unsqueeze(0).unsqueeze(0)
                    x = torch.cat([x, new_result], dim=1)

                    num_detect_fault = int((feedback_signals[:args.candidate_len] > 0).sum().item())

                    if stage == "hit" and num_detect_fault >= int(num_fault):
                        break

            discounted_rewards = compute_discounted_rewards(rewards, gamma=args.reward_discount_gamma)
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
            else:
                discounted_rewards = discounted_rewards - discounted_rewards.mean()

            loss = 0
            for lp, R in zip(log_probs, discounted_rewards):
                loss += -lp * R
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.variant != "no_memory":
                if num_action > 0:
                    if memory_len == 0:
                        new_memory, _, memory_len = memory_selector(
                            x[:, args.candidate_len + args.memo_seq_len:, :].to(device),
                            memory_size=args.memo_seq_len)
                    else:
                        new_memory, _, memory_len = memory_selector(
                            x[:, args.candidate_len + args.memo_seq_len:, :].to(device),
                            x_memory=x[:, args.candidate_len:args.candidate_len + memory_len, :].to(device),
                            memory_size=args.memo_seq_len)
                    memo_with_flag = new_memory.squeeze(0)

            print(f"[{stage}] Epoch {epoch + 1}/{args.num_epoches}, subdatasets {num_subdatasets + 1}/{args.dataset_len // args.candidate_len} completed")
            print(f"action {num_action}, reward {sum(rewards)}, num_fault {num_fault}, "
                  f"num_detected_fault {int((feedback_signals[:args.candidate_len] > 0).sum().item())}, "
                  f"loss {loss.item()}, detected_positions: {[i for i, v in enumerate(is_detected_list) if v]}")

def train_rl_memoryselector(model, memory_selector, selector_optimizer, args):
    memory_selector.train()    
    model.eval()

    region_flags = {"candidate": torch.tensor([1.0, 0.0, 0.0]),
                    "memory":    torch.tensor([0.0, 1.0, 0.0]),
                    "history":   torch.tensor([0.0, 0.0, 1.0])
                    }
    
    if args.task == "stereo":
        model_under_test = ["IGEV_plusplus", "DEFOM-Stereo", "Monster", "Selective_Stereo"]
        num_train_steps = ["100000", "90000", "80000", "70000"]
    else:
        model_under_test = ["inception_resnet", "mobilenet_v2", "svdf_resnet", "tc_resnet"]
        num_train_steps = ["40000", "38000", "36000", "34000"]
    
    for epoch in range(args.num_epoches):
        model_name = model_under_test[epoch % (len(model_under_test) * len(num_train_steps)) % len(model_under_test)]
        num_steps = num_train_steps[epoch % (len(model_under_test) * len(num_train_steps)) // len(num_train_steps)]

        if args.task == "stereo":
            result_file = f"/data/StereoDatasets/results/{model_name}_rltrain_{num_steps}.json"
            feature_file = "/data/StereoDatasets/sceneflow_vit_fusion_features_rltrain.jsonl"
        else:
            result_file = f"/data/gsc_splited/result/{model_name}_{num_steps}_rl_train.json"
            feature_file = "/data/gsc_splited/result/gsc_wav2vec_features_rltrain.jsonl"

        rl_dataset = RLDataLoader(feature_file, result_file, args.task)
        memo = torch.zeros(args.memo_seq_len, args.input_dim + args.num_flag_feedback).to(device)
        memory_flags = region_flags["memory"].repeat(args.num_flag_repeat).unsqueeze(0).expand(args.memo_seq_len, -1).to(device)
        memo_with_flag = torch.cat([memo, memory_flags], dim=1)
        memory_len = 0
        pre_reward = None
        pre_log_prob = None

        for num_subdatasets in range(args.dataset_len//args.candidate_len):
            subdatasets, num_fault = rl_dataset.get_random_batch(args.candidate_len)
            if num_fault == 0:
                continue
            flag = region_flags["candidate"].repeat(args.num_flag_repeat).to(device)
            flag = torch.cat([torch.zeros(args.num_flag_feedback).to(device), flag], dim=0)
            feature_list = []
            for item in subdatasets:
                fused = item["fused_features"]
                fused_with_flag = torch.cat([fused, flag], dim=0)
                feature_list.append(fused_with_flag)
            x = torch.stack(feature_list)
            x = torch.cat([x, memo_with_flag], dim=0)
            x = x.unsqueeze(0)

            caches = None
            feedback_signals = torch.zeros(args.candidate_len, 1).to(device)
            rewards, is_detected_list = [], []
            mask = torch.zeros(args.candidate_len + 1, dtype=torch.bool).to(device)
            num_action = 0
            for step in range(args.candidate_len):
                pointer_probs, caches = model(x, feedback_signals.unsqueeze(0), caches=caches, use_cache=True)
                num_action += 1

                masked_probs = pointer_probs[0].squeeze(0).clone()
                masked_probs[mask] = 0
                masked_probs = masked_probs / masked_probs.sum()

                dist = torch.distributions.Categorical(probs=masked_probs)
                action = dist.sample()
                chosen_idx = action.item()

                if chosen_idx == args.candidate_len:
                    num_detect_fault = sum(1 for idx in range(args.candidate_len) if feedback_signals[idx] > 0)
                    num_undetected_fault = num_fault - num_detect_fault
                    rewards.append(stop_reward(num_action, num_fault, num_undetected_fault, is_detected_list, args.candidate_len,
                                               full_detect_bonus=args.stop_full_detect_bonus,
                                               stop_in_time_bonus=args.stop_in_time_bonus,
                                               undetect_punish=args.stop_undetect_punish,
                                               alpha=args.stop_alpha,beta=args.stop_beta,
                                               C=args.stop_C, D=args.stop_D))
                    num_action -= 1
                    break
                else:
                    mask = mask.clone()
                    mask[chosen_idx] = True

                    feedback_signals = feedback_signals.clone()
                    if subdatasets[chosen_idx]["result"]:
                        feedback_signals[chosen_idx] = -1.0
                        is_detected_list.append(False)
                    else:
                        feedback_signals[chosen_idx] = 1.0
                        is_detected_list.append(True)                    

                    reward = get_reward(is_detected_list, num_fault, args.candidate_len,
                                        br=args.hit_reward_base,
                                        gamma=args.hit_reward_gamma,
                                        punish=args.hit_miss_punish,
                                        progress_bonus=args.hit_progress_bonus,
                                        rel_min=args.hit_rel_min)
                    rewards.append(reward)

                    new_feedback = torch.tensor([feedback_signals[chosen_idx].item()] * args.num_flag_feedback).to(device)
                    new_result = torch.cat([subdatasets[chosen_idx]["fused_features"], new_feedback], dim=0)
                    new_result = torch.cat([new_result, region_flags["history"].repeat(args.num_flag_repeat).to(device)], dim=0)
                    new_result = new_result.unsqueeze(0).unsqueeze(0)
                    x = torch.cat([x, new_result], dim=1)

            discounted_rewards = compute_discounted_rewards(rewards, gamma=args.reward_discount_gamma)
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
            else:
                discounted_rewards = discounted_rewards - discounted_rewards.mean()

            if num_action > 0:
                if memory_len == 0:
                    pre_reward = sum(rewards)
                    new_memory, pre_log_prob, memory_len = memory_selector(x[:, args.candidate_len + args.memo_seq_len:, :].to(device), memory_size=args.memo_seq_len)

                else:
                    pre_reward = sum(rewards)
                    loss = - pre_log_prob * pre_reward
                    selector_optimizer.zero_grad()
                    loss.backward()
                    selector_optimizer.step()

                    new_memory, pre_log_prob, memory_len = memory_selector(x[:, args.candidate_len + args.memo_seq_len:, :].to(device), x_memory=x[:, args.candidate_len:args.candidate_len + memory_len, :].to(device), memory_size=args.memo_seq_len)

                memo_with_flag = new_memory.squeeze(0)

            print(f"Epoch {epoch + 1}/{args.num_epoches}, subdatasets {num_subdatasets + 1}/{args.dataset_len // args.candidate_len} completed")
            print(f"action {num_action}, reward {sum(rewards)}, num_fault {num_fault}, num_detected_fault {sum(1 for idx in range(args.candidate_len) if feedback_signals[idx] > 0)}")

def get_hit_params(model: nn.Module, variant):
    if variant == "no_feedback":
        modules = [
            model.input_proj,
            model.encoder,
            model.pointer_net.main_proj,
            model.pointer_net.scorer,
        ]
    elif variant == "no_gate":
        modules = [
            model.input_proj,
            model.encoder,
            model.pointer_net.main_proj,
            model.pointer_net.feedback_proj,
            model.pointer_net.scorer,
        ]
    else:
        modules = [
            model.input_proj,
            model.encoder,
            model.pointer_net.main_proj,
            model.pointer_net.feedback_proj,
            model.pointer_net.gate_layer,
            model.pointer_net.scorer,
        ]
    for m in modules:
        for p in m.parameters():
            yield p

def get_stop_params(model: nn.Module):
    modules = [
        model.pointer_net.stop_head,
        model.pointer_net.stop_final,
    ]
    for m in modules:
        for p in m.parameters():
            yield p

def set_stage(model: nn.Module, stage: str, variant):
    for p in model.parameters():
        p.requires_grad = False

    if stage == "hit":
        for p in get_hit_params(model, variant):
            p.requires_grad = True
    elif stage == "stop":
        for p in get_stop_params(model):
            p.requires_grad = True
    elif stage == "joint":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")


def train_rl(model, memory_selector, memory_optimizer, args):

    opt_hit  = optim.Adam(get_hit_params(model, args.variant), lr=args.learning_rate)
    opt_stop = optim.Adam(get_stop_params(model), lr=args.learning_rate)

    print("Start RL...")
    print(
        "parameter: "
        f"hit_reward_base={args.hit_reward_base}, "
        f"hit_reward_gamma={args.hit_reward_gamma}, "
        f"hit_miss_punish={args.hit_miss_punish}, "
        f"hit_progress_bonus={args.hit_progress_bonus}, "
        f"hit_rel_min={args.hit_rel_min}, "
        f"stop_full_detect_bonus={args.stop_full_detect_bonus}, "
        f"stop_in_time_bonus={args.stop_in_time_bonus}, "
        f"stop_undetect_punish={args.stop_undetect_punish}, "
        f"stop_alpha={args.stop_alpha}, "
        f"stop_beta={args.stop_beta}, "
        f"reward_discount_gamma={args.reward_discount_gamma}"
    )
    if args.variant != "no_memory":
        memory_selector.mode = 'random'
        memory_selector.is_train = False
    train_rl_testcasemodel(model, opt_hit, memory_selector, args, stage="hit")

    train_rl_testcasemodel(model, opt_stop, memory_selector, args, stage="stop")

    if args.variant != "no_memory":
        memory_selector.mode = 'learnable'
        memory_selector.is_train = True
        train_rl_memoryselector(model, memory_selector, memory_optimizer, args)
        
    if args.variant != "no_memory":
        memory_selector.mode = 'learnable'
        memory_selector.is_train = False
    opt_hit  = optim.Adam(get_hit_params(model, args.variant), lr=args.learning_rate / 10)
    opt_stop = optim.Adam(get_stop_params(model), lr=args.learning_rate / 10)
    train_rl_testcasemodel(model, opt_hit, memory_selector, args, stage="hit")

    train_rl_testcasemodel(model, opt_stop, memory_selector, args, stage="stop")
    
    if args.variant == "no_memory":
        torch.save({"test_case_model": model.state_dict()}, args.model_path)
    else:
        torch.save({
            "test_case_model": model.state_dict(),
            "memory_selector": memory_selector.state_dict()
        }, args.model_path)

def eval_rl(model, memory_selector, args):
    writer = pd.ExcelWriter(f'FDPN-RLAT_{args.variant}_{args.task}_{args.random_seed}.xlsx', engine='xlsxwriter')
    
    model.eval()
    if args.variant != "no_memory":
        memory_selector.eval()
    region_flags = {"candidate": torch.tensor([1.0, 0.0, 0.0]),
                    "memory":    torch.tensor([0.0, 1.0, 0.0]),
                    "history":   torch.tensor([0.0, 0.0, 1.0])
                    }
    
    if args.task == "stereo":
        model_under_test = ["IGEV_plusplus", "DEFOM-Stereo", "Monster", "Selective_Stereo"]
        num_train_steps = ["100000", "90000", "80000", "70000"]
    else:
        model_under_test = ["inception_resnet", "mobilenet_v2", "svdf_resnet", "tc_resnet"]
        num_train_steps = ["40000", "38000", "36000", "34000"]

    for mut in model_under_test:
        for num_steps in num_train_steps:
            if args.task == "stereo":
                result_file = f"/data/StereoDatasets/results/{mut}_rltest_{num_steps}.json"
                feature_file = "/data/StereoDatasets/sceneflow_vit_fusion_features_rltest.jsonl"
            else:
                result_file = f"/data/gsc_splited/result/{mut}_{num_steps}_testing.json"
                feature_file = "/data/gsc_splited/result/gsc_wav2vec_features_rltest.jsonl"

            rl_dataset = RLDataLoader(feature_file, result_file, args.task)
            num_all_fault = rl_dataset.get_num_all_fault()

            if args.variant != "no_memory":
                memo = torch.zeros(args.memo_seq_len, args.input_dim + args.num_flag_feedback).to(device)

                memory_flags = region_flags["memory"].repeat(args.num_flag_repeat).unsqueeze(0).expand(args.memo_seq_len, -1).to(device)

                memo_with_flag = torch.cat([memo, memory_flags], dim=1)
                memory_len = 0

            total_step = 0
            total_detected_fault = 0
            FDT25 = None
            FDT50 = None
            FDT75 = None
            num_tce_def = int(0.25 * len(rl_dataset))
            TCE25 = None

            run_records = []
  
            for num_subdatasets in range(args.test_dataset_len//args.candidate_len):

                subdatasets, num_fault = rl_dataset.get_random_batch(args.candidate_len)
                flag = region_flags["candidate"].repeat(args.num_flag_repeat).to(device)
                flag = torch.cat([torch.zeros(args.num_flag_feedback).to(device), flag], dim=0)
                feature_list = []
                for item in subdatasets:
                    fused = item["fused_features"]
                    fused_with_flag = torch.cat([fused, flag], dim=0)
                    feature_list.append(fused_with_flag)
                x = torch.stack(feature_list)
                if args.variant != "no_memory":
                    x = torch.cat([x, memo_with_flag], dim=0)
                x = x.unsqueeze(0)

                caches = None
                feedback_signals = torch.zeros(args.candidate_len, 1).to(device)
                mask = torch.zeros(args.candidate_len + 1, dtype=torch.bool).to(device)
                num_action = 0

                for step in range(args.candidate_len):
                    pointer_probs, caches = model(x, feedback_signals.unsqueeze(0), caches=caches, use_cache=True)
                    num_action += 1
                    total_step += 1

                    masked_probs = pointer_probs[0].squeeze(0).clone()
                    masked_probs[mask] = 0
                    masked_probs = masked_probs / masked_probs.sum()

                    chosen_idx = torch.argmax(masked_probs).item()

                    if chosen_idx == args.candidate_len:
                        num_action -= 1
                        total_step -= 1
                        break
                    else:
                        mask = mask.clone()
                        mask[chosen_idx] = True

                        feedback_signals = feedback_signals.clone()
                        if subdatasets[chosen_idx]["result"]:
                            feedback_signals[chosen_idx] = -1.0
                        else:
                            feedback_signals[chosen_idx] = 1.0
                            total_detected_fault += 1                  

                        new_feedback = torch.tensor([feedback_signals[chosen_idx].item()] * args.num_flag_feedback).to(device)
                        new_result = torch.cat([subdatasets[chosen_idx]["fused_features"], new_feedback], dim=0)
                        new_result = torch.cat([new_result, region_flags["history"].repeat(args.num_flag_repeat).to(device)], dim=0)
                        new_result = new_result.unsqueeze(0).unsqueeze(0)

                        x = torch.cat([x, new_result], dim=1)

                        if total_detected_fault == int(0.25 * num_all_fault) and FDT25 is None:
                            FDT25 = total_step

                        if total_detected_fault == int(0.50 * num_all_fault) and FDT50 is None:
                            FDT50 = total_step
                        
                        if total_detected_fault == int(0.75 * num_all_fault) and FDT75 is None:
                            FDT75 = total_step

                        if total_step == num_tce_def and TCE25 is None:
                            TCE25 = total_detected_fault

                        run_records.append({
                            "step": total_step,
                            "detected_fault": total_detected_fault,
                        })

                if args.variant != "no_memory":
                    if num_action > 0:
                        if memory_len == 0:
                            new_memory, _, memory_len = memory_selector(x[:, args.candidate_len + args.memo_seq_len:, :].to(device), memory_size=args.memo_seq_len)

                        else:
                            new_memory, _, memory_len = memory_selector(x[:, args.candidate_len + args.memo_seq_len:, :].to(device), x_memory=x[:, args.candidate_len:args.candidate_len + memory_len, :].to(device), memory_size=args.memo_seq_len)

                        memo_with_flag = new_memory.squeeze(0)

            print(f"mut: {mut}, num_steps: {num_steps}, num_all_fault: {num_all_fault}, FDT25: {FDT25}, FDT50: {FDT50}, FDT75: {FDT75}, TCE: {TCE25}")
            df = pd.DataFrame(run_records)
            sheet_name = f"{mut}_{num_steps}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL-based Test Case Model")

    parser.add_argument("--input_dim", type=int, default=1536, help="Input feature dimension")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer internal dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--selector_hidden_dim", type=int, default=128, help="Hidden dimension of the selector")

    parser.add_argument("--num_epoches", type=int, default=48, help="Number of training epochs")
    parser.add_argument("--dataset_len", type=int, default=15000, help="Length of the training dataset")
    parser.add_argument("--test_dataset_len", type=int, default=5000, help="Length of the test dataset")
    parser.add_argument("--candidate_len", type=int, default=500, help="Length of the candidate region")
    parser.add_argument("--memo_seq_len", type=int, default=200, help="Length of the memory region")

    parser.add_argument("--num_flag_feedback", type=int, default=64, help="Dimension of the feedback signal")
    parser.add_argument("--num_flag_repeat", type=int, default=64, help="Number of flag repetitions")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epsilon_start", type=float, default=0, help="Initial epsilon for hit sampling")
    parser.add_argument("--epsilon_end", type=float, default=0, help="Final epsilon for hit sampling")

    parser.add_argument("--hit_reward_base", type=float, default=20.0, help="Base hit reward, corresponding to br in get_reward")
    parser.add_argument("--hit_reward_gamma", type=float, default=0, help="Position decay exponent for the hit reward, corresponding to gamma in get_reward")
    parser.add_argument("--hit_miss_punish", type=float, default=25, help="Penalty coefficient for missing a fault, corresponding to punish in get_reward")
    parser.add_argument("--hit_progress_bonus", type=float, default=0.0, help="Reward coefficient for detection progress, corresponding to progress_bonus in get_reward")
    parser.add_argument("--hit_rel_min", type=float, default=0.0, help="Lower bound of the position decay factor, corresponding to rel_min in get_reward")

    parser.add_argument("--stop_full_detect_bonus", type=float, default=0.0, help="Base stopping reward after all faults are detected")
    parser.add_argument("--stop_in_time_bonus", type=float, default=20.0, help="Reward coefficient for timely stopping")
    parser.add_argument("--stop_undetect_punish", type=float, default=15.0, help="Penalty coefficient for stopping with undetected faults")
    parser.add_argument("--stop_alpha", type=float, default=1.0, help="Penalty exponent for the undetected-fault ratio")
    parser.add_argument("--stop_beta", type=float, default=1.0, help="Reward exponent for timely stopping")
    parser.add_argument("--stop_C", type=str, default="t_fin", help="Baseline expected stopping step, either t_fin or an integer")
    parser.add_argument("--stop_D", type=float, default=0.0, help="Baseline number of allowed undetected faults")

    parser.add_argument("--reward_discount_gamma", type=float, default=0.99, help="Gamma used in discounted return calculation")

    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Running mode: train or eval")
    parser.add_argument("--task", type=str, choices=["stereo", "gsc"], default="stereo", help="Task type")
    parser.add_argument("--variant", type=str, choices=["full", "no_memory", "no_feedback", "no_gate"], default="full", help="Model variant")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--model_path", type=str, default="./model.pth", help="Path to the model checkpoint")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.stop_C != "t_fin":
        try:
            args.stop_C = int(args.stop_C)
        except ValueError:
            raise ValueError("args.stop_C must be 't_fin' or an integer")
    
    if args.mode == "train":
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        model = TestCaseModel(args.input_dim + args.num_flag_feedback + args.num_flag_repeat * 3, 
                            args.d_model, args.nhead, args.num_layers, args.candidate_len, args.variant)
        model = model.to(device)

        if args.variant == "no_memory":
            memory_selector = None
            selector_optimizer = None

        else:
            memory_selector = MemorySelector(args.input_dim + args.num_flag_feedback + args.num_flag_repeat * 3, 
                                            num_flag_repeat=args.num_flag_repeat, 
                                            hidden_dim=args.selector_hidden_dim, mode='random')
            memory_selector = memory_selector.to(device)

            selector_optimizer = optim.Adam(memory_selector.parameters(), lr=args.learning_rate)
            
        train_rl(model, memory_selector, selector_optimizer, args)
        
    elif args.mode == "eval":
        random.seed(args.random_seed)
        checkpoint = torch.load(args.model_path)
        model = TestCaseModel(args.input_dim + args.num_flag_feedback + args.num_flag_repeat * 3, 
                            args.d_model, args.nhead, args.num_layers, args.candidate_len, args.variant)
        model.load_state_dict(checkpoint["test_case_model"])
        model = model.to(device)

        if args.variant == "no_memory":
            memory_selector = None
        
        else:
            memory_selector = MemorySelector(args.input_dim + args.num_flag_feedback + args.num_flag_repeat * 3, 
                                            num_flag_repeat=args.num_flag_repeat, 
                                            hidden_dim=args.selector_hidden_dim, mode='learnable', is_train=False)
            memory_selector.load_state_dict(checkpoint["memory_selector"])
            memory_selector = memory_selector.to(device)

        eval_rl(model, memory_selector, args)
    else:
        raise ValueError("-mode should be 'train' or 'eval'.")
        
