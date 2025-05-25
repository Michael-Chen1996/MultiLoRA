import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoTokenizer

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = self.alpha / self.r

    def forward(self, x):
        lora_out = (self.lora_B @ self.lora_A)  # (out_features, in_features)
        return self.dropout(x) @ lora_out.T * self.scaling

class MultiTaskLoRAModel(nn.Module):
    def __init__(self, model_name, num_labels_list, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        # 为每个任务单独分配一个LoRA头
        self.loras = nn.ModuleList([
            LoRALinear(hidden_size, hidden_size, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            for _ in num_labels_list
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for num_labels in num_labels_list
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # 取CLS向量
        # 每个任务用自己的LoRA头和分类头
        logits = [
            clf(pooled + lora(pooled))
            for clf, lora in zip(self.classifiers, self.loras)
        ]
        return logits
