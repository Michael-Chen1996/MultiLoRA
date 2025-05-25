import json
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model

# 加载数据
with open('dialogue_data.json') as f:
    data = json.load(f)

# 数据预处理
texts = [f"{d['customer']} {d['manager']}" for d in data]
emotions = [d['emotion'] for d in data]
request_fulfilled = [d['request_fulfilled'] for d in data]
question_types = [d['question_type'] for d in data]

# 划分训练集和测试集
train_texts, test_texts, train_emotions, test_emotions, train_fulfilled, test_fulfilled, train_types, test_types = train_test_split(
    texts, emotions, request_fulfilled, question_types, test_size=0.2, random_state=42)

# 创建数据集
train_dataset = Dataset.from_dict({'text': train_texts, 'emotion': train_emotions, 'fulfilled': train_fulfilled, 'type': train_types})
test_dataset = Dataset.from_dict({'text': test_texts, 'emotion': test_emotions, 'fulfilled': test_fulfilled, 'type': test_types})

# 加载预训练模型和tokenizer
model_name = '/root/ELSE/huggingface/models/bge-large-zh-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 只加载backbone，不带分类头
backbone = AutoModel.from_pretrained(model_name).to(device)

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['query', 'value'],
    lora_dropout=0.1,
    bias='none',
    task_type='SEQ_CLS'
)
backbone = get_peft_model(backbone, lora_config)
backbone.print_trainable_parameters()

# 定义多任务分类头
class MultiTaskHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.emotion_head = torch.nn.Linear(hidden_size, 4)      # 4分类
        self.fulfilled_head = torch.nn.Linear(hidden_size, 1)    # 2分类（用BCE）
        self.type_head = torch.nn.Linear(hidden_size, 11)        # 11分类
    def forward(self, pooled):
        emotion_logits = self.emotion_head(pooled)
        fulfilled_logits = self.fulfilled_head(pooled).view(-1)
        type_logits = self.type_head(pooled)
        return emotion_logits, fulfilled_logits, type_logits

# 定义完整模型
class MultiTaskModel(torch.nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.heads = MultiTaskHead(hidden_size)
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取CLS token
        if hasattr(outputs, 'last_hidden_state'):
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = outputs[0][:, 0]
        return self.heads(pooled)

hidden_size = backbone.config.hidden_size
model = MultiTaskModel(backbone, hidden_size).to(device)

# 训练设置
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# 应用预处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 定义collate_fn函数
def collate_fn(batch):
    return {
        'input_ids': torch.stack([torch.tensor(item['input_ids']).to(device) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask']).to(device) for item in batch]),
        'emotion': torch.tensor([{'neutral': 0, 'angry': 1, 'happy': 2, 'frustrated': 3}[item['emotion']] for item in batch]).to(device),
        'fulfilled': torch.tensor([item['fulfilled'] for item in batch], dtype=torch.float).to(device),
        'type': torch.tensor([{'loan': 0, 'interest_rate': 1, 'savings_account': 2, 'fraud': 3, 'investment': 4, 'online_banking': 5, 'credit_card': 6, 'loan_repayment': 7, 'insurance': 8, 'account_update': 9, 'other': 10}[item['type']] for item in batch]).to(device)
    }

# 创建DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

# 损失函数
emotion_loss_fn = torch.nn.CrossEntropyLoss()
fulfilled_loss_fn = torch.nn.BCEWithLogitsLoss()
type_loss_fn = torch.nn.CrossEntropyLoss()

# 训练参数
num_epochs = 3

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        emotion_logits, fulfilled_logits, type_logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        # 多任务损失计算
        emotion_loss = emotion_loss_fn(emotion_logits, batch['emotion'])
        fulfilled_loss = fulfilled_loss_fn(fulfilled_logits, batch['fulfilled'].float())
        type_loss = type_loss_fn(type_logits, batch['type'])
        loss = emotion_loss + fulfilled_loss + type_loss
        # 反向传播
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')