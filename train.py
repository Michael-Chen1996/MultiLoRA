import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
model_name = 'BAAI/bge-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['query', 'value'],
    lora_dropout=0.1,
    bias='none',
    task_type='SEQ_CLS'
)
model = get_peft_model(model, lora_config)

# 训练设置
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# 应用预处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 创建DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

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
        
        # 前向传播
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        
        # 多任务损失计算
        emotion_loss = emotion_loss_fn(outputs.logits[:, :5], batch['emotion'])
        fulfilled_loss = fulfilled_loss_fn(outputs.logits[:, 5], batch['fulfilled'].float())
        type_loss = type_loss_fn(outputs.logits[:, 6:], batch['type'])
        
        loss = emotion_loss + fulfilled_loss + type_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')