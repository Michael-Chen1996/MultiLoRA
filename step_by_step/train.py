import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer
from model import MultiTaskLoRAModel
from tqdm import tqdm

# 1. 伪造多任务数据集
class MultiTaskDataset(Dataset):
    def __init__(self, tokenizer, texts, labels1, labels2, max_length=64, batch_size=8):
        # 分批tokenize，显示进度条
        encodings = {'input_ids': [], 'attention_mask': []}
        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
            batch_texts = texts[i:i+batch_size]
            batch_enc = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            encodings['input_ids'].extend(batch_enc['input_ids'].tolist())
            encodings['attention_mask'].extend(batch_enc['attention_mask'].tolist())
        self.encodings = encodings
        self.labels1 = labels1
        self.labels2 = labels2
    def __len__(self):
        return len(self.labels1)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label1'] = torch.tensor(self.labels1[idx])
        item['label2'] = torch.tensor(self.labels2[idx])
        return item

# 2. 初始化参数
model_name = "BAAI/bge-large-en"
num_labels_list = [4, 2]
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 构造伪数据
texts = [f"示例文本{i}" for i in range(100)]
labels1 = [i % 4 for i in range(100)]  # 4类
labels2 = [i % 2 for i in range(100)]  # 2类
dataset = MultiTaskDataset(tokenizer, texts, labels1, labels2)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. 初始化模型
model = MultiTaskLoRAModel(model_name, num_labels_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 5. 优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# 6. 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label1 = batch['label1'].to(device)
        label2 = batch['label2'].to(device)

        optimizer.zero_grad()
        logits1, logits2 = model(input_ids, attention_mask)
        loss1 = loss_fn(logits1, label1)
        loss2 = loss_fn(logits2, label2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
