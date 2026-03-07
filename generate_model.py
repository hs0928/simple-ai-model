import torch
import torch.nn as nn
from torch.nn import functional as F
from google.colab import drive
import os

drive.mount('/content/drive')
path = "/content/drive/MyDrive/Colab Notebooks/CreateLLM_Data/DataSet_3.txt"

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)

batch_size = 32
block_size = 64
n_embd = 128
n_head = 4
n_layer = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleGPT(nn.Module):
    def init(self, vocab_size):
        super().init()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        mask = torch.triu(torch.ones(T, T, device=device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

model = SimpleGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("1")

save_path = '/content/drive/MyDrive/Colab Notebooks/CreateLLM_Data/minigpt_v3.pth'
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
