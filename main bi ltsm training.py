import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# SEAROUTE AI
# =====================================================

MODEL_NAME = "Searoute-270M"
MODEL_VERSION = "v1.0"

# =====================================================
# HYPERPARAMETER
# =====================================================

BATCH_SIZE = 8
BLOCK_SIZE = 64
EMBED_DIM = 256
N_HEAD = 8
N_LAYER = 6
EPOCHS = 20
LR = 3e-4

# =====================================================
# LOAD DATA
# =====================================================

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

# =====================================================
# DATASET
# =====================================================

class TextDataset(Dataset):

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):

        x = self.data[idx:idx + self.block_size]

        y = self.data[idx + 1:idx + self.block_size + 1]

        return x, y

dataset = TextDataset(
    data,
    BLOCK_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =====================================================
# MODEL
# =====================================================

class SearouteAI(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            EMBED_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=N_HEAD,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYER
        )

        self.ln = nn.LayerNorm(
            EMBED_DIM
        )

        self.head = nn.Linear(
            EMBED_DIM,
            vocab_size
        )

    def forward(self, x):

        B, T = x.shape

        pos = torch.arange(
            T,
            device=x.device
        )

        tok = self.token_embedding(x)

        pos = self.position_embedding(pos)

        x = tok + pos

        x = self.transformer(x)

        x = self.ln(x)

        logits = self.head(x)

        return logits

# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODEL INIT
# =====================================================

model = SearouteAI().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

criterion = nn.CrossEntropyLoss()

# =====================================================
# INFO
# =====================================================

print("=" * 60)
print(f"{MODEL_NAME}")
print("Industrial Language Model")
print(f"Version : {MODEL_VERSION}")
print("=" * 60)
print(f"Device           : {device}")
print(f"Vocabulary Size  : {vocab_size}")
print(f"Embedding Dim    : {EMBED_DIM}")
print(f"Transformer Layer: {N_LAYER}")
print(f"Attention Heads  : {N_HEAD}")
print("=" * 60)

# =====================================================
# TRAINING
# =====================================================

print("\nTraining dimulai...\n")

for epoch in range(EPOCHS):

    total_loss = 0.0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss = {avg_loss:.4f}"
    )

# =====================================================
# SAVE MODEL
# =====================================================

checkpoint = {
    "model_name": MODEL_NAME,
    "version": MODEL_VERSION,
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
    "model_state_dict": model.state_dict()
}

torch.save(
    checkpoint,
    "searoute-270m-v1.0.pth"
)

print("\n" + "=" * 60)
print("TRAINING SELESAI")
print(f"Model tersimpan : searoute-270m-v1.0.pth")
print("=" * 60)

# =====================================================
# SIMPLE GENERATION TEST
# =====================================================

model.eval()

prompt = "PLC"

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long
).to(device)

with torch.no_grad():

    for _ in range(100):

        logits = model(context)

        logits = logits[:, -1, :]

        probs = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        context = torch.cat(
            [context, next_token],
            dim=1
        )

        if context.shape[1] > BLOCK_SIZE:
            context = context[:, -BLOCK_SIZE:]

generated = decode(
    context[0].cpu().tolist()
)

print("\n===== HASIL GENERASI =====")
print(generated)
