# 1. Environment Setup
# ─────────────────────────────────────────────────────────────

import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

# ─────────────────────────────────────────────────────────────
# 2. Install Libraries
# ─────────────────────────────────────────────────────────────

!pip install -q transformers pandas scikit-learn seaborn matplotlib neattext sentence-transformers xgboost tqdm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

import neattext.functions as nfx
from tqdm import tqdm

from transformers import DebertaV2Tokenizer, DebertaV2Model, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch import nn

from sentence_transformers import SentenceTransformer
import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# 3. Load Dataset
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("/content/fake_job_postings.csv", engine="python", on_bad_lines="skip")

cols = ['title','company_profile','description','requirements','benefits','fraudulent']
df = df[cols]

text_cols = ['title','company_profile','description','requirements','benefits']

df['job_content'] = df[text_cols].fillna('').agg(' '.join, axis=1)

df = df[['job_content','fraudulent']]

# ─────────────────────────────────────────────────────────────
# 4. Text Cleaning
# ─────────────────────────────────────────────────────────────

df['job_content'] = df['job_content'].str.lower()
df['job_content'] = df['job_content'].apply(nfx.remove_html_tags)
df['job_content'] = df['job_content'].apply(nfx.remove_urls)
df['job_content'] = df['job_content'].apply(nfx.remove_emails)
df['job_content'] = df['job_content'].apply(nfx.remove_special_characters)

df = df.drop_duplicates().dropna()

print("Dataset shape:", df.shape)

# ─────────────────────────────────────────────────────────────
# 5. Train Test Split
# ─────────────────────────────────────────────────────────────

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['fraudulent'],
    random_state=SEED
)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

# ─────────────────────────────────────────────────────────────
# 6. Tokenization
# ─────────────────────────────────────────────────────────────

tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

MAX_LEN = 256

train_enc = tokenizer(
    list(train_df['job_content']),
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

test_enc = tokenizer(
    list(test_df['job_content']),
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

train_labels = torch.tensor(train_df['fraudulent'].values).long()
test_labels = torch.tensor(test_df['fraudulent'].values).long()

train_dataset = TensorDataset(
    train_enc['input_ids'],
    train_enc['attention_mask'],
    train_labels
)

test_dataset = TensorDataset(
    test_enc['input_ids'],
    test_enc['attention_mask'],
    test_labels
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ─────────────────────────────────────────────────────────────
# 7. Focal Loss
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        loss = self.alpha * (1-pt)**self.gamma * ce_loss

        return loss.mean()

# ─────────────────────────────────────────────────────────────
# 8. DeBERTa + Attention Pooling
# ─────────────────────────────────────────────────────────────

class DebertaAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")

        hidden = self.deberta.config.hidden_size

        self.attention = nn.Linear(hidden,1)

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(hidden,2)

    def forward(self,input_ids,attention_mask):

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        scores = self.attention(hidden_states).squeeze(-1)

        scores = scores.masked_fill(attention_mask==0,-10000)

        weights = torch.softmax(scores,dim=1)

        context = torch.sum(hidden_states*weights.unsqueeze(-1),dim=1)

        context = self.dropout(context)

        logits = self.classifier(context)

        return logits

# ─────────────────────────────────────────────────────────────
# 9. Training Setup
# ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DebertaAttention().to(device).float()

criterion = FocalLoss()

optimizer = AdamW(model.parameters(),lr=2e-5)

EPOCHS = 5

total_steps = len(train_loader)*EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    int(0.1*total_steps),
    total_steps
)

# ─────────────────────────────────────────────────────────────
# 10. Training Loop
# ─────────────────────────────────────────────────────────────

print("Training DeBERTa...")

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in tqdm(train_loader):

        input_ids,mask,labels = [t.to(device) for t in batch]

        optimizer.zero_grad()

        logits = model(input_ids,mask)

        loss = criterion(logits,labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss/len(train_loader)

    print("Epoch",epoch+1,"Loss:",avg_loss)

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(model.state_dict(),"best_model.pt")

model.load_state_dict(torch.load("best_model.pt"))

# ─────────────────────────────────────────────────────────────
# 11. Evaluation
# ─────────────────────────────────────────────────────────────

model.eval()

preds=[]
trues=[]
probs=[]

with torch.no_grad():

    for batch in test_loader:

        input_ids,mask,labels = [t.to(device) for t in batch]

        logits = model(input_ids,mask)

        prob = torch.softmax(logits,dim=1)[:,1]

        pred = torch.argmax(logits,dim=1)

        preds.extend(pred.cpu().numpy())
        trues.extend(labels.cpu().numpy())
        probs.extend(prob.cpu().numpy())

print("\nDeBERTa Results")
print(classification_report(trues,preds))

bal_acc = balanced_accuracy_score(trues,preds)
auc = roc_auc_score(trues,probs)

print("Balanced Accuracy:",bal_acc)
print("ROC AUC:",auc)

# ─────────────────────────────────────────────────────────────
# 12. Sentence Transformer + XGBoost
# ─────────────────────────────────────────────────────────────

print("\nTraining XGBoost Hybrid...")

sem_model = SentenceTransformer("all-mpnet-base-v2")

train_emb = sem_model.encode(train_df['job_content'].tolist())
test_emb = sem_model.encode(test_df['job_content'].tolist())

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr"
)

xgb_clf.fit(train_emb,train_df['fraudulent'])

xgb_probs = xgb_clf.predict_proba(test_emb)[:,1]

# ─────────────────────────────────────────────────────────────
# 13. Ensemble
# ─────────────────────────────────────────────────────────────

weight_deberta = 0.65

ensemble_probs = weight_deberta*np.array(probs)+(1-weight_deberta)*xgb_probs

ensemble_preds = (ensemble_probs>=0.5).astype(int)

print("\nENSEMBLE RESULTS")

print(classification_report(test_df['fraudulent'],ensemble_preds))

# ─────────────────────────────────────────────────────────────
# 14. Confusion Matrix
# ─────────────────────────────────────────────────────────────

cm = confusion_matrix(test_df['fraudulent'],ensemble_preds)

plt.figure(figsize=(6,5))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
xticklabels=["Real","Fake"],
yticklabels=["Real","Fake"])

plt.title("Confusion Matrix")
plt.show()

# ─────────────────────────────────────────────────────────────
# 15. ROC Curve
# ─────────────────────────────────────────────────────────────

fpr,tpr,_ = roc_curve(test_df['fraudulent'],ensemble_probs)

plt.figure(figsize=(7,5))

plt.plot(fpr,tpr,label="ROC Curve")
plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

print("\nPipeline Completed Successfully")
