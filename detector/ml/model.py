import torch
import numpy as np
import os

from transformers import DebertaV2Tokenizer, DebertaV2Model
from torch import nn
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import neattext.functions as nfx


class DebertaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Load base model in float32 from the beginning
        self.deberta = DebertaV2Model.from_pretrained(
            "microsoft/deberta-v3-base",
            torch_dtype=torch.float32   # Force float32
        )
        hidden = self.deberta.config.hidden_size
        self.attention = nn.Linear(hidden, 1, dtype=torch.float32)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 2, dtype=torch.float32)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -10000)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits


class FakeJobPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        deberta_path = os.path.join(base_dir, 'ml', 'best_model.pt')
        xgb_path = os.path.join(base_dir, 'ml', 'xgb_model.json')

        if not os.path.exists(deberta_path):
            raise FileNotFoundError(f"best_model.pt not found at: {deberta_path}")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"xgb_model.json not found at: {xgb_path}")

        print(f"Loading models on device: {self.device}")

        # ====================== DeBERTa with Full float32 Fix ======================
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
        
        # Create model with explicit float32
        self.deberta_model = DebertaAttention()

        # Load saved weights and convert everything to float32
        state_dict = torch.load(deberta_path, map_location='cpu', weights_only=True)
        
        # Force every tensor to float32
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(torch.float32)

        self.deberta_model.load_state_dict(state_dict, strict=False)

        self.deberta_model = self.deberta_model.to(self.device)
        self.deberta_model.eval()
        # ===========================================================================

        # SentenceTransformer
        try:
            self.sem_model = SentenceTransformer("all-mpnet-base-v2", local_files_only=True)
            print("SentenceTransformer loaded from cache.")
        except:
            print("Downloading SentenceTransformer model...")
            self.sem_model = SentenceTransformer("all-mpnet-base-v2")
            print("Download completed.")

        # XGBoost
        self.xgb_clf = xgb.XGBClassifier()
        self.xgb_clf.load_model(xgb_path)

        self.weight_deberta = 0.65
        self.best_threshold = 0.5

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = nfx.remove_html_tags(text)
        text = nfx.remove_urls(text)
        text = nfx.remove_emails(text)
        text = nfx.remove_special_characters(text)
        return text

    def predict(self, job_content: str):
        cleaned = self.clean_text(job_content)
    
    # DeBERTa Inference
        enc = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
    )
        input_ids = enc['input_ids'].to(self.device)
        mask = enc['attention_mask'].to(self.device)
    
        with torch.no_grad():
            logits = self.deberta_model(input_ids, mask)
            prob_deberta = torch.softmax(logits, dim=1)[:, 1].cpu().item()
    
    # XGBoost Inference
        emb = self.sem_model.encode([cleaned])
        prob_xgb = self.xgb_clf.predict_proba(emb)[0, 1]
    
    # Ensemble → probability of FAKE
        ensemble_prob = (
            self.weight_deberta * prob_deberta +
            (1 - self.weight_deberta) * prob_xgb
        )
    
    # Compute REAL confidence
        real_confidence = 1 - ensemble_prob
    
    # Decision Rule
        if real_confidence >= 0.80:
            prediction = "Real"
            is_fraudulent = False
        else:
            prediction = "Fake"
            is_fraudulent = True
    
        return {
            "prediction": prediction,
            "confidence": round(real_confidence * 100, 2),
            "is_fraudulent": is_fraudulent
        }

# Singleton
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = FakeJobPredictor()
    return _predictor