import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

class UltimateModelV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.img_proj = nn.Linear(768, 768)
        
        # Cross-Attention
        self.t2i = nn.MultiheadAttention(768, 8, batch_first=True)
        self.i2t = nn.MultiheadAttention(768, 8, batch_first=True)
        self.normT = nn.LayerNorm(768)
        self.normI = nn.LayerNorm(768)
        
        self.gate = nn.Linear(768 * 2, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.5), nn.Linear(512, 3)
        )
        self.proj = nn.Linear(768, 256)

    def forward(self, ids, mask, img):
        H_T = self.bert(ids, attention_mask=mask).last_hidden_state
        H_I = self.img_proj(self.vit(pixel_values=img).last_hidden_state)
        
        # T2I Attention (Text nhìn vào Image)
        T2I, t2i_attn = self.t2i(H_T, H_I, H_I)
        H_T = self.normT(H_T + T2I)
        
        # I2T Attention (Image nhìn vào Text)
        I2T, i2t_attn = self.i2t(H_I, H_T, H_T, key_padding_mask=(mask == 0))
        H_I = self.normI(H_I + I2T)
        
        v_T = (H_T * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-9)
        v_I = H_I.mean(1)
        g = torch.sigmoid(self.gate(torch.cat([v_T, v_I], dim=-1)))
        v_fuse = g * v_T + (1 - g) * v_I
        
        logits = self.classifier(v_fuse)
        return logits, (t2i_attn, i2t_attn)