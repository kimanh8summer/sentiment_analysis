import streamlit as st
import torch
import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor
from architecture import UltimateModelV9

# --- CONFIGURATION ---
CHECKPOINT_PATH = r"D:\Graduation\Program\Dataset\best_V9_supcon.pt"
CSV_PATH = r"D:\Graduation\Program\Dataset\mvsa_clean\mvsa.csv"
IMAGE_DIR = r"D:\Graduation\Program\Dataset\mvsa_dataset\MVSA\data"
DEVICE = torch.device("cpu")

@st.cache_resource
def load_resources():
    model = UltimateModelV9()
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    df_all = pd.read_csv(CSV_PATH)
    n = len(df_all)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    test_indices = indices[int(0.8*n) + int(0.1*n):] 
    test_df = df_all.iloc[test_indices].copy().reset_index(drop=True)
    return model, tokenizer, processor, test_df

try:
    model, tokenizer, processor, test_df = load_resources()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- UI CUSTOMIZATION ---
st.set_page_config(page_title="MVSA Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1 { font-weight: 800; letter-spacing: -1px; color: #1a1a1a; margin-bottom: 0; }
    h3 { font-weight: 600; color: #999; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1.5px; margin-top: 2rem; }
    .stButton>button {
        width: 100%; border-radius: 0px; border: 1px solid #1a1a1a;
        background-color: transparent; color: #1a1a1a; font-weight: 600; height: 3rem;
    }
    .stButton>button:hover { background-color: #1a1a1a; color: #ffffff; }
    .text-container {
        padding: 20px; background-color: #fcfcfc; border: 1px solid #eee;
        color: #444; line-height: 1.6; margin-top: 10px;
    }
    .result-card {
        padding: 30px; border: 1px solid #1a1a1a; text-align: center; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### System Control")
    test_idx = st.number_input("Sample Index", 0, len(test_df)-1, value=0)
    if st.button("Randomize"):
        st.session_state.current_idx = random.randint(0, len(test_df)-1)
        test_idx = st.session_state.current_idx
        st.rerun()
    st.markdown("---")
    st.caption("Multimodal Research Project")

# --- MAIN ---
st.title("Sentiment Analysis")
st.markdown("Inference Engine v9.0 | Cross-Attention Architecture")

row = test_df.iloc[test_idx]
text_content = str(row['text'])
img_path = os.path.join(IMAGE_DIR, row['image'])
ground_truth = row['label']

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Input Data")
    if os.path.exists(img_path):
        image = Image.open(img_path).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.error("Missing file.")
        image = None
    st.markdown(f'<div class="text-container">{text_content}</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Inference")
    if st.button("Analyze"):
        if image:
            inputs_text = tokenizer(text_content, return_tensors="pt", padding="max_length", 
                                    truncation=True, max_length=64).to(DEVICE)
            inputs_image = processor(images=image, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                logits, (t2i_attn, i2t_attn) = model(
                    ids=inputs_text['input_ids'], mask=inputs_text['attention_mask'], img=inputs_image['pixel_values']
                )
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            colors = {0: "#e63946", 1: "#457b9d", 2: "#2a9d8f"}
            
            st.markdown(f"""
                <div class="result-card">
                    <p style="color: #999; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 1px;">Prediction</p>
                    <h1 style="color: {colors[prediction]}; font-size: 3.5rem;">{labels_map[prediction]}</h1>
                    <p style="color: #bbb; font-size: 0.8rem;">Ground Truth: {labels_map.get(ground_truth, 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

            # --- VISUAL ATTENTION 
            st.markdown("### Visual Attention Map")
            v_attn = t2i_attn[0, 0, 0, 1:] if t2i_attn.dim() == 4 else t2i_attn[0, 0, 1:]
            v_attn = v_attn.cpu().numpy().reshape(14, 14)
            
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            ax1.imshow(image.resize((224, 224)))
            ax1.imshow(v_attn, cmap='jet', alpha=0.5, extent=(0, 224, 224, 0), interpolation='bilinear')
            ax1.axis('off')
            st.pyplot(fig1)

            # --- TEXTUAL ATTENTION 
            st.markdown("### Textual Attention Score")
            tokens = tokenizer.convert_ids_to_tokens(inputs_text['input_ids'][0])
            t_attn = i2t_attn[0].mean(dim=0).mean(dim=0) if i2t_attn.dim() == 4 else i2t_attn[0].mean(dim=0)
            t_attn = t_attn.cpu().numpy()
            
            mask_len = inputs_text['attention_mask'].sum().item()
            tokens_clean, weights_clean = tokens[:mask_len], t_attn[:mask_len]
            
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            
            sns.barplot(x=tokens_clean, y=weights_clean, ax=ax2, palette="viridis")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            sns.despine(left=True, bottom=True) 
            st.pyplot(fig2)