import pandas as pd
import streamlit as st
from PIL import Image

from src.hf_model import (
    DR_CLASSES,
    HF_MODEL_ID,
    attention_heatmap,
    count_parameters,
    load_hf_model,
    overlay_heatmap,
    predict_dr,
)

st.set_page_config(
    page_title="Retinal Disease Classifier — M.Tech Project",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main .block-container {padding-top: 1.5rem; max-width: 1200px;}
      .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white; padding: 1.8rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;
      }
      .hero h1 {margin:0; font-size: 1.9rem;}
      .hero p {margin: 0.3rem 0 0; opacity: 0.88;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px;
             background:#e6f0ff; color:#1d4ed8; font-size:0.75rem; margin-right:4px;}
      .metric-card {
        background:#f6f8fb; border:1px solid #e6eaf1; border-radius:12px;
        padding: 0.9rem 1rem;
      }
      .metric-card h4 {margin:0; color:#546175; font-weight:600; font-size:0.8rem;
                       text-transform:uppercase; letter-spacing:0.04em;}
      .metric-card .v {font-size:1.6rem; font-weight:700; color:#0f2027;}
      .sev-0 {color:#047857; font-weight:700;}
      .sev-1 {color:#0ea5e9; font-weight:700;}
      .sev-2 {color:#d97706; font-weight:700;}
      .sev-3 {color:#dc2626; font-weight:700;}
      .sev-4 {color:#7c2d12; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🩺 Retinal Disease Classification from Fundus Images</h1>
      <p>M.Tech project — diabetic retinopathy severity grading with a
         fine-tuned Vision Transformer (ViT-B/16) loaded from HuggingFace Hub.</p>
      <div style="margin-top:0.6rem;">
        <span class="pill">PyTorch</span>
        <span class="pill">ViT-B/16</span>
        <span class="pill">5-class DR grading</span>
        <span class="pill">Attention Rollout</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading pretrained ViT from HuggingFace…")
def get_model():
    return load_hf_model()


model, processor, device = get_model()

tab_predict, tab_model, tab_dataset, tab_results, tab_about = st.tabs(
    ["🔮 Predict", "🧠 Model", "📊 Dataset", "📈 Results", "ℹ️ About"]
)

# --------------------------- PREDICT ---------------------------
with tab_predict:
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader("Upload a retinal fundus image")
        uploaded = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"],
                                    label_visibility="collapsed")
        show_cam = st.checkbox("Show attention heatmap", value=True)
        st.caption(f"Device: **{device.upper()}** · Model: `{HF_MODEL_ID}`")

    if uploaded:
        image = Image.open(uploaded)
        with st.spinner("Running inference…"):
            result = predict_dr(model, processor, image, device)

        with left:
            st.image(image, caption="Input fundus image", use_column_width=True)

        with right:
            pred = result["pred"]
            sev = pred["severity"]
            conf = result["probs"][result["pred_idx"]]

            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"""<div class="metric-card"><h4>Predicted class</h4>
                    <div class="v sev-{sev}">{pred['name']}</div></div>""",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"""<div class="metric-card"><h4>Confidence</h4>
                    <div class="v">{conf:.1%}</div></div>""",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"""<div class="metric-card"><h4>Severity</h4>
                    <div class="v sev-{sev}">{sev} / 4</div></div>""",
                unsafe_allow_html=True,
            )

            st.info(f"**Clinical note** — {pred['description']}")

            st.markdown("##### Full class probabilities")
            rank_df = pd.DataFrame([
                {"Grade": r["severity"], "Class": r["name"],
                 "Probability": r["prob"]}
                for r in result["ranking"]
            ])
            st.dataframe(
                rank_df.style.format({"Probability": "{:.2%}"})
                .bar(subset=["Probability"], color="#4c8bf5", vmin=0, vmax=1),
                use_container_width=True, hide_index=True,
            )

            if show_cam:
                st.markdown("##### Attention rollout — where the ViT looked")
                heat = attention_heatmap(
                    result["attentions"], input_hw=tuple(result["input_size"])
                )
                overlay = overlay_heatmap(image, heat)
                st.image(
                    overlay,
                    caption="Red regions contributed most to the CLS token.",
                    use_column_width=True,
                )

# --------------------------- MODEL ---------------------------
with tab_model:
    st.subheader("Model: Vision Transformer (ViT-B/16)")
    st.markdown(
        f"""
**HuggingFace repo:** [`{HF_MODEL_ID}`](https://huggingface.co/{HF_MODEL_ID})

The classifier is a **Vision Transformer** (base variant, 16×16 patches)
fine-tuned for 5-class diabetic-retinopathy severity grading on real retinal
fundus images (APTOS-style labels). We use the pretrained weights directly
from the HuggingFace Hub, so predictions are **real** — no random head, no
placeholder — out of the box.

**Why ViT for fundus images?** Self-attention lets every patch exchange
information with every other patch from the very first layer, which is
helpful when lesions (microaneurysms, hemorrhages, hard exudates) are
spread across the retina and need to be correlated globally.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Architecture diagram")
        st.code(
            """Input (3 x 224 x 224)
         │
  ┌──────┴──────┐
  │ Patch embed │  (16×16 patches → 196 tokens + CLS)
  └──────┬──────┘
         │
  ┌──────┴──────┐
  │  12 × ViT   │  multi-head self-attention + MLP
  │  encoder    │  hidden=768, heads=12
  └──────┬──────┘
         │  (CLS token)
     LayerNorm
         │
     Linear(768 → 5)
         │
     Softmax
         │
  5 severity probabilities
  [No DR, Mild, Moderate, Severe, Proliferative]""",
            language="text",
        )
    with col2:
        st.markdown("##### Training recipe (upstream)")
        st.markdown(
            """
- **Backbone:** ViT-B/16, pretrained on ImageNet-21k → ImageNet-1k
- **Fine-tune data:** Kaggle diabetic retinopathy fundus images (APTOS-style)
- **Task:** 5-class single-label severity grading
- **Loss:** Cross-entropy
- **Framework:** HuggingFace Transformers, PyTorch
- **Preprocessing:** resize to 224, ImageNet normalization via
  `ViTImageProcessor`

##### Explainability
- **Attention Rollout** — multiplies attention matrices across all 12 ViT
  layers (with residual) and extracts the CLS-token row, producing a
  14×14 saliency map upsampled to 224×224.
            """
        )

    p = count_parameters(model)
    c1, c2 = st.columns(2)
    c1.metric("Total parameters", f"{p['total']/1e6:.1f} M")
    c2.metric("Inference device", device.upper())

# --------------------------- DATASET ---------------------------
with tab_dataset:
    st.subheader("Diabetic Retinopathy severity grading")

    st.markdown(
        """
The model is trained on **real retinal fundus photographs** labeled by
ophthalmologists on the standard 5-point international DR severity scale.
This is the same scale used in the APTOS 2019 Kaggle challenge and
clinical screening programs.
        """
    )

    classes_df = pd.DataFrame([
        {"Grade": c["severity"], "Code": c["code"],
         "Class": c["name"], "Description": c["description"]}
        for c in DR_CLASSES
    ])
    st.dataframe(classes_df, use_container_width=True, hide_index=True)

    st.markdown("##### Related dataset — RFMiD (not downloaded here)")
    st.markdown(
        """
The full pipeline in this repository is also wired up for the **Retinal
Fundus Multi-Disease Image Dataset (RFMiD)**, a 3,200-image multi-label
dataset with 45 fine-grained disease categories
([Kaggle](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification/data)).

Because the dataset is too large to download locally, we use the
pretrained ViT above for live inference. If you have the RFMiD dataset
locally you can still run `python -m src.train --data-root data` to
fine-tune our ResNet50 multi-label classifier — that code path is
preserved in `src/train.py`, `src/evaluate.py`, and `src/model.py`.
        """
    )

# --------------------------- RESULTS ---------------------------
with tab_results:
    st.subheader("Upstream results")
    st.markdown(
        f"""
This section reports the performance of the pretrained HuggingFace model
we use for live inference, as documented on its model card.

| Metric | Value |
| --- | --- |
| Validation accuracy | **0.7287** |
| Validation loss | 1.0460 |
| Framework | PyTorch + Transformers 4.26 |
| Parameters | ~86 M |
| Input size | 224 × 224 |

Source: [`{HF_MODEL_ID}`](https://huggingface.co/{HF_MODEL_ID})
        """
    )

    st.subheader("Evaluating on your own data")
    st.markdown(
        """
If you want to benchmark this model on a local RFMiD / APTOS split, drop
the images into `data/` and run:

```bash
python -m src.evaluate_hf --data-root data
```
(The legacy `src/evaluate.py` is still available for the ResNet50 multi-label
fine-tune path described in the Dataset tab.)
        """
    )

# --------------------------- ABOUT ---------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
End-to-end M.Tech project for retinal disease classification. The focus
is on shipping a **real, explainable, deployable** diagnosis pipeline on
top of modern pretrained medical-imaging models.

### Pipeline
1. **Model sourcing** — a Vision Transformer (ViT-B/16) fine-tuned for DR
   severity grading is pulled live from HuggingFace Hub
   (`{HF_MODEL_ID}`). No random weights, no placeholder head — real
   predictions from the first click.
2. **Preprocessing** — `ViTImageProcessor` resizes to 224×224 and applies
   ImageNet normalization so the model receives inputs in its training
   distribution.
3. **Inference** — single forward pass through the transformer encoder;
   the CLS-token logits go through a softmax to produce 5 calibrated
   severity probabilities.
4. **Explainability** — Attention Rollout propagates CLS-token attention
   through every layer (with residual), producing a 14×14 saliency map
   upsampled to the original image. The overlay shows which retinal
   regions drove the prediction — critical for clinical trust.
5. **Deployment** — this Streamlit application for interactive diagnosis.

### Running this app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the ViT weights (~345 MB) from HuggingFace into
your local HF cache; subsequent runs are instant.

### Optional — fine-tune on RFMiD yourself

The repository still contains a full ResNet50 + two-stage fine-tuning
pipeline for the 46-label RFMiD task
([`src/train.py`](src/train.py), [`src/evaluate.py`](src/evaluate.py)),
in case you have disk space for the dataset and want to train end-to-end.

### Disclaimer
Academic prototype — **not a medical device**. Predictions must not be
used for clinical decision-making.
        """
    )
