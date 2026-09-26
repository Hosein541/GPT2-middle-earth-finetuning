این فایل `README.md` به صورت کامل، حرفه‌ای و دقیق بر اساس کدهای پیاده‌سازی‌شده شما آماده شده است. در این مستند، علاوه بر ارجاع محترمانه به کتاب سباستین راشکا (*Build a Large Language Model from Scratch*)، به صورت شفاف و پررنگ بر روی **توسعه‌ها و بهینه‌سازی‌های اختصاصی شما** (مانند پایپ‌لاین استخراج PDF، تقسیم‌بندی Interleaved، ماژول گرادیان اکیومیولیشن، کنترل نرخ یادگیری و چک‌پوینتینگ) تأکید شده و تحلیل فنی محدودیت حجم دیتاست در برابر ظرفیت مدل ۱۲۴ میلیونی آورده شده است.

فایل نمودار Loss را که قبلاً کدش را ساختیم، با نام `loss_curve.png` داخل پوشه‌ای به نام `assets` در ریشه ریپازیتوری بگذارید:

```text
your-repo/
├── assets/
│   └── loss_curve.png
├── previous_chapters.py
├── train.py (یا نوت‌بوک پروژه)
├── requirements.txt
└── README.md

```

---

```markdown
# 🧙‍♂️ Training a 124M GPT Architecture on The Lord of the Rings Corpus from Scratch

An end-to-end implementation and domain-adaptive pretraining of a **124-million parameter decoder-only GPT model** from scratch on the complete *The Lord of the Rings* literary corpus using **PyTorch**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```

---

## 📌 Project Overview

This project explores pretraining a generative autoregressive language model (**124M GPT architecture**) directly on domain-specific raw literature.

While the architectural blueprint is inspired by foundational concepts from Sebastian Raschka's book *Build a Large Language Model from Scratch* (Chapters 4 & 5), the entire training infrastructure, data ingestion pipeline, scheduling mechanics, and checkpointing logic were **substantially re-engineered and extended** to handle a real-world multi-volume text corpus under strict single-GPU constraints.

### 🛠️ Key Custom Engineering & Enhancements

Beyond the textbook baseline, this repository implements:

1. **Automated PDF Parsing & Text Normalization Pipeline**: Custom ingestion with `PyMuPDF` (`fitz`), handling hyphenated word reconstructs across line breaks (`re.sub`), whitespace consolidation, and boundary cleaning across 3 full-length volumes.
2. **Interleaved Text Chunking (`split_text_interleaved`)**: Rather than a trivial sequential train/val split (which introduces heavy narrative/lexical distribution shifts across different acts of a novel), the text is partitioned into 10 interleaved temporal blocks with word-boundary awareness (`rfind`).
3. **Advanced Training Loop Mechanics**:
* **Gradient Accumulation** (`accumulation_steps=4`): Emulates larger effective batch sizes ($16 \times 4 = 64$) without OOM errors.
* **Custom Linear Warmup + Cosine Decay Scheduler**: Synchronized with accumulation steps to stabilize early optimization.
* **Gradient Norm Clipping**: Applied post-warmup (`max_norm=1.0`) to avoid gradient explosions.
* **Robust Checkpoint Management**: Supports mid-training resumption (`resume_path`), tracking seen tokens, optimizer states, and best validation checkpoints.



---

## ⚠️ Critical Analysis: Dataset Scale vs. Model Capacity

> **Technical Discussion on Constraints**:
> * **Corpus Scale**: ~2,232,063 characters (~558,015 tokens) for training and ~248,059 characters (~62,014 tokens) for validation.
> * **Parameter Scale**: The architecture hosts **~124 million parameters** (12 layers, 12 attention heads, 768 embedding dimension).
> * **The Data-Hungry Nature of LLMs**: According to scaling laws (e.g., Chinchilla / Hoffmann et al.), training a 124M model from scratch optimally demands hundreds of millions (to billions) of tokens. With ~0.6M tokens, the parameter space is vastly larger than the unique token surface.
> * **Observed Dynamics**: The cross-entropy loss drops rapidly from initial baseline (`~10.90`) down to `3.20` on training data. However, the validation loss plateaus around `~4.44` (producing a ~1.2 loss gap at Step 630), representing natural data saturation and early overfitting.
> * **Takeaway**: While small corpora are insufficient to build an open-domain foundation model, this experiment successfully demonstrates how rapidly the model acquires **domain-specific syntax, Tolkien-esque character naming, and localized narrative rhythm**.
> 
> 

---

## ⚙️ Model Architecture & Training Hyperparameters

| Hyperparameter | Value | Description |
| --- | --- | --- |
| **Model Type** | Decoder-only Transformer | GPT-2 Style Architecture |
| **Parameters** | ~124 Million | `emb_dim=768`, `n_layers=12`, `n_heads=12` |
| **Context Length** | 256 tokens | Reduced context for efficient memory usage |
| **Vocabulary Size** | 50,257 | Byte-pair encoding (`tiktoken` GPT-2 tokenizer) |
| **Per-Device Batch Size** | 16 | Stride = 64 tokens |
| **Accumulation Steps** | 4 | Effective batch size = 64 sequences |
| **Optimizer** | `AdamW` | Weight decay = 0.05 |
| **Peak Learning Rate** | `3e-4` | Linear warmup (80 steps) to `3e-4`, Cosine decay |
| **Gradient Clipping** | 1.0 max norm | Enabled after warmup |

---

## 📈 Training Dynamics & Loss Curve

The model converged smoothly from random token initialization down to coherent phrase structures:

### Progression Summary

* **Step 0**: Train Loss `10.901` | Val Loss `10.910` (Uniform random prediction)
* **Step 120**: Train Loss `4.976` | Val Loss `5.462`
* **Step 330**: Train Loss `3.977` | Val Loss `4.728`
* **Step 630**: Train Loss `3.204` | Val Loss `4.442` (Best validation checkpoint)

---

## 📜 Qualitative Text Generation Progression

Prompt Context: `When Bilbo came to himself...`

* **Early Training (Epoch 1, Step 135 - Val Loss ~5.38):**
> *“‘I have you,’ said Frodo. ‘I am not know,’ said Frodo. ‘I have ’ said Pippin. ‘I am not ’ said Frodo”*
> *(Repeats high-frequency dialogue tokens; characters start appearing)*


* **Mid Training (Epoch 2, Step 285 - Val Loss ~4.81):**
> *“‘I am afraid,’ said Frodo. ‘I don't know, and I ‘I am afraid. I don't know what I don't know what I the Shire, and I had been”*
> *(Acquires Middle-earth proper nouns like ‘the Shire’ and punctuation patterns)*


* **Late Training (Epoch 4, Step 585 - Val Loss ~4.46):**
> *“‘I don't know what I don't know,’ said Frodo. ‘I don't know you can't know what I can't know. I don't think I don't know what I you can't know.’”*
> *(Grammatically structured conversational flow, though repetitive due to limited training tokens)*



---

## 🚀 Quickstart

### 1. Installation

```bash
git clone [https://github.com/](https://github.com/)/.git
cd 
pip install -r requirements.txt

```

### 2. Dataset Setup & Copyright Note

> **Important**: In compliance with international copyright laws and the Tolkien Estate, the complete raw text files (`.txt` / `.pdf`) are **not distributed** in this repository.

To train the model on your own legal copy:

1. Place your text files in the project root: `lotr1.txt`, `lotr2.txt`, `lotr3.txt` (or provide the PDFs for the extraction script).
2. Run the preprocessing and training pipeline:

```bash
python train.py

```

### 3. Generate from Checkpoint

```python
import torch
import tiktoken
from utils import GPTModel, generate_and_print_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# Load trained weights
checkpoint = torch.load("best_model.pt", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Sample generation
context = "Gollum was tugging at Frodo’s cloak and hissing with fear and"
generate_and_print_sample(model, tokenizer, device, context)

```

---

## 📂 Repository Structure

```text
├── assets/
│   └── loss_curve.png             # Training & Validation loss visualization
├── uitls.py                       # Core transformer blocks, attention modules & dataset loaders
├── GPT2_lotr.ipynb                # Data preparation, Training engine with accumulation, warmup & scheduler
├── requirements.txt               # Dependencies
└── README.md                      # Documentation

```

---

##  Acknowledgements

* **Sebastian Raschka** for the educational GPT architecture foundation in *Build a Large Language Model from Scratch*.
* **PyTorch** & **Tiktoken** for efficient tensor computation and tokenization.


---