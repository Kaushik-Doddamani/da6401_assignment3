# Dakshina Transliteration Seq2Seq

This repository contains solutions for character-level transliteration of Dakshina languages (Hindi) using:

* A **vanilla** RNN/LSTM/GRU Seq2Seq model (`solution_1.py`, `solution_2.py`, `solution_4.py`)
* An **attention-augmented** Seq2Seq model (`solution_5_model.py`, `solution_5.py`, `solution_5b.py`)
* An **interactive connectivity** visualization of attention weights (`solution_6.py`)
* Jupyter notebooks demonstrating training and evaluation:

  * `notebooks/solution_vanilla.ipynb`
  * `notebooks/solution_attention.ipynb`

---

## 📁 Project Structure

```
├── checkpoints/                # Saved model checkpoints
│   ├── best_seq2seq.pt
│   └── best_attention.pt
├── configs/                    # W&B sweep configurations
│   ├── sweep_config.yaml
│   └── sweep_attention.yaml
├── fonts/                      # Devanagari fonts for plotting
├── lexicons/                   # Dakshina train/dev/test TSV files
├── notebooks/                  # Demo notebooks
│   ├── solution_vanilla.ipynb
│   └── solution_attention.ipynb
├── predictions_vanilla/        # Vanilla model outputs (TSV/CSV + sample PNG)
├── predictions_attention/      # Attention model outputs (TSV/CSV + heatmaps)
├── solution_1.py               # Q1: Vanilla Seq2Seq implementation
├── solution_2.py               # Q2: W&B sweep driver for vanilla model
├── solution_4.py               # Q4: Evaluate vanilla model on test set
├── solution_5_model.py         # Q5: Attention Seq2Seq model definition
├── solution_5.py               # Q5: W&B sweep driver for attention model
├── solution_5b.py              # Q5.b: Evaluate attention model on test set
├── solution_6.py               # Q6: Interactive attention heatmap generator
├── connectivity.html           # Exported connectivity visualization
├── part_a.ipynb                # Starter code / reference notebook
└── README.md                   # This file
```

---

## ⚙️ Installation

```bash
git clone <repo-url>
cd <repo-directory>
# (Optional) create a virtual environment
pip install -r requirements.txt
```

**Key dependencies:** PyTorch, pandas, numpy, wandb, plotly, matplotlib, wcwidth.

---

## 📦 Data

Place the Dakshina Hindi lexicon files under `lexicons/`:

* `hi.translit.sampled.train.tsv`
* `hi.translit.sampled.dev.tsv`
* `hi.translit.sampled.test.tsv`

Format: `native_word \t romanized_word \t count`.

---

## 🚀 Usage

### 1. Train & evaluate vanilla Seq2Seq (Q1–Q4)

```bash
# 1a) Single training run
python solution_1.py --train_tsv lexicons/hi.translit.sampled.train.tsv \
                    --dev_tsv   lexicons/hi.translit.sampled.dev.tsv \
                    --test_tsv  lexicons/hi.translit.sampled.test.tsv \
                    --embedding_size 256 --hidden_size 512 --cell LSTM \
                    --encoder_layers 2 --decoder_layers 2 --epochs 15

# 1b) Hyperparameter sweep
python solution_2.py --mode sweep --sweep_config configs/sweep_config.yaml \
                    --wandb_project transliteration --wandb_run_tag vanilla \
                    --train_tsv lexicons/...train.tsv \
                    --dev_tsv lexicons/...dev.tsv --test_tsv lexicons/...test.tsv \
                    --sweep_count 30

# 1c) Evaluate best model on test set
python solution_4.py --train_tsv lexicons/...train.tsv \
                    --dev_tsv   lexicons/...dev.tsv \
                    --test_tsv  lexicons/...test.tsv \
                    --checkpoint checkpoints/best_seq2seq.pt \
                    --output_dir predictions_vanilla \
                    --wandb_project transliteration --wandb_run_tag eval_vanilla
```

### 2. Train & evaluate attention Seq2Seq (Q5)

```bash
# 2a) Hyperparameter sweep with attention
python solution_5.py --mode sweep --sweep_config configs/sweep_attention.yaml \
                    --wandb_project transliteration --wandb_run_tag attention \
                    --train_tsv lexicons/...train.tsv \
                    --dev_tsv lexicons/...dev.tsv --test_tsv lexicons/...test.tsv \
                    --sweep_count 30

# 2b) Evaluate best attention model
python solution_5b.py --train_tsv lexicons/...train.tsv \
                     --dev_tsv   lexicons/...dev.tsv \
                     --test_tsv  lexicons/...test.tsv \
                     --checkpoint checkpoints/best_attention.pt \
                     --output_dir predictions_attention \
                     --wandb_project transliteration --wandb_run_tag eval_attention
```

### 3. Visualize attention connectivity (Q6)

```bash
python solution_6.py --checkpoint checkpoints/best_attention.pt \
                     --train_tsv lexicons/...train.tsv \
                     --test_tsv  lexicons/...test.tsv \
                     --n_examples 5 --output_html connectivity.html \\  
                     --wandb_project transliteration --wandb_run_tag connectivity
```

---

## 📓 Notebooks

* **`solution_vanilla.ipynb`**: Interactive walkthrough of Q1–Q4.
* **`solution_attention.ipynb`**: Interactive walkthrough of Q5 and Q6.

Feel free to run these notebooks to visualize training curves, sample predictions, and attention heatmaps inline.

---

## 📑 License & Acknowledgments

This project is for academic purposes (DA6401 Assignment 3).
Inspired by tutorials on Keras, PyTorch, and the “Connectivity” blog post.
