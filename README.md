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
│   ├── best_seq2seq.pt         # Vanilla Seq2Seq best model
│   └── best_attention.pt       # Attention model best model
├── configs/                    # W&B sweep configurations
│   ├── sweep_config.yaml       # Vanilla model sweep
│   └── sweep_attention.yaml    # Attention model sweep
├── fonts/                      # Devanagari fonts for plotting
├── lexicons/                   # Dakshina train/dev/test TSV files
│   ├── hi.translit.sampled.train.tsv
│   ├── hi.translit.sampled.dev.tsv
│   └── hi.translit.sampled.test.tsv
├── notebooks/                  # Demo notebooks
│   ├── solution_vanilla.ipynb
│   └── solution_attention.ipynb
├── predictions_vanilla/        # Vanilla model outputs (TSV/CSV + sample PNG)
├── predictions_attention/      # Attention model outputs (TSV/CSV + heatmaps)
├── solution_1.py               # Q1: Vanilla Seq2Seq implementation
├── solution_2.py               # Q2: W&B sweep driver for vanilla model
├── solution_2b.py              # Q2b: Beam-size tuning for vanilla model
├── solution_4.py               # Q4: Evaluate vanilla model on test set
├── solution_5_model.py         # Q5: Attention Seq2Seq model definition
├── solution_5.py               # Q5: W&B sweep driver for attention model
├── solution_5a.py              # Q5a: Beam-size tuning for attention model
├── solution_5b.py              # Q5b: Evaluate attention model on test set
├── solution_6.py               # Q6: Interactive attention heatmap generator
├── connectivity.html           # Exported connectivity visualization
└── README.md                   # This file
```

---

## ⚙️ Installation

```bash
git clone <repo-url>
cd <repo-directory>
# (Optional) create a virtual environment
pip install pytorch, pandas, numpy, wandb, plotly, matplotlib, wcwidth
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
# 1) Single training run
python solution_1.py \ 
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --embedding_size 128 \
    --hidden_size    256 \
    --encoder_layers 1 \
    --decoder_layers 1 \
    --cell GRU \
    --epochs 10 \
    --embedding_method svd_ppmi \
    --use_attestations

# 2) Hyperparameter sweep on vanilla model
python solution_2.py \
    --mode sweep \
    --sweep_config sweep_config.yaml \
    --wandb_project DA6401_Intro_to_DeepLearning_Assignment_3 \
    --wandb_run_tag solution_2 \
    --gpu_ids 0 1 \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --sweep_count 50

# 2b) Beam-size tuning for vanilla model
python solution_2b.py \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --gpu_ids   3

# 4) Evaluate best model on test set
python solution_4.py \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --checkpoint ./checkpoints/best_seq2seq.pt \
    --output_dir predictions_vanilla \
    --gpu_ids 3 \
    --wandb_project transliteration \
    --wandb_run_name solution_4_run \
    --wandb_run_tag solution_4
```

### 2. Train & evaluate attention Seq2Seq (Q5)

```bash
# 5) Hyperparameter sweep with attention
python solution_5.py \
    --mode sweep \
    --sweep_config sweep_attention.yaml \
    --wandb_project DA6401_Intro_to_DeepLearning_Assignment_3 \
    --wandb_run_tag solution_5 \
    --gpu_ids 0 2 3 \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --sweep_count 75

# 5a) Beam-size tuning for attention model
python solution_5a.py \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --gpu_ids   3

# 5b) Evaluate best attention model
python solution_5b.py \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --checkpoint ./checkpoints/best_attention.pt \
    --output_dir predictions_attention \
    --gpu_ids 3 \
    --wandb_project DA6401_Intro_to_DeepLearning_Assignment_3 \
    --wandb_run_name solution_5b_run \
    --wandb_run_tag solution_5b
```

### 3. Visualize attention connectivity (Q6)

```bash
python solution_6.py \
    --checkpoint ./checkpoints/best_attention.pt \
    --train_tsv   ./lexicons/hi.translit.sampled.train.tsv \
    --test_tsv    ./lexicons/hi.translit.sampled.test.tsv \
    --n_examples  5 \
    --output_html connectivity.html \
    --wandb_project DA6401_Intro_to_DeepLearning_Assignment_3 \
    --wandb_run_name solution_6_run \
    --wandb_run_tag  solution_6
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
