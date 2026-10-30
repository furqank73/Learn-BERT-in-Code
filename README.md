# Learn BERT in Code

This folder contains three independent, hands-on notebooks about Transformer
models. One builds a small BERT-style model; the other two train
encoder–decoder Transformers for translation.

## Notebooks

### 1. `EncoderBertModel.ipynb` — a small BERT-style model

Builds a compact Transformer encoder called `BabyBERT` and trains it on
sentence pairs created in the notebook. The notebook walks through:

- Preparing sentence pairs labeled as consecutive or unrelated.
- Creating a vocabulary with BERT-style special tokens, including `[CLS]`,
  `[SEP]`, and `[MASK]`.
- Converting sentence pairs into padded inputs with segment IDs and attention
  masks.
- Training with two objectives: masked language modeling (MLM) and
  next-sentence prediction (NSP).
- Using the trained model to predict whether sentences belong together and to
  fill in a masked word.

This is a small educational model trained from scratch on a tiny example
dataset. It is not a pretrained BERT model, and its predictions are for
demonstration rather than reliable language understanding.

### 2. `Simple_Transformer_Translation_Explained_fixed.ipynb` — German to English

Introduces an encoder–decoder Transformer for German-to-English translation
with the Multi30K dataset. It covers token vocabularies, positional encoding,
decoder look-ahead masks, shifted targets, a training step, and greedy
translation.

Before running the data-loading cell, place the extracted files in a
`multi30k_data` directory relative to the notebook's working directory:

```text
multi30k_data/
├── train.de
├── train.en
├── val.de
└── val.en
```

The notebook trains for one epoch by default as a quick check. Increase the
epoch count in the training cell for a longer experiment. After training, it
saves the model weights to `simple_translation_model.pt` in the current
working directory.

### 3. `English_Urdu_Transformer_High_Accuracy.ipynb` — English to Urdu

Trains an encoder–decoder Transformer using an English–Urdu parallel-text
dataset. It loads and checks the data, creates a reproducible 90/10
train/validation split, builds vocabularies and batches, trains with validation
loss tracking, plots the losses, and demonstrates translations alongside
reference sentences.

The notebook expects a file named `eng_urdu_dataset.csv` with columns named
`english` and `urdu`. It searches several common notebook/Colab locations for
the CSV and also provides an upload path when running in Google Colab. If your
file uses different column names, update `SOURCE_COLUMN` and `TARGET_COLUMN`
in the notebook.

The word “High_Accuracy” is part of the notebook's filename, not a guarantee of
translation quality. Results depend on the dataset, its quality and size, and
the training configuration.

## Running the notebooks

- Open a notebook in Jupyter or Google Colab and run its cells from top to
  bottom.
- The notebooks are independent; you do not need to run them in a particular
  order.
- A Python environment with Jupyter and PyTorch is needed. The translation
  notebooks also import packages such as NumPy, pandas, matplotlib, and
  scikit-learn; install any missing packages in your environment.
- The models use a CUDA GPU when one is available and otherwise run on CPU.
  Training time will vary with the hardware and dataset size.
