# Cryptographic Encoding Classifier (LSTM)

A pytorch POC that I built to understand better practical usage of LSTM. The aim of this project is to identify the encoding type of a given string using a **Long Short-Term Memory (LSTM)** neural network. The model is trained to distinguish between character-level patterns in different type of encodings:

- **Caesar**
- **Rot13**
- **MD5**

## Overview

The project uses a character-level sequence modeling approach. Since cryptographic ciphers rely on specific shifts or hashing structures, an LSTM is used to capture these long-term dependencies across the characters of an encoded string.

## Installation & Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/cryptographic-classifier.git
    cd cryptographic-classifier
    ```

2. **Install Dependencies:**
   Ensure you have Python 3.10+ and the following libraries:

    ```bash
    pip install torch pandas scikit-learn
    ```

3. **Prepare the Data:**
   The project automatically pulls the dataset from a remote gist, but ensures your `src` folder is properly configured with `__init__.py` to allow local imports.

## Dataset

I built the [dataset](https://gist.githubusercontent.com/FlavRomano/2f0b37a3d0d5b7230d548a0de563c4a0/raw/8ace8e7eb27e6432b963659d68c02dccb35ab108/en_anagram_dictionary_encoding.csv) starting from a plain anagram dataset, that I enriched with two more columns.

### 1. `anagram` (The Plain Text)

The original column contains the **original English word** (the "Ground Truth").

- **Purpose:** It serves as the source for the encoding process.
- **Example:** `aardvark`, `hello`, `apple`.
- **Role in Model:** The model actually **ignores** this column. We are training the model to recognize the _encoding style_ of the scrambled text, not to translate it back to the original word.

### 2. `encoded` (The Feature/Input)

This is the **scrambled or hashed string** produced by applying a specific algorithm to the `anagram`.

- **Purpose:** This is the `X` (input).
- **Data Characteristics:**
    - **Caesar/Rot13:** The length matches the original word. The character set remains alphabetic (a-z).
    - **MD5:** The length is always exactly **32 characters**, consisting of hexadecimal values (0-9 and a-f).
- **Example:** For the word `aardvark`, the encoded value is `nneqinex` (Rot13).

### 3. `encoding` (The Label/Target)

This column identifies **which algorithm** was used to transform the anagram into the encoded string.

- **Purpose:** This is the `y` (target label) for your classification task.
- **Values:** `caesar`, `rot13`, or `md5`.
- **Role in Model:** The `EncodingsDataset` converts these string labels into integers (e.g., `0`, `1`, `2`) so the `CrossEntropyLoss` function can function properly.

## Model Architecture

The model is built using `torch.nn` with the following layers:

- **Embedding Layer:** Maps characters to a `128-dimensional` space.
- **LSTM Layer:** A recurrent layer with a `256-dimensional` hidden state to process the sequence.
- **Linear Layer:** A fully connected head that maps the hidden state to the 3 output classes.

**Total Trainable Parameters:** ~230,000 (Varies based on vocab size).

## Performance

- **Dataset Size:** 72,000+ samples.
- **Max Sequence Length:** 32 (optimized for MD5 hash lengths).
- **Classification:** 3 Classes (Caesar, Rot13, MD5).

The model achieved a test accuracy of 97.99%

## Usage

### Training

The notebook includes a safety check. If `model_weights.pth` exists, it will skip training and load the weights. To force a re-train, simply delete the `.pth` file.

### Inference

You can use the `Utils` class to predict the encoding of any string:

```python
from src.utils import Utils

word = "nneqinex"
result = Utils.predict_encoding(word, model, train_dataset, device)
print(f"Detected: {result}")
```
