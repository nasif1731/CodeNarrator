# 🧠 CodeNarrator — Automatic Code-to-Docstring Generator

A Generative AI–powered pipeline that automatically generates meaningful docstrings and summaries for source code functions, using tokenization, embedding, and sequence modeling techniques.

## 🚀 Overview

CodeNarrator takes raw code (Python functions, modules, etc.) as input and produces natural-language docstrings that explain the functionality, purpose, and parameters. It combines Byte Pair Encoding (BPE) tokenization, Word2Vec embeddings, and a Recurrent Neural Network (RNN)–based language model (LSTM/GRU) to learn relationships between code syntax and documentation semantics.

## ⚙️ Architecture

| Stage                | Description                                                                                      | Tools / Models                     |
|----------------------|--------------------------------------------------------------------------------------------------|------------------------------------|
| 1. Tokenization      | Trains a domain-specific BPE tokenizer for code and documentation corpora.                     | sentencepiece, pickle              |
| 2. Embedding Generation | Converts tokens into dense semantic vectors using Word2Vec.                                   | gensim.models.Word2Vec            |
| 3. Sequence Modeling | Trains an LSTM model to map code embeddings → docstring embeddings.                             | tensorflow, keras                  |
| 4. Evaluation        | Measures quality using BLEU, ROUGE, and cosine similarity metrics.                              | sacrebleu, sklearn                 |
| 5. Inference         | Given new code input, generates a predicted docstring with language model decoding.            | streamlit or command line          |

## 🧩 Project Structure

```
CodeNarrator/
│
├── app.py                     # Main pipeline script
├── bpe_models/                # Trained BPE tokenizers (.pkl)
├── lstm_models/               # Trained LSTM weights (.pkl)
├── w2v_models/                # Word2Vec embedding models
├── vocabs/                    # Vocabulary files (.json)
└── Assignment 1 Report.pdf     # Detailed technical report
```

## 🛠️ Installation & Setup

1. **Create environment**
   ```bash
   conda create -n codenarrator python=3.10
   conda activate codenarrator
   pip install -r requirements.txt
   ```

2. **Run the pipeline**
   ```bash
   python app.py
   ```

3. **Or launch Streamlit UI**
   ```bash
   streamlit run app.py
   ```

## 🧾 Example

**Input (Python Function):**
```python
def add_numbers(a, b):
    return a + b
```

**Generated Docstring:**
```python
"""Returns the sum of two numeric inputs a and b."""
```

## 🧮 Model Files

| Folder           | Purpose                                                               |
|------------------|-----------------------------------------------------------------------|
| bpe_models/      | Tokenization models (merges_code.pkl, merges_doc.pkl)                |
| w2v_models/      | Code & doc embedding models (w2v_code.pkl, w2v_doc.pkl)             |
| lstm_models/     | Trained seq2seq LSTM for code→docstring mapping                      |
| vocabs/          | BPE vocabulary JSONs for encoding / decoding                          |

## 🧠 Evaluation Metrics

- **BLEU** — Translation quality between generated & reference docstrings
- **ROUGE-L** — Overlap between predicted and ground-truth text
- **Cosine Similarity** — Semantic similarity between embedding spaces
- **Perplexity** — Model confidence on unseen samples


## 📚 References

- Python Functions with Docstrings Dataset (Kaggle)
- Mikolov et al., Word2Vec: Efficient Estimation of Word Representations
- Sennrich et al., Neural Machine Translation of Rare Words with Subword Units
- Hochreiter & Schmidhuber, Long Short-Term Memory
