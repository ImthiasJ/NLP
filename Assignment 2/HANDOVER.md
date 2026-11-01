# Handover: PS4 SMT assignment, Parts B, C and E-metrics

**Assignment:** AIML ZG521, Assignment 2, Part 1 (10 marks).
**Task:** English → German translation for a news agency that syndicates wire stories and headlines to European partners.
**Deliverable:** a single executed Jupyter notebook. Every cell must run from top to bottom with outputs visible.

Member 1 has already built the corpus (Part A), the evaluation and stress sets, the NMT side of Part D and the references. That work is in `A_D_member1.ipynb`, and its outputs are in `data/` and `outputs/`. **You own Parts B and C and the BLEU implementation for Part E.**

## 1. Who does what

| Part | Marks | Owner | Status |
|---|---|---|---|
| A: corpus, preprocessing, analysis | 1 | Member 1 | done |
| B: IBM Model 1 with EM, written from scratch | 2 | **Member 2** | to do |
| C: phrase extraction, phrase table, n-gram LM, own decoder | 3 | **Member 2** | to do |
| D: pretrained NMT comparison | 2 | Member 1 | NMT side done; the SMT row fills in automatically from your output files |
| E: BLEU implementation, written from scratch | part of 1 | **Member 2** | to do |
| E: eval set and stress set | part of 1 | Member 1 | done (`data/eval_set.csv`, `data/stress_set.csv`) |
| E: error-tagged SMT vs NMT table | part of 1 | Both | after your outputs exist |
| F: reflection, 250–400 words | 1 | Both | last |

## 2. Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m ipykernel install --user --name nlp-a2
```

No API keys or Hugging Face tokens are needed. Never put any in the notebook.

## 3. Input files you receive (`data/`)
All files are UTF-8 with one sentence per line. The `.en` and `.de` files are line-aligned.

| File | Lines | Notes |
|---|---|---|
| `train.tok.en` / `train.tok.de` | 10,000 | Lowercased, Moses-tokenized, 1–30 tokens. Tokens are separated by spaces, so `line.split()` gives the token list. |
| `test.tok.en` / `test.tok.de` | 500 | Held-out pairs, preprocessed the same way |
| `test.raw.en` / `test.raw.de` | 500 | The same pairs, untokenized and in original case. Use them as sacreBLEU references. |
| `lm_heldout.tok.de` | 500 | German sentences for LM perplexity. They are in neither train nor test. |
| `eval_set.csv` | 15 | Columns: `id, category, en, de_ref, source` |
| `stress_set.csv` | 8 | Columns: `id, phenomenon, en, de_ref, news_focus` |

To translate `eval_set` and `stress_set`, tokenize the English with **the same function** that was used for the corpus. It is defined in `A_D_member1.ipynb` (section A.3):

```python
import re, unicodedata
from sacremoses import MosesTokenizer
QUOTE_MAP = str.maketrans({'„': '"', '“': '"', '”': '"', '«': '"', '»': '"',
                           '‚': "'", '‘': "'", '’': "'", '–': '-', '—': '-'})
_MOSES = {'en': MosesTokenizer(lang='en'), 'de': MosesTokenizer(lang='de')}
def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()
def preprocess(text, lang):
    text = normalize(text).translate(QUOTE_MAP)
    return [tok.lower() for tok in _MOSES[lang].tokenize(text, escape=False)]
```

Shared config. Keep these values identical:
`SEED=42, MAX_LEN=30, N_TRAIN=10000, N_TEST=500, EM_ITERS=10, MAX_PHRASE_LEN=4, LM_ORDER=3, BEAMS=[1,2,5,10]`.

## 4. Files you deliver (`outputs/`)
Part D picks these up automatically. Re-running `A_D_member1.ipynb` then fills in the SMT row of the table and the charts.

| File | Content |
|---|---|
| `smt_test.de` | 500 lines. SMT output for `test.tok.en`, **detokenized** (`sacremoses.MosesDetokenizer(lang='de')`) |
| `smt_eval.de` | 15 lines. SMT output for `eval_set.en`, in row order |
| `smt_stress.de` | 8 lines. SMT output for `stress_set.en`, in row order |
| `smt_timing.json` | `{"beam_width": 5, "seconds_per_sentence": [ ... 500 floats ... ]}` for the test set, using your chosen beam width |

Keep exactly one output line per input line. If the decoder produces nothing, write an empty line.

## 5. Requirements and acceptance criteria

> **Notation warning:** the PDF's decoder score is `score(e) = λ1·log P(f|e) + λ2·log P_LM(e)`. Here **e is the German output and f is the English input**. This is the noisy-channel view, so the translation model is P(English | German). Say this explicitly in the notebook.

### Part B: IBM Model 1 (2 marks). No MT toolkits allowed (no NLTK `IBMModel1`, no GIZA++).
- [ ] Add a NULL token on the conditioning side. Initialize t(f|e) uniformly.
- [ ] Run EM for `EM_ITERS` iterations. Record the corpus log-likelihood (or perplexity) after each iteration and **plot it**. The log-likelihood must increase monotonically.
- [ ] Train **both directions** (EN→DE and DE→EN). Symmetrization in Part C needs both.
- [ ] Show a table of the **top 5 translation candidates for at least 10 source words**. Pick news-relevant words, e.g. *government, bank, market, president, crisis, war, economy, election, minister, said*.
- [ ] Plot **alignment heatmaps (matrices of t or posteriors) for at least 3 sentence pairs**. Include one with a German compound and one with verb-final word order.
- [ ] Discuss **at least 2 limitations** using examples from your own output:
  - No fertility: a compound such as *Bundesverfassungsgericht* cannot align to three English words.
  - No distortion: the position of the verb-final German verb is ignored.
  - The NULL word and the garbage-collector effect on rare words.
- Efficiency tip: store t as a dict of dicts or use integer ids. For 10k pairs of up to 30 tokens, pure Python takes a few minutes for 10 iterations, and NumPy per sentence is faster.

### Part C: phrase-based SMT (3 marks, the largest part)
- [ ] Viterbi alignments from both directions. Symmetrize with intersection, union and grow-diag(-final), and say which one you use and why.
- [ ] Extract phrase pairs consistent with the alignment (Koehn et al. 2003), with max length `MAX_PHRASE_LEN`. **Report the number of extracted phrase pairs.**
- [ ] Phrase table: φ(f|e) = count(e,f) / count(e). Optionally add the inverse direction and lexical weights. Prune to the top-k translations per source phrase to keep decoding fast.
- [ ] German **trigram LM** with smoothing (Kneser-Ney via `nltk.lm.KneserNeyInterpolated`, or your own add-k). Train it on `train.tok.de`. **Report perplexity on `lm_heldout.tok.de`.** KenLM is hard to install on Windows, so avoid it.
- [ ] **Decoder: your own code.** NLTK may only be used for utilities. A beam/stack decoder over source spans is enough: monotone, or with a small distortion limit. Score each hypothesis as `λ1·TM + λ2·LM`, optionally plus a word penalty. Pass unknown source words through unchanged; this matters for the OOV analysis.
- [ ] Beam-width sweep over `BEAMS = [1, 2, 5, 10]`, measuring BLEU and decode time. Show it as **one chart** with time on one axis and BLEU on the other.
- [ ] **At least 10 example translations**, each showing source, reference and SMT output.
- Expect low BLEU (roughly 3–12) from 10k pairs. That is normal. The analysis is what earns marks.

### Part E: BLEU written from scratch
- [ ] `sentence_bleu(hyp_tokens, ref_tokens)` and `corpus_bleu(list_hyp, list_ref)`:
  - Clipped (modified) n-gram precision for n = 1 to 4.
  - Geometric mean of the precisions.
  - Brevity penalty `BP = exp(1 − r/c)` when c < r.
  - Smoothing for sentence-level BLEU, e.g. add-1 on n-gram counts. State which method you use.
- [ ] **Validate against sacreBLEU:** use `sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')` on the same pre-tokenized, lowercased text, and show the scores agree (difference below about 0.1). Optionally also compare against NLTK's `corpus_bleu`.
- [ ] Use sacreBLEU with `lowercase=True` on detokenized text for the cross-system table in Part D. SMT output is lowercased, so cased BLEU would be unfair to it.

## 6. Merging and submission checklist
1. Assemble the final notebook in this order: Header and config → A → B → C → D → E (BLEU implementation, eval/stress tables, error tags) → F (reflection) → References.
2. Both people fill in `outputs/ratings.csv`, giving Adequacy and Fluency scores from 1 to 5 for SMT and NMT on 40 rows. Both people also tag the errors in `outputs/stress_error_tags.csv` using these categories:
   - wrong lexical choice
   - reordering error
   - dropped word
   - hallucinated/inserted word
   - untranslated word
   - none
3. Reflection, 250–400 words:
   - (a) low-resource pair with limited compute: SMT or NMT?
   - (b) this high-volume news scenario: SMT or NMT?
   - Back both answers with our own numbers, and name one major limitation of the chosen approach.
4. Restart the kernel and run all cells. Check that every output is visible and that no secrets appear anywhere.
5. File name: `PS4_Group<N>_AIMLZG521_SMT_Assignment.ipynb`. The Instructions PDF suggests `Assignment2_Programming_Group_<N>.ipynb` instead, so agree on one. The header must include the PS number, group number, names and IDs.

## 7. Key references for your parts
These are already listed in the References cell.
- Brown et al. 1993: IBM models
- Och & Ney 2003: alignment and symmetrization
- Koehn, Och & Marcu 2003: phrase-based SMT
- Kneser & Ney 1995: LM smoothing
- Papineni et al. 2002: BLEU
- Post 2018: sacreBLEU
- Michael Collins' lecture notes on phrase-based decoding: http://www.cs.columbia.edu/~mcollins/pb.pdf
