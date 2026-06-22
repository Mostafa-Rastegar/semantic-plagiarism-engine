# Semantic Plagiarism Engine

موتور تشخیص سرقت ادبی و اسناد مشابه — پروژه سوم درس داده‌کاوی.

این پروژه دو مسیر را پیاده‌سازی و مقایسه می‌کند:

1. **MinHash + LSH** روی word-shingles
2. **TF-IDF weighted SimHash** روی توکن‌ها/n-gramها

تمام الگوریتم‌های اصلی (MinHash، LSH، SimHash) از صفر نوشته شده‌اند و فقط
به `numpy`/`pandas`/`scikit-learn` (برای ابزارهای کمکی) وابسته‌اند.

## نصب

```bash
python -m venv .venv
# linux/mac
source .venv/bin/activate
# windows
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## ساختار پروژه

```
semantic-plagiarism-engine/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .github/workflows/tests.yml
├── docs/
│   └── report.md
├── data/
│   ├── sample_corpus/        # نمونه اسناد + pairs_demo.csv
│   ├── raw/                  # gitignore — دیتاست بزرگ
│   └── processed/            # gitignore
├── src/plagiarism_engine/
│   ├── __init__.py
│   ├── preprocessing.py      # نرمال‌سازی، tokenization، shingles
│   ├── minhash.py            # MinHash از صفر
│   ├── lsh.py                # Banding + bucket hashing
│   ├── simhash.py            # TF-IDF SimHash از صفر
│   ├── dataset.py            # لود corpus و pairs csv
│   ├── evaluation.py         # precision/recall/F1 و timing
│   └── cli.py                # argparse با ۳ subcommand
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   └── test_engine.py        # unit testهای pytest
└── outputs/
    ├── metrics.csv
    ├── candidates.csv
    └── two_file_compare.json
```

## دستورهای CLI

سه دستور اصلی:

### ۱. compare — مقایسه دو فایل

```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_02.txt \
    --shingle-size 2 \
    --output outputs/two_file_compare.json
```

خروجی JSON شامل سه شباهت: Jaccard دقیق، تخمین MinHash و SimHash similarity.

### ۲. corpus — جست‌وجوی اسناد مشابه در یک پوشه

```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.1 \
    --shingle-size 2 \
    --num-perm 128 \
    --bands 64 \
    --output outputs/candidates.csv
```

از MinHash+LSH برای پیدا کردن کاندیدها استفاده می‌کند. خروجی CSV با ستون‌های
`doc_a, doc_b, minhash_jaccard`.

### ۳. pairs — ارزیابی روی دیتاست برچسب‌دار

```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/train.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --limit 5000 \
    --shingle-size 2 \
    --num-perm 128 \
    --output outputs/metrics.csv
```

برای هر آستانه در sweep، دقت/بازیابی/F1 و زمان اجرای هر دو روش (MinHash و
SimHash) را محاسبه و در `outputs/metrics.csv` ذخیره می‌کند.

## انتخاب پارامترها

| پارامتر | پیش‌فرض | توضیح |
| --- | --- | --- |
| `--shingle-size` | 3 | اندازه word shingle. برای متن کوتاه/paraphrase معمولاً ۲ بهتر است |
| `--num-perm` | 128 | تعداد permutations در MinHash (طول signature) |
| `--bands` | 32 | تعداد bandها در LSH. باید `num_perm` بر آن بخش‌پذیر باشد |
| `--ngram` | 1 | اندازه n-gram برای SimHash |
| `--threshold` | 0.25 | آستانه شباهت MinHash برای حالت corpus |

رابطه bands و rows با آستانه:

```
threshold ≈ (1/bands)^(1/rows_per_band)
```

با `num_perm=128` :

| bands | rows | threshold |
| ----- | ---- | --------- |
| 8     | 16   | ≈ 0.84    |
| 16    | 8    | ≈ 0.71    |
| 32    | 4    | ≈ 0.42    |
| 64    | 2    | ≈ 0.125   |

برای detection روی paraphrase معمولاً `bands` بزرگ‌تر (آستانه پایین‌تر) لازم است.

## اجرای تست‌ها

```bash
pytest -q
```

## دیتاست‌ها

- `data/sample_corpus/` — هشت سند نمونه (ML / Eiffel / Photosynthesis / Cricket / Quantum) به‌علاوه‌ی `pairs_demo.csv` با ۲۰ جفت برچسب‌دار.
- برای ارزیابی واقعی Quora Question Pairs را دانلود و در `data/raw/quora/train.csv` قرار دهید (gitignore شده).
- جایگزین‌ها: Stack Exchange Duplicates، PAN-PC-11.

## گزارش

گزارش فنی فارسی در `docs/report.md` قرار دارد و شامل بخش‌های:

1. مقدمه
2. پیش‌پردازش
3. MinHash و LSH
4. SimHash + TF-IDF
5. نتایج تجربی
6. تحلیل خطا
