# گزارش فنی پروژه سوم داده‌کاوی

## موتور تشخیص سرقت ادبی و اسناد مشابه

**درس:** داده‌کاوی — بهار ۱۴۰۵
**استاد:** دکتر فاطمه شاکری

---

## ۱. مقدمه

هدف این پروژه طراحی و پیاده‌سازی یک موتور خط‌فرمان برای تشخیص اسناد مشابه،
کپی‌شده یا paraphrase شده است. در عمل، در سامانه‌های واقعی (سامانه‌های ضد سرقت
ادبی، خوشه‌بندی خبر، de-duplication جست‌وجو و سامانه‌های پرسش‌وپاسخ) لازم است
از میان میلیون‌ها سند، جفت‌هایی پیدا شوند که محتوای آن‌ها بسیار شبیه است،
حتی اگر واژه‌ها دقیقاً یکسان نباشند.

دو الگوریتم اصلی پیاده‌سازی و مقایسه شده‌اند:

1. **MinHash + LSH** روی word-shingles
2. **TF-IDF weighted SimHash** روی توکن‌ها/n-gramها

هیچ کتابخانه آماده‌ای برای MinHash، LSH یا SimHash استفاده نشده و فقط ابزارهای
عمومی `numpy`، `pandas` و در حد بسیار جزئی `scikit-learn` مجاز بوده‌اند.

---

## ۲. پیش‌پردازش متن

ماژول `preprocessing.py` مراحل زیر را انجام می‌دهد:

1. **نرمال‌سازی unicode** و حذف diacritic‌ها (مهم برای فارسی و عربی).
2. **یکسان‌سازی letterها**: `ي → ی`، `ك → ک`.
3. **lowercase** و حذف punctuation (با regex یونیکد).
4. **collapse whitespace**.
5. **stopword removal** برای انگلیسی و فارسی با لیست‌های دستی کوچک.
6. **shingling** کلمه‌ای با اندازه‌ی پیش‌فرض ۳ (قابل تغییر تا ۵).
7. **مدیریت سندهای کوتاه**: اگر `len(tokens) < k`، اندازه‌ی k به طول سند کاهش
   داده می‌شود تا مجموعه‌ی خالی برنگردد.

### چرا shingle می‌سازیم؟

shingles مجموعه‌ای از زیررشته‌های کلمه‌ای متوالی هستند. مزیت آن این است که
ترتیب کلمات را تا حدی حفظ می‌کنند، در حالی‌که bag-of-words ترتیب را به‌کلی از
دست می‌دهد. این برای تشخیص paraphrase نسبتاً قوی است.

### چرا Jaccard مستقیم گران است؟

برای مجموعه‌ای از n سند، محاسبه‌ی $J(A_i, A_j)$ روی همه‌ی جفت‌ها O(n²) جفت
دارد و هر مقایسه نیز O(|A|+|B|) است. در عمل برای n=10⁵ این عملاً امکان‌پذیر
نیست. MinHash و LSH دقیقاً برای این مشکل طراحی شده‌اند.

---

## ۳. MinHash از صفر

### ایده

برای دو مجموعه A و B، یک permutation تصادفی روی universe انتخاب می‌کنیم و
به این پرسش پاسخ می‌دهیم: «حداقل element در A تحت این permutation با حداقل
element در B تحت همان permutation برابر است یا نه؟»

می‌توان نشان داد:

$$
\Pr[\min_\pi(A) = \min_\pi(B)] = J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

اگر این آزمایش را K بار با Kتا permutation مستقل تکرار کنیم، **میانگین تعداد
برابری‌ها** برآوردگر اریب‌نشده‌ی شباهت Jaccard است. واریانس برآورد متناسب با
$1/K$ است.

### پیاده‌سازی

به جای ساختن permutation کامل روی universe (که ممکن است بسیار بزرگ باشد) از
خانواده‌ی universal hashing استفاده می‌کنیم:

$$
h_i(x) = (a_i \cdot x + b_i) \mod p
$$

که در آن:

- $x = \text{md5}(\text{shingle})[:32\text{bit}]$
- $p = 2^{61} - 1$ (Mersenne prime)
- $a_i \in [1, p-1]$ و $b_i \in [0, p-1]$ با seed ثابت RNG تولید می‌شوند

برای هر سند، signature یک بردار با طول `num_perm` (پیش‌فرض ۱۲۸) است که هر
خانه‌اش حداقل مقدار $h_i$ روی تمام shingleهای آن سند است.

```python
hashed = (self.a[:, None] * x[None, :] + self.b[:, None]) % _LARGE_PRIME
sig = hashed.min(axis=1)
```

استفاده از broadcasting در numpy باعث می‌شود محاسبه‌ی همه‌ی permutations برای
یک سند فقط با یک عمل ماتریسی انجام شود.

### تخمین Jaccard از روی signatures

$$
\hat{J}(A, B) = \frac{1}{K} \sum_{i=1}^{K} \mathbb{1}[h_i^{\min}(A) = h_i^{\min}(B)]
$$

که در کد:

```python
def estimated_jaccard(sig_a, sig_b):
    return float(np.mean(sig_a == sig_b))
```

---

## ۴. LSH (Locality-Sensitive Hashing)

### مشکل

حتی اگر MinHash هر مقایسه را به O(K) کاهش بدهد، تعداد جفت‌ها هنوز O(n²)
است. LSH این تعداد را به O(n) (با ضریبی که به آستانه بستگی دارد) کاهش
می‌دهد.

### راه‌حل: banding

signature طول K را به b تا band با r ردیف تقسیم می‌کنیم (K = b·r). هر band
به یک bucket هش می‌شود. دو سند **کاندید** هستند اگر در حداقل یک band به یک
bucket بیفتند.

احتمال این‌که دو سندِ با Jaccard واقعی s کاندید شوند:

$$
P_\text{candidate}(s) = 1 - (1 - s^r)^b
$$

این یک S-curve است که در $s \approx (1/b)^{1/r}$ به سرعت از نزدیک صفر به نزدیک یک می‌رود.

### پیاده‌سازی

- ساختمان داده: یک `defaultdict(list)` به ازای هر band.
- کلید bucket = `md5(band_vals.tobytes())` (برای stable و بدون collision عملی).
- در batch mode تابع `candidate_pairs()` همه‌ی جفت‌های هم‌سطل را برمی‌گرداند.

### مثال انتخاب پارامتر

برای `num_perm = 128`:

| bands | rows | threshold تقریبی |
|------|------|-----------------|
| 8    | 16   | 0.84            |
| 16   | 8    | 0.71            |
| 32   | 4    | 0.42            |
| 64   | 2    | 0.125           |

برای داده‌ی paraphrase که Jaccard واقعی پایین (0.2-0.4) دارد، انتخاب
`bands=64` منطقی است.

---

## ۵. TF-IDF Weighted SimHash

### ایده

SimHash هر سند را به یک fingerprint با طول ثابت (۶۴ بیت) نگاشت می‌کند، طوری
که سندهای مشابه fingerprint با Hamming distance کم داشته باشند.

### الگوریتم

برای هر سند:

1. تبدیل به token (یا n-gram).
2. وزن TF-IDF هر token محاسبه می‌شود.
3. بردار $v \in \mathbb{R}^{64}$ مقدار اولیه صفر می‌گیرد.
4. برای هر token $w$ با وزن $W_w$ و hash $h(w)$ ۶۴ بیتی:
   - برای هر بیت $i$:
     - اگر بیت i ام در h(w) برابر ۱ باشد، $v_i \mathrel{+}= W_w$
     - در غیر این صورت $v_i \mathrel{-}= W_w$
5. fingerprint نهایی: بیت i ام = ۱ اگر $v_i > 0$، در غیر این صورت ۰.

### TF-IDF محاسبه

با smoothing کلاسیک:

$$
\text{idf}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1
$$

$$
W_w = \text{tf}(w, d) \cdot \text{idf}(w)
$$

### پیاده‌سازی برداری

نسخه‌ی برداری با numpy:

```python
hashes = np.array([hash64(t) for t in toks], dtype=np.uint64)
bit_positions = np.arange(64, dtype=np.uint64)
bits = ((hashes[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
signed = (2 * bits - 1).astype(np.float64) * weights[:, None]
v = signed.sum(axis=0)
```

### شباهت

$$
\text{sim}(d_1, d_2) = 1 - \frac{\text{Hamming}(f_1, f_2)}{64}
$$

که در `[0, 1]` نرمالایز شده است.

---

## ۶. ساختار CLI

سه subcommand طراحی شده است که در فایل `cli.py` پیاده‌سازی شده‌اند:

1. **`compare`** — مقایسه‌ی دو فایل و گزارش هر سه شباهت (Jaccard دقیق،
   تخمین MinHash، SimHash similarity).
2. **`corpus`** — اجرای MinHash+LSH روی یک پوشه از فایل‌های `.txt`، خروجی
   لیست کاندیدها به‌علاوه‌ی شباهت تخمینی.
3. **`pairs`** — ارزیابی بر روی یک CSV جفتی برچسب‌دار (مانند Quora) با
   sweep روی آستانه‌ها و گزارش P/R/F1 و زمان اجرا.

نمونه اجرا در `README.md` آمده است.

---

## ۷. نتایج تجربی

### الف) corpus نمونه

روی پوشه‌ی `data/sample_corpus` (هشت سند: دو جفت paraphrase شده ML و
Eiffel به‌علاوه‌ی چهار سند مستقل) با تنظیمات `num_perm=128, bands=64,
shingle_size=2, threshold=0.1`:

```
docs=8  candidate_pairs=2  >=threshold(0.1)=2
doc_01, doc_02, 0.4688   # ML paraphrase
doc_03, doc_04, 0.3516   # Eiffel paraphrase
```

LSH تنها ۲ کاندید برگرداند که هر دو صحیح بودند (precision = 1.0). جفت
photosynthesis (Jaccard واقعی = 0.017) به‌درستی فیلتر شد.

### ب) ارزیابی روی pairs_demo.csv

روی ۲۰ جفت سؤال (۱۰ duplicate + ۱۰ غیر duplicate):

| روش    | بهترین threshold | P | R | F1 |
|--------|------------------|---|---|----|
| MinHash | 0.20 | 1.00 | 0.60 | **0.75** |
| SimHash | 0.60 | 0.90 | 0.90 | **0.90** |

روی این داده‌ی paraphrase (با تنوع کلامی بالا و طول کم) SimHash برتری
محسوسی دارد، چون SimHash به vocabulary level مشابهت حساس است و TF-IDF
کلمات کم‌اطلاعات را خودکار downweight می‌کند، در حالی‌که MinHash روی word-
shingle عمل می‌کند و paraphrase ترتیب کلمات را به‌هم می‌ریزد.

### ج) زمان اجرا

روی همان ۲۰ جفت:

- MinHash: ~ 0.01s در هر آستانه
- SimHash: ~ 0.01s در هر آستانه

اختلاف در مقیاس بزرگ مشهود می‌شود: SimHash هزینه‌ی محاسباتی بالاتری دارد
(۶۴ بیت × |vocab| ضرب)، اما constant factor آن کوچک‌تر است چون batch matrix
multiplication است.

> برای ارزیابی روی Quora Question Pairs کافی است فایل `train.csv` در
> `data/raw/quora/` قرار گیرد و دستور:
>
> ```bash
> python -m plagiarism_engine.cli pairs \
>     --pairs data/raw/quora/train.csv \
>     --text-col-a question1 --text-col-b question2 \
>     --label-col is_duplicate --limit 5000 \
>     --shingle-size 2 --num-perm 128 \
>     --output outputs/metrics.csv
> ```

---

## ۸. تحلیل خطا

سه نمونه از خطاها روی pairs_demo:

**۱. False Negative (MinHash, t=0.3):**
- `"What is machine learning?"` ↔ `"Explain machine learning in simple terms."`
- شینگل‌های ۲-کلمه‌ای مشترک کم هستند چون عبارت دوم structure متفاوتی دارد. Jaccard تخمینی ≈ 0.18، زیر آستانه.
- **چاره:** کاهش shingle_size به ۱ (unigram) یا استفاده از SimHash که به word-level matching حساس‌تر است.

**۲. False Positive (SimHash, t=0.5):**
- `"How to swim freestyle?"` ↔ `"Capital of Brazil?"`
- هر دو متن کوتاه و TF-IDF dimensionality کم. fingerprint ۶۴ بیتی برای متن
  کوتاه پایدار نیست — چند توکن کافی نیست تا signs پایدار شوند.
- **چاره:** افزایش آستانه به ≥0.6 یا استفاده از padding یا حداقل length filter.

**۳. False Negative (SimHash، paraphrase شدید):**
- `"How does a car engine work?"` ↔ `"Explain how internal combustion engines work."`
- کلمات کلیدی متفاوت ("car engine" vs "internal combustion engines") و TF-
  IDF دو کلمه را مستقل می‌بیند.
- **چاره:** embeddingهای معنایی (word2vec/BERT)، که از scope این پروژه خارج
  است.

### جمع‌بندی

- روی متن کوتاه و paraphrase شدید، هر دو روش محدودیت دارند.
- روی متن بلندتر (>۱۰۰ کلمه) MinHash+LSH عموماً precision بالاتری دارد.
- SimHash برای real-time deduplication بهتر است چون fingerprint ثابت ۶۴
  بیتی دارد و مقایسه‌ی آن O(1) است.
- بهترین استراتژی عملی: ترکیب دو روش (LSH برای shortlist، SimHash برای
  rescoring).

---

## ۹. ساختار repository

```
src/plagiarism_engine/
├── __init__.py
├── preprocessing.py     # نرمال‌سازی، tokenization، shingles
├── minhash.py           # MinHasher class با universal hashing
├── lsh.py               # LSHIndex با banding و bucket dictهای دینامیک
├── simhash.py           # SimHasher با TF-IDF، vectorized
├── dataset.py           # corpus loader و pairs csv loader
├── evaluation.py        # P/R/F1، timing، sweep
└── cli.py               # argparse با ۳ subcommand
tests/
└── test_engine.py       # ۱۴ unit test، همگی pass
```

اجرای تست‌ها:

```bash
pytest -q
# ..............                              [100%]
# 14 passed in 0.28s
```

---

## ۱۰. کارهای آتی پیشنهادی

1. اضافه کردن sentence-level embedding برای semantic similarity (مثلاً
   sentence-BERT) به‌عنوان مسیر سوم و مقایسه با LSH/SimHash.
2. پیاده‌سازی character n-gram به جای word shingle برای زبان‌هایی با
   tokenization پیچیده (فارسی، چینی).
3. caching signatureها روی disk با joblib برای avoid recomputation روی
   datasetهای بزرگ.
4. ارزیابی روی PAN-PC-11 (دیتاست رسمی plagiarism detection) و گزارش
   precision/recall در سطح passage-level (نه pair-level).
