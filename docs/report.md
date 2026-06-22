# گزارش فنی پروژه سوم داده‌کاوی

## موتور تشخیص سرقت ادبی و اسناد مشابه

**درس:** داده‌کاوی — بهار ۱۴۰۵
**استاد:** دکتر فاطمه شاکری

---

## ۱. مقدمه

هدف این پروژه طراحی و پیاده‌سازی یک موتور خط‌فرمان برای تشخیص اسناد مشابه،
کپی‌شده یا paraphrase شده است. در سامانه‌های واقعی (سامانه‌های ضد سرقت ادبی،
خوشه‌بندی خبر، de-duplication جست‌وجو و سامانه‌های پرسش‌وپاسخ) لازم است از
میان میلیون‌ها سند، جفت‌هایی پیدا شوند که محتوای آن‌ها بسیار شبیه است، حتی
اگر واژه‌ها دقیقاً یکسان نباشند.

دو الگوریتم اصلی پیاده‌سازی و مقایسه شده‌اند:

1. **MinHash + LSH** روی word-shingles
2. **TF-IDF weighted SimHash** روی توکن‌ها/n-gramها

هیچ کتابخانه آماده‌ای برای MinHash، LSH یا SimHash استفاده نشده و فقط ابزارهای
عمومی `numpy`، `pandas` و `scikit-learn` به‌عنوان وابستگی‌های جانبی استفاده
شده‌اند.

---

## ۲. پیش‌پردازش متن

ماژول `preprocessing.py` مراحل زیر را انجام می‌دهد:

1. **نرمال‌سازی unicode** و حذف diacritic‌ها (مهم برای فارسی و عربی).
2. **یکسان‌سازی letterها:** `ي → ی`، `ك → ک`.
3. **lowercase** و حذف punctuation با regex یونیکد.
4. **collapse whitespace**.
5. **stopword removal** برای انگلیسی و فارسی (لیست دستی).
6. **shingling** کلمه‌ای با اندازه‌ی پیش‌فرض ۳ (قابل تنظیم).
7. **مدیریت سندهای کوتاه:** اگر `len(tokens) < k`، مقدار k به طول سند کاهش
   می‌یابد تا مجموعه‌ی خالی برنگردد و Jaccard NaN نشود.

### چرا shingle می‌سازیم؟

shingleها مجموعه‌ای از زیررشته‌های کلمه‌ای متوالی هستند؛ ترتیب کلمات را تا
حدی حفظ می‌کنند، در حالی‌که bag-of-words ترتیب را به‌کلی از دست می‌دهد. این
خاصیت برای تشخیص paraphrase کمی نسبتاً قوی است (هرچه شینگل بزرگ‌تر، حساسیت
به paraphrase بیشتر).

### چرا Jaccard مستقیم گران است؟

برای مجموعه‌ای از n سند، محاسبه‌ی $J(A_i, A_j)$ روی همه‌ی جفت‌ها O(n²) جفت
دارد و هر مقایسه O(|A|+|B|) است. برای n=10⁵ این عملاً امکان‌پذیر نیست.
MinHash و LSH دقیقاً برای حل این مسأله طراحی شده‌اند.

---

## ۳. MinHash از صفر

### ایده

برای دو مجموعه A و B و یک permutation تصادفی $\pi$ روی universe:

$$
\Pr_\pi[\min_\pi(A) = \min_\pi(B)] = J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

اگر K بار با Kتا permutation مستقل این آزمایش را تکرار کنیم، **میانگین تعداد
برابری‌ها** برآوردگرِ اریب‌نشده‌ی Jaccard است و واریانس آن از مرتبه‌ی $1/K$
است.

### پیاده‌سازی

به جای ساختن permutation کامل روی universe (بسیار بزرگ) از خانواده‌ی universal
hashing استفاده می‌کنیم:

$$
h_i(x) = (a_i \cdot x + b_i) \mod p
$$

- $x = \text{md5}(\text{shingle})[:32\text{bit}]$
- $p = 2^{61} - 1$ (Mersenne prime)
- $a_i \in [1, p-1]$ و $b_i \in [0, p-1]$ با seed ثابت تولید می‌شوند

برای هر سند، signature یک بردار با طول `num_perm` (پیش‌فرض ۱۲۸) است که هر
خانه‌اش حداقل مقدار $h_i$ روی تمام shingleهای آن سند است.

```python
hashed = (self.a[:, None] * x[None, :] + self.b[:, None]) % _LARGE_PRIME
sig = hashed.min(axis=1)
```

broadcasting در numpy اجازه می‌دهد همه‌ی permutationها برای یک سند با یک
عمل ماتریسی محاسبه شوند.

### تخمین Jaccard از روی signatures

$$
\hat{J}(A, B) = \frac{1}{K} \sum_{i=1}^{K} \mathbb{1}[h_i^{\min}(A) = h_i^{\min}(B)]
$$

```python
def estimated_jaccard(sig_a, sig_b):
    return float(np.mean(sig_a == sig_b))
```

---

## ۴. LSH (Locality-Sensitive Hashing)

### مشکل

حتی اگر MinHash هر مقایسه را به O(K) کاهش بدهد، تعداد جفت‌ها هنوز O(n²)
است. LSH تعداد مقایسه‌ها را به O(n) (با ضریبی که به آستانه بستگی دارد) کاهش
می‌دهد.

### راه‌حل: banding

signature طول K را به b تا band با r ردیف تقسیم می‌کنیم (K = b·r). هر band
به یک bucket هش می‌شود. دو سند **کاندید** هستند اگر در حداقل یک band به
یک bucket بیفتند.

احتمال این‌که دو سندِ با Jaccard واقعی s کاندید شوند:

$$
P_\text{candidate}(s) = 1 - (1 - s^r)^b
$$

این یک S-curve است که در $s \approx (1/b)^{1/r}$ به سرعت از نزدیک صفر به نزدیک یک می‌رود.

### پیاده‌سازی

- ساختمان داده: یک `defaultdict(list)` به ازای هر band.
- کلید bucket: `md5(band_vals.tobytes())` (stable و بدون collision عملی).
- در batch mode تابع `candidate_pairs()` همه‌ی جفت‌های هم‌سطل را برمی‌گرداند.

### پارامترها برای num_perm = 128

| bands | rows | threshold تقریبی |
|-------|------|------------------|
| 8     | 16   | 0.84             |
| 16    | 8    | 0.71             |
| 32    | 4    | 0.42             |
| 64    | 2    | 0.125            |

برای داده‌ی paraphrase که Jaccard واقعی پایین (0.2-0.4) دارد، انتخاب
`bands=64` (آستانه ≈ 0.125) منطقی است.

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
   - برای هر بیت i:
     - اگر بیت i ام در h(w) برابر ۱ باشد، $v_i \mathrel{+}= W_w$
     - در غیر این صورت $v_i \mathrel{-}= W_w$
5. fingerprint نهایی: بیت i ام = ۱ اگر $v_i > 0$، در غیر این صورت ۰.

### TF-IDF با smoothing

$$
\text{idf}(t) = \log\!\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1, \quad
W_w = \text{tf}(w, d) \cdot \text{idf}(w)
$$

### پیاده‌سازی برداری

نسخه‌ی numpy:

```python
hashes = np.array([hash64(t) for t in toks], dtype=np.uint64)
bit_positions = np.arange(64, dtype=np.uint64)
bits = ((hashes[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
signed = (2 * bits - 1).astype(np.float64) * weights[:, None]
v = signed.sum(axis=0)
```

### شباهت

$$
\text{sim}(d_1, d_2) = 1 - \frac{\text{Hamming}(f_1, f_2)}{64} \in [0, 1]
$$

---

## ۶. ساختار CLI

سه subcommand در فایل `cli.py`:

1. **`compare`** — مقایسه‌ی دو فایل و گزارش هر سه شباهت.
2. **`corpus`** — اجرای MinHash+LSH روی پوشه‌ی فایل‌های `.txt`.
3. **`pairs`** — ارزیابی روی CSV جفتی برچسب‌دار (Quora) با sweep آستانه.

نمونه اجرا در `README.md`.

---

## ۷. نتایج تجربی روی Quora Question Pairs

### تنظیمات

- دیتاست: Quora Question Pairs، ۵۰۰۰ سطر اول
- توزیع برچسب: ۱۹۱۲ duplicate (38.2%) و ۳۰۸۸ non-duplicate (61.8%)
- `shingle_size = 2`, `num_perm = 128`, `ngram = 1`, `use_idf = True`
- machine: Windows 11, Python 3.12

### جدول کامل (از `outputs/metrics.csv`)

| method  | threshold | precision | recall | F1     | TP   | FP   | FN   | TN   | runtime (s) |
|---------|-----------|-----------|--------|--------|------|------|------|------|-------------|
| minhash | 0.20      | 0.546     | 0.508  | **0.526** | 971  | 807  | 941  | 2281 | 0.97 |
| minhash | 0.30      | 0.557     | 0.370  | 0.445  | 708  | 563  | 1204 | 2525 | 0.85 |
| minhash | 0.40      | 0.558     | 0.272  | 0.366  | 520  | 412  | 1392 | 2676 | 0.98 |
| minhash | 0.50      | 0.561     | 0.204  | 0.300  | 391  | 306  | 1521 | 2782 | 1.06 |
| minhash | 0.60      | 0.602     | 0.151  | 0.242  | 289  | 191  | 1623 | 2897 | 1.40 |
| minhash | 0.70      | 0.654     | 0.115  | 0.195  | 219  | 116  | 1693 | 2972 | 1.14 |
| minhash | 0.80      | 0.724     | 0.095  | 0.167  | 181  | 69   | 1731 | 3019 | 1.12 |
| simhash | 0.20      | 0.382     | 1.000  | 0.553  | 1912 | 3088 | 0    | 0    | 1.06 |
| simhash | 0.30      | 0.383     | 1.000  | 0.553  | 1912 | 3087 | 0    | 1    | 1.03 |
| simhash | 0.40      | 0.385     | 0.998  | 0.555  | 1908 | 3053 | 4    | 35   | 0.93 |
| simhash | 0.50      | 0.407     | 0.973  | 0.574  | 1861 | 2714 | 51   | 374  | 0.79 |
| simhash | 0.60      | 0.475     | 0.864  | **0.613** | 1651 | 1828 | 261  | 1260 | 0.93 |
| simhash | 0.70      | 0.538     | 0.630  | 0.580  | 1204 | 1036 | 708  | 2052 | 1.16 |
| simhash | 0.80      | 0.591     | 0.295  | 0.394  | 564  | 390  | 1348 | 2698 | 1.05 |

### تحلیل

- **بهترین F1:**
  - MinHash: **0.526** در آستانه 0.20
  - SimHash: **0.613** در آستانه 0.60

- **رفتار رد precision/recall:**
  - MinHash با افزایش آستانه precision را بالا می‌برد (تا 0.72) ولی recall
    خیلی سریع افت می‌کند. این دقیقاً ویژگی روش‌های set-based روی متن‌های
    کوتاه است: paraphrase معمولاً بیش از نصفِ tokenها را عوض می‌کند، پس
    Jaccard واقعی کم می‌ماند.
  - SimHash در آستانه‌های پایین recall ≈ ۱ دارد ولی precision حدود 0.38 (که
    عملاً نسبت positiveها در داده است) — یعنی پیش‌بینی همیشه «duplicate».
    با بالا رفتن آستانه نقطه‌ی تعادل recall و precision در 0.60 ظاهر می‌شود.

- **زمان اجرا:** هر دو روش روی 5000 جفت، حدود ۱ ثانیه برای هر آستانه. عملاً
  یکسان. SimHash به‌نظر می‌رسد در عمل سریع‌تر باشد چون vector ۶۴ بیتی است،
  ولی برای متن کوتاه فاصله‌ی محسوسی دیده نشد.

- **برتری SimHash:** روی این دیتاست (سؤال‌های کوتاه با paraphrase شدید
  vocabulary-level) SimHash برتری روشن دارد. علت اصلی این است که SimHash
  روی token-level کار می‌کند و TF-IDF کلمات کم‌اطلاعات را خودکار downweight
  می‌کند، در حالی‌که MinHash روی word-shingle‌های ۲-تایی است که با هر
  paraphrase اصطلاحاً break می‌شوند.

- **برتری بالقوه‌ی MinHash:** روی متن‌های بلندتر (>۱۰۰ کلمه، مثلاً PAN-PC-11)
  که paraphrase معمولاً فقط بخشی از سند را تغییر می‌دهد، Jaccard روی
  shingleها بالا می‌ماند و MinHash precision بسیار بهتری می‌دهد. علاوه بر
  این، MinHash به‌خاطر LSH روی n سند O(n) candidate-generation دارد در
  حالی‌که SimHash به‌طور پیش‌فرض O(n²) است (مگر با hamming-LSH اضافه).

---

## ۸. تحلیل خطا

ابزار `scripts/extract_errors.py` با آستانه‌های بهینه‌ی هر روش، جفت‌هایی که
سیستم اشتباه پیش‌بینی کرده را استخراج می‌کند. فایل کامل در
`outputs/error_examples.json`.

### MinHash — False Positives (پیش‌بینی duplicate، برچسب 0)

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | What is the step by step guide to invest in share market in **india**? | What is the step by step guide to invest in share market? | 0.836 |
| 2 | What are the laws to change your status from a student visa to a green card in the US, how do they compare to the immigration laws in **Canada**? | ... in **Japan**? | 0.781 |
| 3 | Which is the best digital marketing institution in **banglore**? | Which is the best digital marketing institute in **Pune**? | 0.438 |
| 4 | What are the questions should not ask on Quora? | Which question should I ask on Quora? | 0.281 |

**تحلیل:** Quora این دو سؤال را به‌خاطر **تفاوت جزئی در یک کلمه** (یک کشور
یا یک شهر) duplicate نمی‌داند، ولی Jaccard بالا است (>0.7). از دید
الگوریتمی این False Positive نیست — تقریباً کل متن یکی است. این یک abuse-
case‌ی برچسب‌گذاری Quora است که bias annotation را نشان می‌دهد.

### MinHash — False Negatives (پیش‌بینی 0، برچسب duplicate)

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? | I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me? | 0.062 |
| 2 | How can I be a good geologist? | What should I do to be a great geologist? | 0.0 |
| 3 | What does manipulation mean? | What does manipulation means? | 0.0 |
| 4 | Why are so many Quora users posting questions that are readily answered on Google? | Why do people ask Quora questions which can be answered easily by Google? | 0.0 |

**تحلیل:** paraphrase شدید (synonym ها، تغییر ترتیب، تغییر فرم گرامری
"mean / means"). word-shingle ۲-تایی هیچ tuple مشترکی پیدا نمی‌کند. این
محدودیت ذاتی MinHash روی متن کوتاه است.

### SimHash — False Positives (پیش‌بینی duplicate، برچسب 0)

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | How can I increase the speed of my internet connection while using a VPN? | How can Internet speed be increased by hacking through DNS? | 0.719 |
| 2 | Which one dissolve in water quickly sugar, salt, methane and carbon dioxide? | Which fish would survive in salt water? | 0.609 |
| 3 | When do you use シ instead of し? | When do you use \"&\" instead of \"and\"? | 0.703 |
| 4 | Motorola: Can I hack my Charter Motorolla DCX3400? | How do I hack Motorola DCX3400 for free internet? | 0.719 |

**تحلیل:** SimHash به اسکلت جمله حساس است؛ هر دو جمله structure سؤالی
"how/when do you use/increase X instead of/by/...?" دارند. مدل bag-of-tokens
نمی‌تواند بفهمد که موضوع اصلی متفاوت است. این محدودیت ذاتی SimHash بدون
semantic embedding است.

### SimHash — False Negatives (پیش‌بینی 0، برچسب duplicate)

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | How do we prepare for UPSC? | How do I prepare for civil service? | 0.500 |
| 2 | How is the new Harry Potter book 'Harry Potter and the Cursed Child'? | How bad is the new book by J.K Rowling? | 0.578 |
| 3 | What causes a nightmare? | What causes nightmares that seem real? | 0.422 |

**تحلیل:** نیاز به دانش جهانی است (UPSC = civil service exam in India، J.K.
Rowling = Harry Potter author). هیچ tokenی مشترک نیست. SimHash بدون
embedding معنایی نمی‌تواند این رابطه را کشف کند.

### overlap

تعداد جفت‌هایی که هر دو روش هم‌زمان اشتباه پیش‌بینی کردند:
**977 از 5000** (تقریباً 20%). این نشان می‌دهد دو روش خطاهای **مستقل
نسبتاً مختلفی** دارند — ترکیب آن‌ها (Ensemble: مثلاً OR کردن predictionها در
آستانه‌های مناسب) می‌تواند recall ترکیبی را بهبود دهد.

### جمع‌بندی تحلیل خطا

1. روی متن کوتاه با paraphrase vocabulary-level، هیچ‌کدام از دو روش بدون
   embedding معنایی به F1 > 0.7 نمی‌رسد.
2. MinHash error pattern → کمبود overlap shingle (FN). SimHash error pattern
   → similarity ساختاری بدون semantic equivalence (FP).
3. ترکیب دو روش با weighting (مثلاً $\alpha \cdot J_\text{minhash} + (1-\alpha) \cdot
   \text{sim}_\text{simhash}$) می‌تواند هر دو نوع خطا را کاهش دهد.

---

## ۹. ساختار repository

```
src/plagiarism_engine/
├── __init__.py
├── preprocessing.py     # نرمال‌سازی، tokenization، shingles
├── minhash.py           # MinHasher با universal hashing
├── lsh.py               # LSHIndex با banding و bucket dictهای دینامیک
├── simhash.py           # SimHasher با TF-IDF، vectorized
├── dataset.py           # corpus loader و pairs csv loader
├── evaluation.py        # P/R/F1، timing، sweep
└── cli.py               # argparse با ۳ subcommand
tests/
└── test_engine.py       # ۱۴ unit test، همه pass
scripts/
└── extract_errors.py    # ابزار جانبی برای تحلیل خطا
outputs/
├── metrics.csv          # نتایج کامل
├── candidates.csv       # خروجی corpus mode
├── two_file_compare.json
└── error_examples.json  # FP/FN واقعی Quora
```

```bash
pytest -q
# 14 passed in 0.28s
```

---

## ۱۰. کارهای آتی پیشنهادی

1. اضافه کردن sentence-level embedding (مثل sentence-BERT) به‌عنوان مسیر سوم
   و ترکیب با MinHash/SimHash برای پوشش هر دو نوع خطا.
2. پیاده‌سازی character n-gram به‌جای word shingle برای زبان‌هایی با
   tokenization پیچیده (فارسی، چینی) و کلمات با املای متفاوت.
3. caching signatureها روی disk با joblib برای جلوگیری از recomputation.
4. ارزیابی روی PAN-PC-11 و گزارش precision/recall در سطح passage-level (نه
   pair-level).
5. آستانه‌ی adaptive بر اساس طول سند: متن‌های کوتاه‌تر آستانه‌ی بالاتر
   نیاز دارند چون SimHash روی متن کوتاه ناپایدارتر است.
