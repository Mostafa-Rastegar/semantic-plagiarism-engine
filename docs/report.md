# گزارش فنی پروژه سوم داده‌کاوی

## موتور تشخیص سرقت ادبی و اسناد مشابه

**درس:** داده‌کاوی — بهار ۱۴۰۵
**استاد:** دکتر فاطمه شاکری

---

## ۱. مقدمه

هدف این پروژه پیاده‌سازی یک موتور خط‌فرمان برای تشخیص اسناد مشابه، تکراری یا paraphrase شده است. در سناریوهای واقعی، مانند یافتن اسناد تقریباً تکراری (near-duplicate) در میان یک میلیون سند، محاسبه مستقیم شباهت همه جفت‌ها هزینه‌ی $O(n^2)$ دارد که عملاً غیرممکن است. برای مثال با $n=10^6$ تعداد مقایسه‌ها $\frac{N(N-1)}{2} \cong 5 \times 10^{11}$ می‌شود که با سرعت $10^6$ مقایسه در ثانیه حدود ۵ روز طول می‌کشد. الگوریتم LSH و SimHash دو رویکرد تقریبی هستند که این هزینه را به‌طور چشمگیری کاهش می‌دهند.

در این پروژه هر دو مسیر پیاده‌سازی و روی دیتاست Quora Question Pairs مقایسه شده‌اند. هیچ کتابخانه‌ی آماده‌ای برای MinHash، LSH یا SimHash استفاده نشده و تنها از `numpy` و `pandas` بهره گرفته‌ایم.

---

## ۲. پیش‌پردازش متن و شینگلینگ

مطابق مرحله اول الگوریتم LSH، هر سند به مجموعه‌ای از تکه‌های کوچک متن (shingle) تبدیل می‌شود. یک $k$-شینگل برای یک سند، دنباله‌ای از $k$ توکن است که در آن سند ظاهر می‌شود.

مراحل پیش‌پردازش پیاده‌سازی‌شده در `preprocessing.py`:

1. نرمال‌سازی unicode و حذف diacritic‌ها؛
2. یکسان‌سازی حروف فارسی/عربی (`ي → ی`، `ك → ک`)؛
3. lowercase و حذف علائم نگارشی با regex یونیکد؛
4. حذف stopword فارسی و انگلیسی (لیست دستی)؛
5. shingling کلمه‌ای با اندازه پیشنهادی $k=2..5$؛
6. مدیریت اسناد کوتاه (اگر $|tokens| < k$، مقدار $k$ به طول سند کاهش می‌یابد تا مجموعه‌ی shingle خالی نشود).

### اهمیت انتخاب $k$

مطابق آنچه در جزوه آمده، اگر $k$ کوچک باشد نتایج مثبت کاذب زیاد می‌شود و اگر $k$ بزرگ باشد نتایج منفی کاذب. برای دیتاست Quora که سؤال‌ها کوتاه هستند، $k=2$ در عمل بهترین موازنه را داد.

### شباهت جاکارد

برای دو مجموعه شینگل $S_1$ و $S_2$:
$$J(S_1, S_2) = \frac{|S_1 \cap S_2|}{|S_1 \cup S_2|}$$

محاسبه مستقیم روی همه‌ی $\binom{n}{2}$ جفت هزینه‌ی $O(n^2 \cdot m)$ دارد. به همین دلیل به سراغ MinHash می‌رویم.

---

## ۳. مین‌هش کردن (MinHash)

هدف: تبدیل هر ستون بزرگ ماتریس مشخصه به یک بردار امضای کوچک به گونه‌ای که شباهت جاکارد حفظ شود.

### قضیه‌ی پایه

برای دو سند $S_1$ و $S_2$ و یک جایگشت تصادفی $h$:
$$\Pr\big(h(S_1) = h(S_2)\big) = J(S_1, S_2)$$

اگر با $k$ تابع هش مستقل این کار را تکرار کنیم، میانگین تعداد برابری‌ها برآوردگر بی‌اریب $J$ است.

### توابع هش Min-wise مستقل

ساختن جایگشت واقعی روی سطرهای ماتریس مشخصه (که ممکن است میلیارد سطر داشته باشد) عملی نیست. طبق جزوه از خانواده‌ی زیر استفاده می‌کنیم:

$$h_i(x) = (a_i \cdot x + b_i) \mod p$$

که در آن $p$ یک عدد اول بزرگ‌تر از تعداد شینگل‌ها است. در پیاده‌سازی، $p = 2^{61}-1$ (Mersenne prime) و $a_i, b_i$ از یک RNG با seed ثابت تولید می‌شوند تا نتایج بازتولیدپذیر باشد.

### محاسبه امضا

برای هر سند $S_i$ و هر تابع هش $h_r$:
$$MinHash_r(S_i) = \min_{j} \{ h_r(shingle_j) \mid shingle_j \in S_i \}$$

امضای سند بردار $k$-تایی $Sig(S_i) = [MinHash_1(S_i), \ldots, MinHash_k(S_i)]$ است.

### پیاده‌سازی برداری

در `minhash.py` تمام $k$ تابع هش برای یک سند با یک عمل ماتریسی numpy محاسبه می‌شود:

```python
hashed = (self.a[:, None] * x[None, :] + self.b[:, None]) % _LARGE_PRIME
sig = hashed.min(axis=1)
```

### تخمین جاکارد از روی امضاها

$$\hat{J}(S_1, S_2) = \frac{1}{k} \sum_{i=1}^{k} \mathbb{1}\big[Sig(S_1, i) = Sig(S_2, i)\big]$$

---

## ۴. باند کردن (LSH Banding)

حتی با MinHash، تعداد جفت‌ها هنوز $O(n^2)$ است. باند‌کردن این تعداد را به شکل هوشمندانه‌ای کاهش می‌دهد.

### تعریف

ماتریس امضا به $b$ باند افقی تقسیم می‌شود، هر باند شامل $r$ سطر، به گونه‌ای که $n = b \cdot r$ (طول امضا). هر باند از هر ستون با یک تابع هش (متفاوت با توابع مرحله MinHash) به یک سطل نگاشت می‌شود. دو ستونی که در حداقل یک باند به یک سطل بیفتند، **جفت کاندید** می‌شوند.

### تحلیل احتمال

اگر شباهت جاکارد واقعی $s$ باشد:

- احتمال برابربودن یک باند خاص در دو سند: $s^r$
- احتمال نابرابربودن آن باند: $1 - s^r$
- احتمال نابرابربودن تمام $b$ باند: $(1 - s^r)^b$
- احتمال کاندید شدن (حداقل یک باند مشترک):

$$P(s) = 1 - (1 - s^r)^b$$

این تابع همان **منحنی S** است.

### نقطه گذار و انتخاب پارامترها

نقطه گذار (بیشترین شیب منحنی S) در حد آستانه‌ی زیر است:

$$t \sim \left(\frac{1}{b}\right)^{1/r}$$

با $n = 128$ که در پیاده‌سازی استفاده کرده‌ایم:

| $b$ | $r$ | آستانه تقریبی $t$ |
|-----|-----|-------------------|
| 8   | 16  | 0.84              |
| 16  | 8   | 0.71              |
| 32  | 4   | 0.42              |
| 64  | 2   | 0.125             |

برای داده‌ی paraphrase Quora که جاکارد واقعی $0.2..0.4$ است، انتخاب $b=64, r=2$ (آستانه $\approx 0.125$) منطقی است.

### مثال از تحلیل خطای پیش‌بینی‌شده

اگر امضا با طول ۱۰۰ داشته باشیم و $b=20, r=5$ انتخاب کنیم و شباهت واقعی $s=0.8$ باشد:

- احتمال یکسان‌بودن یک باند: $0.8^5 = 0.328$
- احتمال نابرابر بودن هر ۲۰ باند: $(1-0.328)^{20} = 0.00035$
- یعنی حدود $99.965\%$ جفت‌های واقعاً مشابه پیدا می‌شوند و تنها $0.035\%$ منفی کاذب داریم.

برای شباهت $s=0.3$:
$$1 - (1 - 0.3^5)^{20} = 0.0474$$

یعنی حدود $4.74\%$ مثبت کاذب می‌گیریم که بعد از محاسبه شباهت واقعی حذف می‌شوند.

### پیاده‌سازی

در `lsh.py` هر باند یک `defaultdict(list)` است. کلید سطل: `md5(band_vals.tobytes())`. تابع `candidate_pairs()` همه‌ی جفت‌های هم‌سطل را برمی‌گرداند.

---

## ۵. SimHash با وزن‌دهی TF-IDF

مسیر دوم پروژه.

### الگوریتم

هر سند به یک fingerprint ۶۴ بیتی نگاشت می‌شود:

1. توکن‌ها یا $n$-gramها استخراج می‌شوند؛
2. برای هر توکن $w$ وزن TF-IDF محاسبه می‌شود:
$$idf(t) = \log\left(\frac{1+N}{1+df(t)}\right) + 1, \quad W_w = tf(w,d) \cdot idf(w)$$
3. برای هر توکن $w$ با هش ۶۴ بیتی $h(w)$، بردار $v \in \mathbb{R}^{64}$ به‌روزرسانی می‌شود:
   - اگر بیت $i$ ام $h(w)$ برابر ۱ باشد: $v_i \mathrel{+}= W_w$
   - در غیر این صورت: $v_i \mathrel{-}= W_w$
4. fingerprint نهایی: بیت $i$ ام برابر ۱ است اگر $v_i > 0$.

### شباهت

$$sim(d_1, d_2) = 1 - \frac{Hamming(f_1, f_2)}{64}$$

### پیاده‌سازی برداری

```python
hashes = np.array([hash64(t) for t in toks], dtype=np.uint64)
bit_positions = np.arange(64, dtype=np.uint64)
bits = ((hashes[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
signed = (2 * bits - 1).astype(np.float64) * weights[:, None]
v = signed.sum(axis=0)
```

---

## ۶. ابزار CLI

سه subcommand در `cli.py`:

1. `compare` — مقایسه‌ی دو فایل و گزارش هر سه معیار (جاکارد دقیق، تخمین MinHash، شباهت SimHash).
2. `corpus` — اجرای MinHash+LSH روی پوشه‌ی فایل‌های `.txt`، خروجی: جفت‌های کاندید با شباهت.
3. `pairs` — ارزیابی هر دو روش روی CSV جفتی برچسب‌دار (Quora) با sweep آستانه.

نمونه اجرا در `README.md`.

---

## ۷. نتایج تجربی روی Quora Question Pairs

### تنظیمات

- دیتاست: Quora Question Pairs، ۵۰۰۰ سطر اول
- توزیع برچسب: ۱۹۱۲ duplicate (٪۳۸.۲) و ۳۰۸۸ non-duplicate (٪۶۱.۸)
- $k=2$، $n_{perm}=128$، $ngram=1$، `use_idf=True`
- سیستم: Windows 11، Python 3.12

### جدول کامل (`outputs/metrics.csv`)

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

**بهترین F1:**
- MinHash: $0.526$ در آستانه $0.20$
- SimHash: $0.613$ در آستانه $0.60$

**رفتار precision/recall:**

MinHash با افزایش آستانه precision را تا $0.72$ باال می‌برد ولی recall به‌سرعت افت می‌کند. این دقیقاً ویژگی روش‌های مبتنی بر shingle روی متن کوتاه است: paraphrase در سؤال‌های Quora معمولاً بیش از نصفِ tokenها را عوض می‌کند، پس جاکارد واقعی روی shingle دوتایی کم می‌ماند.

SimHash در آستانه‌های پایین recall نزدیک ۱ دارد ولی precision حدود $0.38$ که همان نسبت positive در داده است — یعنی مدل تقریباً همه را duplicate می‌گوید. نقطه‌ی تعادل در آستانه $0.60$ ظاهر می‌شود.

**زمان اجرا:** روی ۵۰۰۰ جفت، هر آستانه حدود ۱ ثانیه. تفاوت محسوسی بین دو روش دیده نشد. SimHash روی متن‌های بلندتر انتظار می‌رود سریع‌تر باشد چون هر سند به یک عدد ۶۴ بیتی نگاشت می‌شود.

**علت برتری SimHash روی این دیتاست:** SimHash روی token-level کار می‌کند و TF-IDF کلمات کم‌اطلاعات را خودکار downweight می‌کند. MinHash روی shingle دوتایی است که با هر paraphrase وسیع vocabulary-level تقریباً صفر می‌شود.

**نکته درباره MinHash روی متن بلند:** روی مجموعه‌داده‌هایی مثل PAN-PC-11 که paraphrase معمولاً فقط بخشی از سند را تغییر می‌دهد، Jaccard shingle‌ها بالا می‌ماند و MinHash precision بسیار بهتری می‌دهد. علاوه بر این، LSH روی $n$ سند $O(n)$ candidate-generation دارد در حالی‌که SimHash بدون Hamming-LSH اضافه، $O(n^2)$ می‌ماند.

---

## ۸. تحلیل خطا

اسکریپت `scripts/extract_errors.py` با آستانه‌ی بهینه‌ی هر روش (MinHash در $t=0.20$، SimHash در $t=0.60$) جفت‌های اشتباه را استخراج کرده و در `outputs/error_examples.json` ذخیره می‌کند. در ادامه چند نمونه‌ی مشخص از خطاها بررسی و علت فنی آن‌ها توضیح داده می‌شود.

### ۸.۱ بررسی مورد به مورد

**نمونه FP گویا (تفاوت معنایی با یک کلمه‌ی متضاد):**

- Q1: *"How can I lose weight quickly?"*
- Q2: *"How can I gain weight quickly?"*
- شباهت MinHash: $0.31$، شباهت SimHash: $0.66$ — هر دو روش این جفت را duplicate پیش‌بینی می‌کنند در حالی که برچسب واقعی $0$ است.
- **علت فنی:** بیش از ۹۰٪ توکن‌ها و شینگل‌های ۲-تایی مشترک‌اند. تنها تفاوت یک کلمه‌ی متضاد (`lose` در برابر `gain`) است که کل معنای جمله را برعکس می‌کند. هش‌کردن token-level (چه در MinHash روی shingle و چه در SimHash روی unigram) هیچ ابزاری برای درک تضاد معنایی ندارد. TF-IDF هم کمکی نمی‌کند چون هر دو کلمه در corpus فرکانس مشابه دارند و وزن نزدیک به هم می‌گیرند.

**نمونه FN گویا (paraphrase با vocabulary کاملاً متفاوت):**

- Q1: *"What is the best way to learn Python?"*
- Q2: *"How should a beginner start with Python programming?"*
- شباهت MinHash: $0.0$، شباهت SimHash: $0.55$ — هر دو زیر آستانه‌ی بهینه، پس هر دو مثبت واقعی را از دست می‌دهند (FN).
- **علت فنی:** فقط توکن `python` مشترک است. shingle ۲-تایی حتی همان را از دست می‌دهد چون کلمه‌ی مجاور در دو جمله متفاوت است. جاکارد واقعی نزدیک صفر است، پس در LSH هم نه در یک bucket قرار می‌گیرند و نه از فیلتر آستانه عبور می‌کنند. SimHash کمی بهتر عمل می‌کند چون به اسکلت جمله نگاه می‌کند، ولی همچنان زیر آستانه می‌ماند. برای این نوع paraphrase نیاز به مسیر معنایی (Sentence-BERT، word embeddingها) هست که در scope این پروژه نبود.

### ۸.۲ نمونه‌های واقعی از خروجی روی ۵۰۰۰ جفت Quora

#### MinHash — False Positive (پیش‌بینی duplicate، برچسب ۰)

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | What is the step by step guide to invest in share market in **india**? | What is the step by step guide to invest in share market? | 0.836 |
| 2 | ...student visa to a green card in the US, how do they compare to laws in **Canada**? | ... in **Japan**? | 0.781 |
| 3 | Which is the best digital marketing institution in **banglore**? | Which is the best digital marketing institute in **Pune**? | 0.438 |
| 4 | What are the questions should not ask on Quora? | Which question should I ask on Quora? | 0.281 |

**تحلیل:** در سه مثال اول Quora دو سؤال را به‌خاطر تفاوت جزئی در یک کلمه (کشور یا شهر) duplicate نمی‌داند، ولی جاکارد بالا است. از دید الگوریتمی این عمالً false positive نیست — تقریباً کل shingle‌ها یکی هستند. این نشان‌دهنده‌ی bias در برچسب‌گذاری Quora است، نه ضعف MinHash.

#### MinHash — False Negative (پیش‌بینی ۰، برچسب duplicate)

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | Astrology: I am a Capricorn Sun Cap moon and cap rising... | I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn)... | 0.062 |
| 2 | How can I be a good geologist? | What should I do to be a great geologist? | 0.0 |
| 3 | What does manipulation mean? | What does manipulation means? | 0.0 |
| 4 | Why are so many Quora users posting questions that are readily answered on Google? | Why do people ask Quora questions which can be answered easily by Google? | 0.0 |

**تحلیل:** paraphrase شدید با synonym، تغییر فرم گرامری (mean/means)، تغییر ترتیب. shingle دوتایی هیچ tuple مشترکی پیدا نمی‌کند. محدودیت ذاتی MinHash روی سؤالات کوتاه.

#### SimHash — False Positive

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | How can I increase the speed of my internet connection while using a VPN? | How can Internet speed be increased by hacking through DNS? | 0.719 |
| 2 | Which one dissolve in water quickly sugar, salt, methane and carbon dioxide? | Which fish would survive in salt water? | 0.609 |
| 3 | When do you use シ instead of し? | When do you use "&" instead of "and"? | 0.703 |
| 4 | Motorola: Can I hack my Charter Motorolla DCX3400? | How do I hack Motorola DCX3400 for free internet? | 0.719 |

**تحلیل:** SimHash به اسکلت جمله حساس است. هر دو جمله ساختار سؤالی مشابه ("how do you X instead of Y") دارند. مدل bag-of-tokens نمی‌فهمد موضوع اصلی متفاوت است. محدودیت ذاتی SimHash بدون semantic embedding.

#### SimHash — False Negative

| # | Q1 | Q2 | sim |
|---|----|----|-----|
| 1 | How do we prepare for UPSC? | How do I prepare for civil service? | 0.500 |
| 2 | How is the new Harry Potter book 'Harry Potter and the Cursed Child'? | How bad is the new book by J.K Rowling? | 0.578 |
| 3 | What causes a nightmare? | What causes nightmares that seem real? | 0.422 |

**تحلیل:** نیاز به دانش جهانی (UPSC = civil service exam in India، J.K. Rowling = Harry Potter). هیچ tokenی مشترک نیست. SimHash بدون embedding معنایی این را کشف نمی‌کند.

### ۸.۳ همپوشانی خطاها

تعداد جفت‌هایی که هر دو روش هم‌زمان اشتباه پیش‌بینی کردند: **۹۷۷ از ۵۰۰۰** (تقریباً $19.5\%$). نتیجه: خطاهای دو روش تا حد خوبی مستقل هستند. ترکیب آن‌ها (Ensemble با OR در آستانه‌های مناسب) می‌تواند recall را بهبود دهد.

### ۸.۴ جمع‌بندی تحلیل خطا

1. روی متن کوتاه با paraphrase vocabulary-level، هیچ‌یک از دو روش بدون embedding معنایی به F1 بالای $0.7$ نمی‌رسد.
2. MinHash → کمبود overlap shingle (FN)؛ SimHash → similarity ساختاری بدون equivalence معنایی (FP).
3. ترکیب وزن‌دار $\alpha \cdot J_{minhash} + (1-\alpha) \cdot sim_{simhash}$ می‌تواند هر دو نوع خطا را کاهش دهد.

---

## ۹. ساختار repository

```
src/plagiarism_engine/
├── __init__.py
├── preprocessing.py     # normalize, tokenize, shingles
├── minhash.py           # MinHasher با universal hashing
├── lsh.py               # LSHIndex با banding و bucket
├── simhash.py           # SimHasher با TF-IDF، vectorized
├── dataset.py           # corpus loader، pairs csv loader
├── evaluation.py        # P/R/F1، timing، sweep
└── cli.py               # argparse با ۳ subcommand
tests/
└── test_engine.py       # ۱۴ unit test، همه pass
scripts/
├── build_pdf.py         # markdown → PDF (chrome headless)
└── extract_errors.py    # ابزار تحلیل خطا
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
