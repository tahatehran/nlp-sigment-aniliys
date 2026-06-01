# توضیح دقیق و خط‌به‌خط کدهای پروژه

در این بخش، کدهای اصلی پروژه را با جزئیات آموزشی برای ارائه و یادگیری بررسی می‌کنیم.

---

## ۱. فایل `processor.py` (مغز متفکر سیستم)

این فایل کلاس اصلی `SentimentRAG` را در خود جای داده است.

### ساختار کلاس و لود کردن مدل‌ها:
```python
class SentimentRAG:
    _instance = None # متغیر برای ذخیره تنها نمونه ساخته شده (Singleton)

    def __init__(self, data_path="data/digikala_samples.csv"):
        # تنظیم تعداد هسته‌های پردازشی برای بهینگی در سرورهای ضعیف
        torch.set_num_threads(2)

        # لود کردن پایپ‌لاین تحلیل احساسات BERT
        # این مدل متن را می‌گیرد و امتیازی بین ۱ تا ۵ ستاره می‌دهد
        self.sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/...")

        # لود کردن مدل تبدیل متن به بردار (Embedding)
        # این مدل جملات فارسی را به بردار ۳۸۴ بعدی تبدیل می‌کند
        self.embed_model = SentenceTransformer('sentence-transformers/...')
```

### متد تولید پاسخ (Explanation Generation):
این بخش از معماری RAG استفاده می‌کند:
```python
def generate_explanation(self, text, sentiment_score):
    # ۱. پیدا کردن نظرات مشابه از دیتابیس FAISS
    similar_comments = self.retrieve_similar(text, k=2)

    # ۲. ساختن یک "پرامپت" (Prompt) برای مدل GPT-2
    # ما متن کاربر + نظرات مشابه را ترکیب می‌کنیم تا مدل دلیل بیاورد
    prompt = f"متن: {text}\nاحساس: {sentiment_label}\nشواهد: {context}\nدلیل فنی:"

    # ۳. تولید متن توسط GPT-2 فارسی
    outputs = self.gen_model.generate(**inputs, max_new_tokens=50)
```

---

## ۲. فایل `prepare_data.py` (مهندسی داده)

وظیفه این فایل، گلچین کردن ۲۰۰۰ نظر از میان میلیون‌ها نظر موجود در اینترنت است.

### استریم کردن داده‌ها (Streaming):
```python
def get_samples_from_stream(ds_name, split, token, target_count):
    # streaming=True یعنی به جای دانلود کل فایل چند گیگابایتی،
    # فقط تکه‌های کوچک را می‌خوانیم تا رم پر نشود.
    ds = load_dataset(ds_name, split=split, streaming=True)

    for item in ds:
        # جستجو برای پیدا کردن ستون متن (چون در هر دیتاست نام ستون متفاوت است)
        potential_cols = ['text', 'comment', 'body']
        # ...
```

### ساخت ایندکس FAISS:
```python
def generate_faiss_index(df):
    # تبدیل تمام ۲۰۰۰ نظر به بردار
    embeddings = model.encode(texts)
    # ایجاد یک ایندکس FlatL2 که برای مقایسه فواصل اقلیدسی بردارهاست
    index = faiss.IndexFlatL2(embeddings.shape[1])
    # اضافه کردن بردارها به ایندکس برای جستجوی سریع
    index.add(np.array(embeddings).astype('float32'))
```

---

## ۳. فایل `app.py` (رابط کاربری)

این فایل از کتابخانه Streamlit برای ساخت سایت استفاده می‌کند.

### تنظیمات RTL (راست به چپ):
```python
st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    # ... تنظیمات فونت وزیر برای نمایش زیبای فارسی
    </style>
""", unsafe_allow_html=True)
```

### مدیریت اجرای مدل:
```python
if st.button("تحلیل نظر"):
    with st.spinner("در حال پردازش هوشمند..."):
        # فراخوانی متدهای کلاس SentimentRAG برای خروجی نهایی
        score, conf = rag.get_sentiment(user_input)
        explanation = rag.generate_explanation(user_input, score)
```

---
**نکته آموزشی:** در سیستم ما، مدل GPT-2 فقط بر اساس تخیل خودش حرف نمی‌زند؛ بلکه "شواهد" (Retrieve) شده از دیتابیس را می‌بیند و سپس بر اساس آن دلیل می‌آورد. این کار باعث کاهش خطا (Hallucination) می‌شود.
