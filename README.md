---
title: Digikala Sentiment RAG
emoji: 📊
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
license: mit
---

# 🚀 سامانه تحلیل هوشمند نظرات دیجی‌کالا (RAG-based)

این پروژه یک سیستم پیشرفته برای تحلیل احساسات (Sentiment Analysis) نظرات کاربران دیجی‌کالا است که با استفاده از معماری **RAG (Retrieval-Augmented Generation)** پیاده‌سازی شده است.

## ✨ ویژگی‌های کلیدی
- **تحلیل دقیق احساسات:** استفاده از مدل mBERT برای تشخیص شدت احساس (۱ تا ۵ ستاره).
- **استدلال هوش مصنوعی (RAG):** توضیح علت تحلیل بر اساس نظرات مشابه در دیتابیس.
- **جستجوی معنایی:** بهره‌گیری از **FAISS** و **Sentence-Transformers** برای یافتن سریع نظرات مشابه.
- **داشبورد مدیریتی:** نمایش آمار و توزیع داده‌ها با استفاده از Plotly و Streamlit.
- **استقرار خودکار (CI/CD):** متصل به GitHub Actions برای تست و دیپلوی خودکار روی Hugging Face.

## 🛠 تکنولوژی‌های استفاده شده
- **Language:** Python 3.12+
- **Models:**
  - `nlptown/bert-base-multilingual-uncased-sentiment` (Sentiment)
  - `HooshvareLab/gpt2-fa-comment` (Reasoning)
  - `paraphrase-multilingual-MiniLM-L12-v2` (Embedding)
- **Vector DB:** FAISS
- **UI:** Streamlit

## 📦 نصب و راه‌اندازی محلی

۱. ابتدا مخزن را کلون کنید:
```bash
git clone https://github.com/tahatehran/nlp-sigment-aniliys.git
cd nlp-sigment-aniliys
```

۲. کتابخانه‌های مورد نیاز را نصب کنید:
```bash
pip install -r requirements.txt
```

۳. داده‌ها را آماده کنید:
```bash
python prepare_data.py
```

۴. برنامه را اجرا کنید:
```bash
streamlit run app.py
```

## 🌐 میزبانی در Hugging Face
این پروژه در **Hugging Face Spaces** میزبانی می‌شود. هر تغییر در شاخه `main` این مخزن، به صورت خودکار توسط GitHub Actions تست شده و در صورت موفقیت، روی Hugging Face آپدیت می‌شود.

---
### 🎓 پروژه درس پردازش زبان طبیعی
**مقطع:** کارشناسی ارشد هوش مصنوعی
**توسعه‌دهنده:** Taha Tehrani Nasab
