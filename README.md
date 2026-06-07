---
title: NLP Sentiment Analysis Digikala
emoji: 📊
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: 1.58.0
python_version: 3.12
app_file: app.py
pinned: false
---

# سیستم هوشمند تحلیل احساسات نظرات دیجی‌کالا (RAG-based)

این پروژه یک سیستم پیشرفته برای تحلیل نظرات کاربران دیجی‌کالا است که با استفاده از معماری **RAG (Retrieval-Augmented Generation)** طراحی شده است. این سیستم نه تنها احساس نظر (مثبت/منفی) را تشخیص می‌دهد، بلکه با جستجو در میان ۲۰۰۰ نظر واقعی دیگر، دلیلی برای تحلیل خود ارائه می‌کند.

---

## 🚀 ویژگی‌های کلیدی
- **تحلیل احساسات ۵ سطحی**: استفاده از مدل bert-base-multilingual-uncased-sentiment برای تشخیص دقیق میزان رضایت.
- **تولید توضیح هوشمند**: استفاده از GPT-2 فارسی برای ارائه دلیل بر اساس شواهد.
- **جستجوی معنایی سریع**: بهره‌گیری از FAISS برای یافتن نظرات مشابه در کمتر از چند میلی‌ثانیه.
- **دیتابیس غنی**: ترکیب ۴ دیتاست معتبر از نظرات دیجی‌کالا (شامل نظرات ویدئویی و متنی).
- **رابط کاربری زیبا**: طراحی شده با Streamlit و پشتیبانی کامل از زبان فارسی (RTL).

---

## 📚 منابع علمی و ارجاعات (References)

این پروژه با بهره‌گیری از متدولوژی‌های نوین در حوزه NLP و بازیابی اطلاعات طراحی شده است:

1. **RAG-Enhanced Sentiment Analysis System Using Transformer Models and FAISS for Customer Review Analysis**
   - *متمرکز بر بهینه‌سازی تحلیل احساسات با استفاده از بردارهای بازنمایی و سیستم‌های بازیابی.*

2. **Optimization of Customer Feedback Summarization Using Large Language Models (LLM) and Advanced Retrieval-Augmented Generation**
   - *مطالعه تخصصی در زمینه بهبود خلاصه‌سازی و استخراج دلیل در بازخوردهای مشتریان با استفاده از معماری پیشرفته RAG.*

3. **Sentiment Analysis of Customer Reviews in E-commerce: A Retrieval-Augmented Generation Approach**
   - *مقاله مرجع جهت بررسی متدهای پیشرفته در مدل‌سازی نظرات کاربران.*
   - *لینک مستقیم:* [DiVA Portal - Full Text Access](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1935432)

---

## 🛠 نحوه اجرا در محیط محلی
۱. ابتدا کتابخانه‌های مورد نیاز را نصب کنید:
```bash
pip install -r requirements.txt
```

۲. داده‌ها و ایندکس FAISS را آماده کنید:
```bash
python prepare_data.py
```

۳. برنامه را اجرا کنید:
```bash
streamlit run app.py
```

---

## ☁️ میزبانی و استقرار
این پروژه به صورت خودکار در **Hugging Face Spaces** میزبانی می‌شود. برای حفظ سرعت و بهینگی، فایل‌های مستندات و تست‌ها در نسخه سرور آپلود نمی‌شوند.
👉 [لینک مشاهده زنده پروژه](https://huggingface.co/spaces/tahatehrani/nlp-segment-analysis)

---
**توسعه‌دهنده:** Taha Tehrani Nasab
**(پروژه درس پردازش زبان طبیعی - NLP)**
