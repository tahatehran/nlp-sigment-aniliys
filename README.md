---
title: NLP Segment Analysis
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# NLP Segment Analysis 🚀
### تحلیل پیشرفته احساسات نظرات دیجی‌کالا با معماری RAG

این پروژه برای مقطع کارشناسی ارشد هوش مصنوعی طراحی شده و قادر است علاوه بر تشخیص احساس (مثبت/منفی/خنثی)، دلیل این تحلیل را با استفاده از مدل‌های مولد توضیح دهد.

## ویژگی‌ها:
- **تحلیل ۵ مرحله‌ای احساسات** (از ۱ تا ۵ ستاره)
- **معماری RAG** برای بازیابی نظرات مشابه و استدلال دقیق
- **داشبورد مدیریتی** با Streamlit
- **کاملاً فارسی** و بهینه شده برای رم پایین (Under 8GB)

## نحوه استفاده:
۱. متن خود را در کادر مربوطه وارد کنید.
۲. دکمه "تحلیل و بررسی" را بزنید.
۳. نتیجه، دقت مدل و دلیل هوش مصنوعی را مشاهده کنید.

## مدل‌های مورد استفاده:
- `nlptown/bert-base-multilingual-uncased-sentiment`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `HooshvareLab/gpt2-fa-comment`

---
*ساخته شده برای ارائه آکادمیک و استقرار در Hugging Face*
