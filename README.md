---
title: NLP Segment Analysis
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
license: mit
---

# NLP Segment Analysis

تحلیل بخش‌های متنی با پردازش زبان طبیعی

## نحوه استفاده

۱. متن خود را وارد کنید
۲. دکمه تحلیل را بزنید
۳. نتیجه را مشاهده کنید

---
### جزئیات پروژه (مقطع ارشد هوش مصنوعی)
این پروژه با استفاده از داده‌های نظرات دیجی‌کالا و مدل‌های زیر پیاده‌سازی شده است:
- **داده‌ها:** fibonacciai/Digikala-Comments، ParsiAI/digikala-sentiment-analysis و EhsanShahbazi/digikala-comments
- **مدل‌ها:**
  - `openai-community/gpt2` (بهینه شده با `HooshvareLab/gpt2-fa-comment`)
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - `nlptown/bert-base-multilingual-uncased-sentiment`

**ویژگی اصلی:** استفاده از معماری RAG برای توضیح علت تحلیل احساس توسط هوش مصنوعی.
