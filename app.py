import os
import streamlit as st
import pandas as pd
import plotly.express as px
from processor import SentimentRAG
import time

# Page Config
st.set_page_config(
    page_title="تحلیل هوشمند نظرات دیجی‌کالا",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');

    :root {
        --primary-color: #ef4056;
        --bg-light: #f8f9fa;
        --text-dark: #343a40;
    }

    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif;
        direction: rtl;
        text-align: right;
    }

    .main {
        background-color: var(--bg-light);
    }

    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: var(--primary-color);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton>button:hover {
        background-color: #d8364b;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(239, 64, 86, 0.3);
    }

    .sentiment-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: none;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }

    .sentiment-box:hover {
        transform: scale(1.01);
    }

    .positive { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-right: 8px solid #4caf50; color: #2e7d32; }
    .negative { background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border-right: 8px solid #f44336; color: #c62828; }
    .neutral { background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-right: 8px solid #ff9800; color: #ef6c00; }

    .custom-header {
        background-color: white;
        padding: 20px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 30px;
    }

    .stExpander {
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-radius: 10px !important;
        background-color: white;
    }

    div[data-testid="stDataFrame"] {
        direction: rtl;
        text-align: right;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_rag():
    # Data is pre-prepared by GitHub Actions or handled via online fallback in SentimentRAG
    return SentimentRAG()

def get_rag_instance_safe():
    """Returns the RAG instance ONLY if it has already been initialized."""
    if SentimentRAG._instance and hasattr(SentimentRAG._instance, 'initialized') and SentimentRAG._instance.initialized:
        return SentimentRAG._instance
    return None

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/29/Digikala_logo.svg", width=200)
    st.title("کنترل پنل")
    st.markdown("---")
    st.info("این سیستم با ترکیب مدل‌های BERT و معماری RAG، تحلیل دقیقی از نظرات ارائه می‌دهد.")

    with st.expander("تنظیمات پیشرفته"):
        k_val = st.slider("تعداد نظرات مشابه (RAG)", 1, 5, 3)
        clear_cache = st.button("بازنشانی حافظه")
        if clear_cache:
            st.cache_resource.clear()
            st.rerun()

# Header Section
st.markdown("""
    <div class="custom-header">
        <h1 style='color: #ef4056; margin: 0;'>سامانه تحلیل هوشمند نظرات دیجی‌کالا</h1>
        <p style='color: #666; margin-top: 5px;'>استخراج احساسات و تولید استدلال مبتنی بر داده (معماری RAG)</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["تحلیل متن", "داشبورد آماری"])

with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("### ورود اطلاعات")
        user_input = st.text_area(
            "نظر خود را برای تحلیل وارد کنید:",
            placeholder="مثال: این محصول عالی است اما بسته‌بندی آن ضعیف بود...",
            height=200,
            help="متن نظر را اینجا تایپ کنید."
        )
        analyze_btn = st.button("شروع تحلیل")

    with col2:
        if analyze_btn and user_input:
            with st.spinner("در حال پردازش و تحلیل هوشمند..."):
                start_time = time.time()
                # Check if data exists locally to inform user
                if not os.path.exists("data/digikala_samples.csv"):
                    st.warning("⚠️ دیتابیس محلی یافت نشد. در حال فراخوانی آنلاین داده‌ها...")

                rag = load_rag()
                score, confidence = rag.get_sentiment(user_input)
                explanation = rag.generate_explanation(user_input, score)
                similar = rag.retrieve_similar(user_input, k=k_val)
                elapsed = time.time() - start_time

                if score > 3:
                    css_class, label = "positive", "مثبت"
                elif score < 3:
                    css_class, label = "negative", "منفی"
                else:
                    css_class, label = "neutral", "خنثی"

                st.markdown(f"""
                    <div class="sentiment-box {css_class}">
                        <h2 style='margin:0;'>وضعیت: {label}</h2>
                        <hr style='border: 0.5px solid rgba(0,0,0,0.1);'>
                        <p style='font-size: 1.1em;'><b>امتیاز شدت:</b> {score} از ۵</p>
                        <p style='font-size: 1.1em;'><b>ضریب اطمینان:</b> {confidence:.2%}</p>
                        <small>زمان پردازش: {elapsed:.2f} ثانیه</small>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("### دلیل و استدلال:")
                st.info(explanation)
        else:
            st.info("متن نظر را وارد کرده و دکمه تحلیل را بزنید تا نتایج اینجا نمایش داده شوند.")

    if analyze_btn and user_input:
        st.markdown("---")
        with st.expander("شواهد بازیابی شده (RAG Context)", expanded=False):
            for i, s in enumerate(similar, 1):
                st.write(f"**{i}.** {s}")

with tab2:
    st.header("داده‌های مرجع")
    try:
        data_file = "data/digikala_samples.csv"
        df = None
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
        else:
            # ONLY attempt to retrieve from RAG if it is already initialized to avoid OOM/Bottlenecks
            rag_instance = get_rag_instance_safe()
            if rag_instance:
                df = rag_instance.df

        if df is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("تعداد نظرات", len(df))
            m2.metric("مدل", "bert-base")
            m3.metric("معماری", "RAG")
            m4.metric("زبان", "فارسی")

            c1, c2 = st.columns(2)

            with c1:
                df['length'] = df['text'].str.len()
                fig_hist = px.histogram(
                    df, x="length",
                    title="توزیع طول نظرات",
                    color_discrete_sequence=['#ef4056'],
                    labels={'length': 'طول متن'}
                )
                fig_hist.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig_hist, width='stretch')

            with c2:
                st.markdown("### پیش‌نمایش داده‌ها")
                st.dataframe(df.head(15), width='stretch')
        else:
            st.warning("دیتابیس نظرات یافت نشد. لطفاً منتظر بروزرسانی خودکار بمانید یا دکمه تحلیل (در زبانه اول) را بزنید تا داده‌ها بارگذاری شوند.")

    except Exception as e:
        st.error(f"⚠️ خطا در بارگذاری دیتابیس: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        (پروژه NLP - Taha Tehrani Nasab)<br>
        <small>قدرت گرفته از Hugging Face Transformers & FAISS</small>
    </div>
""", unsafe_allow_html=True)
