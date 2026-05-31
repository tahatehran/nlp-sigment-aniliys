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

# Custom CSS for RTL and styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ef4056;
        color: white;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    .positive { background-color: #e8f5e9; border-color: #4caf50; }
    .negative { background-color: #ffebee; border-color: #f44336; }
    .neutral { background-color: #fff3e0; border-color: #ff9800; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_rag():
    return SentimentRAG()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/29/Digikala_logo.svg", width=200)
    st.title("داشبورد مدیریتی")
    st.info("این پروژه با استفاده از معماری RAG و مدل‌های ترنسفورمر برای تحلیل احساسات نظرات دیجی‌کالا طراحی شده است.")

    if st.button("پاکسازی حافظه"):
        st.cache_resource.clear()
        st.rerun()

# Main UI
st.title("🚀 سیستم تحلیل هوشمند بخش‌های متنی (NLP)")
st.subheader("تحلیل دقیق احساسات به همراه استدلال هوش مصنوعی")

tab1, tab2 = st.tabs(["🔍 تحلیل نظر", "📊 آمار و داشبورد"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("متن نظر خود را وارد کنید:", placeholder="مثلاً: کیفیت ساخت این گوشی عالیه ولی باتریش زود خالی میشه...", height=150)
        analyze_btn = st.button("تحلیل و بررسی")

    if analyze_btn and user_input:
        with st.spinner("در حال پردازش با هوش مصنوعی..."):
            rag = load_rag()
            score, confidence = rag.get_sentiment(user_input)
            explanation = rag.generate_explanation(user_input, score)
            similar = rag.retrieve_similar(user_input, k=3)

            # Determine color and label
            if score > 3:
                css_class = "positive"
                label = "مثبت"
                icon = "😊"
            elif score < 3:
                css_class = "negative"
                label = "منفی"
                icon = "😞"
            else:
                css_class = "neutral"
                label = "خنثی"
                icon = "😐"

            st.markdown(f"""
                <div class="sentiment-box {css_class}">
                    <h3>نتیجه تحلیل: {label} {icon}</h3>
                    <p><b>امتیاز شدت (۱ تا ۵):</b> {score}</p>
                    <p><b>دقت مدل:</b> {confidence:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

            st.success("### 🤖 دلیل و استدلال هوش مصنوعی:")
            st.write(explanation)

            with st.expander("📚 نظرات مشابه یافت شده در پایگاه داده (RAG):"):
                for s in similar:
                    st.write(f"- {s}")

with tab2:
    st.header("نمای کلی داده‌ها")
    try:
        df = pd.read_csv("data/digikala_samples.csv")

        c1, c2, c3 = st.columns(3)
        c1.metric("تعداد نظرات مرجع", len(df))
        c2.metric("مدل پایه", "mBERT")
        c3.metric("تکنولوژی RAG", "FAISS + GPT2-FA")

        # Simple Length distribution
        df['length'] = df['text'].str.len()
        fig = px.histogram(df, x="length", title="توزیع طول نظرات (تعداد کاراکتر)", color_discrete_sequence=['#ef4056'])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"خطا در بارگذاری داده‌ها: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>پروژه درس NLP - مقطع کارشناسی ارشد هوش مصنوعی</p>", unsafe_allow_html=True)
