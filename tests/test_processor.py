import unittest
import os
import pandas as pd
from processor import SentimentRAG

class TestSentimentRAG(unittest.TestCase):
    dummy_created = False
    data_path = "data/test_samples.csv"

    @classmethod
    def setUpClass(cls):
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(cls.data_path):
            df = pd.DataFrame({
                'text': [
                    "این یک گوشی عالی است و من از خرید آن بسیار راضی هستم.",
                    "اصلا کیفیت خوبی ندارد و زود خراب شد.",
                    "معمولی است، نه خیلی خوب و نه خیلی بد.",
                    "نسبت به قیمتش ارزش خرید دارد.",
                    "ارسال خیلی دیر انجام شد و جعبه پاره بود."
                ]
            })
            df.to_csv(cls.data_path, index=False)
            cls.dummy_created = True

        cls.rag = SentimentRAG(cls.data_path)

    @classmethod
    def tearDownClass(cls):
        if cls.dummy_created and os.path.exists(cls.data_path):
            os.remove(cls.data_path)

    def test_sentiment_scoring(self):
        score, confidence = self.rag.get_sentiment("محصول فوق العاده ای بود")
        self.assertGreaterEqual(score, 1)
        self.assertLessEqual(score, 5)
        self.assertGreater(confidence, 0)

    def test_retrieve_similar(self):
        similar = self.rag.retrieve_similar("گوشی خوب", k=2)
        self.assertEqual(len(similar), 2)
        self.assertIsInstance(similar[0], str)

    def test_generate_explanation(self):
        explanation = self.rag.generate_explanation("کیفیت بد", 1)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 5)

if __name__ == "__main__":
    unittest.main()
