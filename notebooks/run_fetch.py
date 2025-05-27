import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import fetch_news, save_news_to_csv
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("NEWS_API_KEY")

if not api_key:
    raise ValueError("Missing NEWS_API_KEY in .env file")

query = "AAPL"
df = fetch_news(api_key=api_key, query=query)
save_news_to_csv(df, query)
