import requests
import pandas as pd
import os
from datetime import datetime, timedelta


def fetch_news(api_key, query="AAPL", from_date=None, to_date=None, page_size=100, max_pages=5):
    """
    Fetches news articles using NewsAPI.
    """
    if from_date is None:
        from_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = datetime.today().strftime("%Y-%m-%d")

    all_articles = []

    for page in range(1, max_pages + 1):
        url = (
            f"https://newsapi.org/v2/everything?q={query}"
            f"&from={from_date}&to={to_date}"
            f"&language=en&sortBy=publishedAt"
            f"&pageSize={page_size}&page={page}&apiKey={api_key}"
        )

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}: {response.status_code}")
            break

        articles = response.json().get("articles", [])
        if not articles:
            break

        all_articles.extend([{
            "publishedAt": art["publishedAt"],
            "title": art["title"],
            "source": art["source"]["name"],
            "url": art["url"]
        } for art in articles])

    return pd.DataFrame(all_articles)


def save_news_to_csv(df, query):
    today = datetime.today().strftime("%Y-%m-%d")
    path = f"data/raw_news_{query}_{today}.csv"
    df.to_csv(path, index=False)
    print(f"[✔] News saved to: {path}")
