import os
import time
import pandas as pd
from datetime import datetime
from gnews import GNews
from typing import Dict, List, Tuple

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

TICKER_SEARCH_MAP = {
    "CBA.AX": "Commonwealth Bank Australia",
    "BHP.AX": "BHP Group",
    "NAB.AX": "National Australia Bank",
    "CSL.AX": "CSL Limited",
    "WBC.AX": "Westpac Banking Corporation",
}

START_DATE_STR = "2022-05-04"


def generate_date_chunks(start_str: str, end_str: str = None) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:

    if end_str is None:
        end_str = datetime.now().strftime("%Y-%m-%d")

    # Generate 30-day interval checkpoints
    date_points = pd.date_range(start=start_str, end=end_str, freq="30D")
    
    # Include today as final boundary if not aligned
    if date_points[-1] < pd.to_datetime(end_str):
        date_points = date_points.append(pd.DatetimeIndex([pd.to_datetime(end_str)]))

    chunks = []
    for i in range(len(date_points) - 1):
        d_start = date_points[i]
        d_end = date_points[i + 1]
        
        start_tuple = (d_start.year, d_start.month, d_start.day)
        end_tuple = (d_end.year, d_end.month, d_end.day)
        chunks.append((start_tuple, end_tuple))

    return chunks


def extract_historical_gnews(
    ticker_map: Dict[str, str],
    start_date: str = "2022-05-04"
) -> Dict[str, pd.DataFrame]:

    google_news = GNews(
        language="en",
        country="AU",
        max_results=100
    )

    date_chunks = generate_date_chunks(start_str=start_date)
    print(f"=== Extracted {len(date_chunks)} time windows from {start_date} to Today ===")

    news_dict = {}
    master_news_list = []

    for ticker, query in ticker_map.items():
        print(f"\n==========================================")
        print(f"Fetching Historical News for: {ticker} ({query})")
        print(f"==========================================")
        
        ticker_articles = []

        for start_t, end_t in date_chunks:
            # Set date range in GNews client
            google_news.start_date = start_t
            google_news.end_date = end_t

            date_label = f"{start_t[0]}-{start_t[1]:02d}-{start_t[2]:02d} to {end_t[0]}-{end_t[1]:02d}-{end_t[2]:02d}"

            try:
                raw_articles = google_news.get_news(query)
                if raw_articles:
                    print(f"  [{date_label}] Found {len(raw_articles)} items.")
                    
                    for item in raw_articles:
                        publisher = item.get("publisher", {})
                        publisher_name = publisher.get("title", "") if isinstance(publisher, dict) else str(publisher)

                        pub_date_str = item.get("published date", "")
                        try:
                            pub_date = pd.to_datetime(pub_date_str).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pub_date = f"{start_t[0]}-{start_t[1]:02d}-{start_t[2]:02d} 00:00:00"

                        article_data = {
                            "Ticker": ticker,
                            "Date": pub_date,
                            "Title": item.get("title", ""),
                            "Publisher": publisher_name,
                            "Summary": item.get("description", ""),
                            "Link": item.get("url", "")
                        }
                        ticker_articles.append(article_data)
                        master_news_list.append(article_data)
                
                # Small pause to avoid HTTP 429 rate limits
                time.sleep(0.3)

            except Exception as e:
                print(f"  [{date_label}] Error: {e}")

        if ticker_articles:
            df_news = pd.DataFrame(ticker_articles)
            
            # Deduplicate and sort
            df_news = df_news.drop_duplicates(subset=["Title"]).reset_index(drop=True)
            df_news["Date"] = pd.to_datetime(df_news["Date"])
            df_news = df_news.sort_values(by="Date", ascending=False).reset_index(drop=True)

            safe_ticker = ticker.replace(".", "_").replace("^", "")
            file_path = os.path.join(DATA_DIR, f"news_{safe_ticker}.csv")
            df_news.to_csv(file_path, index=False)

            news_dict[ticker] = df_news
            print(f"--> Finished {ticker}: Total {len(df_news)} unique articles saved to '{file_path}'")

    # Save master file
    if master_news_list:
        df_master = pd.DataFrame(master_news_list)
        df_master = df_master.drop_duplicates(subset=["Ticker", "Title"]).reset_index(drop=True)
        df_master["Date"] = pd.to_datetime(df_master["Date"])
        df_master = df_master.sort_values(by=["Ticker", "Date"], ascending=[True, False]).reset_index(drop=True)

        master_file_path = os.path.join(DATA_DIR, "news_all_tickers.csv")
        df_master.to_csv(master_file_path, index=False)
        print(f"\n=== SUCCESS: Saved Master Historical News Dataset ({len(df_master)} total articles) to '{master_file_path}' ===")

    return news_dict


if __name__ == "__main__":
    extract_historical_gnews(
        ticker_map=TICKER_SEARCH_MAP,
        start_date=START_DATE_STR
    )