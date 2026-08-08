import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DATA_DIR = "data_cache"
MODEL_NAME = "ProsusAI/finbert"


def run_finbert_sentiment():
    master_file = os.path.join(DATA_DIR, "news_all_tickers.csv")
    if not os.path.exists(master_file):
        raise FileNotFoundError(f"Could not find '{master_file}'. Please run news extraction first.")

    print(f"=== Loading News Dataset: '{master_file}' ===")
    df_news = pd.read_csv(master_file)
    
    # Combine Title and Summary for richer context
    df_news["Title"] = df_news["Title"].fillna("")
    df_news["Summary"] = df_news["Summary"].fillna("")
    df_news["FullText"] = (df_news["Title"] + ". " + df_news["Summary"]).str.strip()
    df_news = df_news[df_news["FullText"].str.len() > 5].reset_index(drop=True)

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    print(f"=== Initializing FinBERT model on {'GPU' if device == 0 else 'CPU'} ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    sentiment_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,  # Replaces deprecated return_all_scores=True
        device=device,
        max_length=512,
        truncation=True
    )

    print("\n=== Extracting Sentiment Scores ===")
    batch_size = 32
    texts = df_news["FullText"].tolist()
    
    scores, pos_probs, neg_probs = [], [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT Inferencing"):
        batch_texts = texts[i : i + batch_size]
        results = sentiment_pipe(batch_texts)

        for res in results:
            # Defensive parsing: handle both list of dicts and single dict
            if isinstance(res, list):
                prob_dict = {item['label'].lower(): item['score'] for item in res if isinstance(item, dict)}
            elif isinstance(res, dict):
                prob_dict = {res['label'].lower(): res['score']}
            else:
                prob_dict = {}

            p_pos = prob_dict.get('positive', 0.0)
            p_neg = prob_dict.get('negative', 0.0)
            
            # Composite score: Range [-1.0 to +1.0]
            sentiment_score = p_pos - p_neg

            scores.append(sentiment_score)
            pos_probs.append(p_pos)
            neg_probs.append(p_neg)

    df_news["Sentiment_Score"] = scores
    df_news["Prob_Positive"] = pos_probs
    df_news["Prob_Negative"] = neg_probs
    df_news["Date"] = pd.to_datetime(df_news["Date"]).dt.date

    # Aggregate to Daily Ticker Level
    print("\n=== Aggregating Scores to Daily Per-Ticker Level ===")
    daily_df = df_news.groupby(["Ticker", "Date"]).agg(
        Daily_Sentiment=("Sentiment_Score", "mean"),
        News_Count=("Sentiment_Score", "count")
    ).reset_index()

    daily_df["Date"] = pd.to_datetime(daily_df["Date"])

    # Save daily sentiment per ticker
    for ticker in daily_df["Ticker"].unique():
        safe_ticker = ticker.replace(".", "_").replace("^", "")
        ticker_daily = daily_df[daily_df["Ticker"] == ticker].reset_index(drop=True)
        out_path = os.path.join(DATA_DIR, f"daily_sentiment_{safe_ticker}.csv")
        ticker_daily.to_csv(out_path, index=False)
        print(f"  Saved daily sentiment for {ticker} -> '{out_path}'")

    print("\n=== Sentiment Extraction Finished Successfully ===")


if __name__ == "__main__":
    run_finbert_sentiment()