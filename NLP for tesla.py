import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

# Heavy NLP dependencies are imported lazily inside TeslaAINarrativeFactor.
torch = None
AutoTokenizer = None
AutoModelForSequenceClassification = None
pipeline = None


class TeslaAINarrativeFactor:
    def __init__(self):
        # 1. 定义特斯拉 AI 业务相关的核心关键词字典 (支持中英文)
        self.ai_keywords = [
            r'AI', r'artificial intelligence', r'terafab',
            r'FSD', r'full self-driving', r'chip factory', r'自动驾驶',
            r'robotaxi', r'无人出租车',
            r'optimus', r'tesla bot', r'人形机器人', r'擎天柱',
            r'dojo', r'supercomputer', r'超算',
            r'neural network', r'神经网络', r'machine learning'
        ]
        self.keyword_pattern = re.compile('|'.join(self.ai_keywords), re.IGNORECASE)
        
        # 2. 初始化 FinBERT 金融情感分析模型
        # ProsusAI/finbert 是专门在金融语料上微调的 BERT 模型
        print("Loading FinBERT model...")
        try:
            import torch as _torch  # type: ignore
            from transformers import (  # type: ignore
                AutoTokenizer as _AutoTokenizer,
                AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
                pipeline as _pipeline,
            )
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing dependencies for FinBERT sentiment. Install with: pip install transformers torch"
            ) from e

        self.tokenizer = _AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = _AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.nlp_pipeline = _pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if _torch.cuda.is_available() else -1,
        )
        print("Model loaded successfully.")



def _compile_ai_narrative_regex() -> re.Pattern:
    """Regex for AI-related Tesla narratives (ASCII-only for robustness)."""
    keywords = [
        r"\bAI\b",
        r"artificial\s+intelligence",
        r"machine\s+learning",
        r"neural\s+network",
        r"autonomous\s+driving",
        r"self[-\s]?driving",
        r"FSD",
        r"robotaxi",
        r"autonomy",
        r"Optimus",
        r"Tesla\s+Bot",
        r"humanoid\s+robot",
        r"Dojo",
        r"supercomputer",
        r"inference",
        r"training\s+cluster",
        r"LLM",
        r"foundation\s+model",
    ]
    return re.compile("|".join(keywords), flags=re.IGNORECASE)


def build_tsla_ai_narrative_daily_from_rpa_csv(
    csv_path: str,
    *,
    out_daily_csv: str = "tsla_ai_narrative_daily_from_rpa.csv",
    chunksize: int = 300_000,
    max_chunks: Optional[int] = None,
    progress_every_chunks: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_relevance: Optional[int] = 90,
) -> pd.DataFrame:
    """Stream TSLA_AI.csv and compute a daily AI narrative factor.

    The input file is assumed to be RavenPack-style event rows (very large).
    We avoid loading the full CSV into memory by using pandas chunked reading.

    Factor definition (event-level):
      - ai_hit: keyword match on headline/event_text/category/type/sub_type
      - weight: (relevance/100) * (event_relevance/100)
      - factor_event: weight * keyword_count * event_sentiment_score
      - volume_event: weight * keyword_count

    Daily aggregation:
      - factor_sum: sum(factor_event)
      - volume_sum: sum(volume_event)
      - sentiment_wavg = factor_sum / volume_sum (if volume_sum>0)
    """

    pattern = _compile_ai_narrative_regex()
    usecols = [
        "rpa_date_utc",
        "timestamp_utc",
        "rp_story_id",
        "headline",
        "event_text",
        "category",
        "type",
        "sub_type",
        "relevance",
        "event_relevance",
        "event_sentiment_score",
        "news_type",
        "source_name",
    ]

    dtype = {
        "rpa_date_utc": "string",
        "timestamp_utc": "string",
        "rp_story_id": "string",
        "headline": "string",
        "event_text": "string",
        "category": "string",
        "type": "string",
        "sub_type": "string",
        "news_type": "string",
        "source_name": "string",
        "relevance": "float64",
        "event_relevance": "float64",
        "event_sentiment_score": "float64",
    }

    start_dt = _parse_date_ymd(start_date) if start_date else None
    end_dt = _parse_date_ymd(end_date) if end_date else None

    agg: Dict[str, Dict[str, float]] = {}
    rows_seen = 0
    rows_ai = 0
    chunks_seen = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=True,
    ):
        chunks_seen += 1
        rows_seen += int(len(chunk))

        date_series = pd.to_datetime(chunk["rpa_date_utc"], errors="coerce").dt.date
        chunk = chunk.assign(date=date_series.astype("string"))
        chunk = chunk.dropna(subset=["date"])  # type: ignore[arg-type]

        if start_dt is not None or end_dt is not None:
            dt = pd.to_datetime(chunk["date"], errors="coerce").dt.date
            if start_dt is not None:
                chunk = chunk.loc[dt >= start_dt]
            if end_dt is not None:
                chunk = chunk.loc[dt <= end_dt]

        if min_relevance is not None and "relevance" in chunk.columns:
            chunk = chunk.loc[chunk["relevance"].fillna(0) >= float(min_relevance)]

        if chunk.empty:
            continue

        text = (
            chunk["headline"].fillna("")
            + " "
            + chunk["event_text"].fillna("")
            + " "
            + chunk["category"].fillna("")
            + " "
            + chunk["type"].fillna("")
            + " "
            + chunk["sub_type"].fillna("")
        )

        ai_hit = text.str.contains(pattern, na=False)
        if not bool(ai_hit.any()):
            continue

        ai = chunk.loc[ai_hit].copy()
        text_ai = text.loc[ai_hit]
        rows_ai += int(len(ai))

        keyword_count = text_ai.str.count(pattern).astype("float64").clip(lower=1.0)
        w = (ai["relevance"].fillna(0) / 100.0) * (ai["event_relevance"].fillna(0) / 100.0)
        s = ai["event_sentiment_score"].fillna(0)

        factor_event = (w * keyword_count * s).astype("float64")
        volume_event = (w * keyword_count).astype("float64")

        daily = (
            pd.DataFrame(
                {
                    "date": ai["date"].astype("string"),
                    "row_count": 1.0,
                    "factor_sum": factor_event,
                    "volume_sum": volume_event,
                }
            )
            .groupby("date", as_index=True)
            .sum(numeric_only=True)
        )

        for d, r in daily.iterrows():
            key = str(d)
            if key not in agg:
                agg[key] = {"row_count": 0.0, "factor_sum": 0.0, "volume_sum": 0.0}
            agg[key]["row_count"] += float(r["row_count"])
            agg[key]["factor_sum"] += float(r["factor_sum"])
            agg[key]["volume_sum"] += float(r["volume_sum"])

        if progress_every_chunks > 0 and (chunks_seen % int(progress_every_chunks) == 0):
            print(f"Scanned {rows_seen:,} rows; AI-hit rows {rows_ai:,}; days {len(agg):,}")

        if max_chunks is not None and chunks_seen >= int(max_chunks):
            break

    if not agg:
        out = pd.DataFrame(columns=["date", "ai_row_count", "factor_sum", "volume_sum", "sentiment_wavg", "factor_zscore"])
        out.to_csv(out_daily_csv, index=False, encoding="utf-8-sig")
        return out

    out = (
        pd.DataFrame.from_dict(agg, orient="index")
        .rename_axis("date")
        .reset_index()
        .rename(columns={"row_count": "ai_row_count"})
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["sentiment_wavg"] = np.where(out["volume_sum"] > 0, out["factor_sum"] / out["volume_sum"], 0.0)

    roll = out["factor_sum"].rolling(window=30, min_periods=10)
    out["factor_zscore"] = (out["factor_sum"] - roll.mean()) / roll.std(ddof=0)

    out.to_csv(out_daily_csv, index=False, encoding="utf-8-sig")
    print(f"Saved daily factor to: {out_daily_csv} (days={len(out):,}, scanned_rows={rows_seen:,}, ai_rows={rows_ai:,})")
    return out

def _parse_date_ymd(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()
    def calculate_narrative_intensity(self, text):
        """
        计算单篇文章的 AI 叙事强度
        强度 = AI关键词出现的次数 / 文本总长度 (标准化到每千字)
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        
        matches = self.keyword_pattern.findall(text)
        word_count = len(text.split()) if text.isascii() else len(text) # 粗略中英文分词
        
        if word_count == 0:
            return 0.0
            
        intensity = (len(matches) / word_count) * 1000  # 每千字出现的频率
        return intensity

    def extract_ai_sentences(self, text):
        """
        提取包含 AI 关键词的句子，以便更精准地进行情感分析
        而不是把整篇文章丢给模型（避免噪声）
        """
        # 简单的分句逻辑（支持中英文句号、感叹号、问号）
        sentences = re.split(r'[.!?。！？]', text)
        ai_sentences = [s.strip() for s in sentences if self.keyword_pattern.search(s) and len(s.strip()) > 5]
        return ai_sentences

    def calculate_sentiment(self, text):
        """
        计算文本的 AI 情绪得分
        返回：情绪得分 (-1 到 1 之间)
        """
        ai_sentences = self.extract_ai_sentences(text)
        if not ai_sentences:
            return 0.0 # 没有提及 AI，情绪为中立
            
        # 截断处理，防止超出 BERT 的 512 token 限制
        # 对于多句，我们取前 10 句最重要的进行分析（或者你可以做滑动窗口）
        ai_sentences = ai_sentences[:10] 
        
        results = self.nlp_pipeline(ai_sentences)
        
        score = 0.0
        # FinBERT labels: 'positive', 'neutral', 'negative'
        for res in results:
            label = res['label']
            prob = res['score']
            
            if label == 'positive':
                score += prob
            elif label == 'negative':
                score -= prob
            # neutral 记为 0
            
        # 平均化单篇文章的情绪得分 (-1 到 1)
        avg_score = score / len(ai_sentences)
        return avg_score

    def process_corpus(self, df):
        """
        处理爬取的数据集
        输入 df 应包含 ['date', 'source', 'title', 'content']
        """
        print(f"Processing {len(df)} documents...")
        
        df['ai_intensity'] = df['content'].apply(self.calculate_narrative_intensity)
        
        # 只对有一定 AI 提及强度的文章进行情绪计算节省算力）
        df['ai_sentiment'] = 0.0
        mask = df['ai_intensity'] > 0
        df.loc[mask, 'ai_sentiment'] = df.loc[mask, 'content'].apply(self.calculate_sentiment)
        
        # 计算综合因子：强度 * 情绪 (Intensity-weighted Sentiment)
        df['ai_narrative_factor'] = df['ai_intensity'] * df['ai_sentiment']
        
        return df

    def aggregate_daily_factor(self, processed_df):
        """
        合成时间序列因子，可直接输入你的量化回测引擎
        """
        processed_df['date'] = pd.to_datetime(processed_df['date']).dt.date
        
        daily_factor = processed_df.groupby('date').agg(
            doc_count=('content', 'count'),
            avg_intensity=('ai_intensity', 'mean'),
            avg_sentiment=('ai_sentiment', 'mean'),
            aggregate_factor=('ai_narrative_factor', 'sum') # 当日总的AI叙事冲击
        ).reset_index()
        
        # 归一化处理 (Z-score) 便于输入模型
        daily_factor['factor_zscore'] = (daily_factor['aggregate_factor'] - daily_factor['aggregate_factor'].rolling(window=30).mean()) / daily_factor['aggregate_factor'].rolling(window=30).std()
        
        return daily_factor

# ================= 测试用例 =================
if __name__ == "__main__":
    # Default workflow: stream TSLA_AI.csv and export a daily AI narrative factor.
    USE_TSLA_AI_CSV = True
    if USE_TSLA_AI_CSV:
        csv_path = r"D:\TSLA_AI.csv"
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        daily = build_tsla_ai_narrative_daily_from_rpa_csv(
            csv_path,
            out_daily_csv="tsla_ai_narrative_daily_from_rpa.csv",
            chunksize=300_000,
            min_relevance=90,
            progress_every_chunks=1,
        )
        print(daily.tail(10))
        raise SystemExit(0)

    # 模拟爬取到的新闻、公告和研报数据
    mock_data = [
        {
            "date": "2024-03-01",
            "source": "Morgan Stanley",
            "title": "Tesla's Dojo Supercomputer Could Add $500B to Enterprise Value",
            "content": "We believe Tesla's Dojo supercomputer can add up to $500 billion to its enterprise value, driven by a faster adoption rate of Mobility (Robotaxi) and Network Services. Dojo is a game-changer. The artificial intelligence capability is overwhelmingly positive."
        },
        {
            "date": "2024-03-01",
            "source": "Reuters",
            "title": "Tesla delays FSD beta rollout again",
            "content": "Tesla faces massive regulatory hurdles for its full self-driving technology. The new FSD version has been delayed, and engineers are struggling with the neural network's edge cases. The outlook for autonomous driving remains uncertain and challenging."
        },
        {
            "date": "2024-03-02",
            "source": "Company Announcement",
            "title": "Q1 Production Update",
            "content": "Tesla delivered 400,000 vehicles this quarter. Gross margins remained stable at 18%."
        }
    ]
    
    df = pd.DataFrame(mock_data)
    
    # 实例化并计算因子
    factor_builder = TeslaAINarrativeFactor()
    
    # 获取单篇打分
    processed_data = factor_builder.process_corpus(df)
    print("\n--- Document Level NLP Results ---")
    print(processed_data[['source', 'ai_intensity', 'ai_sentiment', 'ai_narrative_factor']])
    
    # 合成每日因子
    daily_ts = factor_builder.aggregate_daily_factor(processed_data)
    print("\n--- Daily Factor Time Series ---")
    print(daily_ts)