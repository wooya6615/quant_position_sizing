"""
가격 데이터 로더 (yfinance).
국면 클러스터링은 변동성/모멘텀만 쓰므로 pykrx(수급/밸류에이션) 데이터는 불필요.

quant_unsupervised/src/data/price_data.py를 그대로 이식 (자기완결적 레포 컨벤션).
"""

import pandas as pd
import yfinance as yf


def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    ticker: 종목 코드 (예: 현대로템 = "064350.KS")
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()