import re
from time import sleep
import traceback
from src.config.logger import Logger
from typing import List
from vnstock import Vnstock
import pandas_ta as ta  # Thay đổi ở đây
from datetime import datetime, timedelta
import pandas as pd

pd.options.mode.chained_assignment = None


class BuildStockFeature:
    def __init__(self):
        pass

    @staticmethod
    def build_stock_features(data: List[object]):
        logger = Logger(BuildStockFeature.__name__)
        result_df = pd.DataFrame()
        retry_max = 5
        item_cnt = 0

        for item in data:
            is_done = False
            retry_cnt = 0
            while not is_done:
                try:
                    if item_cnt >= 15:
                        item_cnt = 0
                        raise Exception("Rate limit reached, pausing...")
                    item_cnt += 1

                    end_date = item.get('date', '')
                    symbol = item.get('symbol', '')

                    logger.info(
                        f"Building features for symbol: {symbol}, date: {end_date}")

                    end_date_obj = datetime.strptime(end_date, '%Y%m%d')
                    start_date_obj = end_date_obj - timedelta(days=200)
                    start_date = start_date_obj.strftime('%Y-%m-%d')
                    end_date_str = end_date_obj.strftime('%Y-%m-%d')

                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    df = stock.quote.history(start=start_date, end=end_date_str, interval='1D')

                    if df is None or df.empty or len(df) < 100:
                        logger.warning("Insufficient data retrieved.")
                        is_done = True
                        continue

                    df['date'] = pd.to_datetime(
                        df['time']).dt.strftime('%Y%m%d')
                    if df.tail(1)['date'].values[0] != end_date:
                        logger.warning(
                            f"Data for symbol: {symbol} does not contain the end date: {end_date}. Skipping...")
                        is_done = True
                        continue

                    if not all(col in df.columns for col in ['close', 'high', 'low', 'volume', 'open']):
                        raise ValueError("Dữ liệu thiếu các cột cần thiết")

                    # --- TÍNH TOÁN BẰNG PANDAS-TA ---

                    # Phần trăm thay đổi
                    df['price_change_ratio'] = (
                        df['close'] - df['open']) / df['open']

                    # Volume Mean Ratios
                    df['volume_mean_7_days'] = df['volume'].rolling(
                        window=7).mean()
                    df['volume_mean_7_days_ratio'] = df['volume'] / \
                        df['volume_mean_7_days']

                    df['volume_mean_20_days'] = df['volume'].rolling(
                        window=20).mean()
                    df['volume_mean_20_days_ratio'] = df['volume'] / \
                        df['volume_mean_20_days']

                    df['volume_mean_50_days'] = df['volume'].rolling(
                        window=50).mean()
                    df['volume_mean_50_days_ratio'] = df['volume'] / \
                        df['volume_mean_50_days']

                    # Moving Averages (Sử dụng sma)
                    df['MA7']  = df.ta.sma(close='close', length=7)
                    df['MA9']  = df.ta.sma(close='close', length=9)
                    df['MA14'] = df.ta.sma(close='close', length=14)
                    df['MA18'] = df.ta.sma(close='close', length=18)
                    df['MA20'] = df.ta.sma(close='close', length=20)
                    df['MA26'] = df.ta.sma(close='close', length=26)
                    df['MA50'] = df.ta.sma(close='close', length=50)
                    df['MA52'] = df.ta.sma(close='close', length=52)

                    # Shifted values cho trend
                    df['MA7_prev7'] = df['MA7'].shift(7)
                    df['MA9_prev7'] = df['MA9'].shift(7)
                    df['MA26_prev7'] = df['MA26'].shift(7)
                    df['MA50_prev7'] = df['MA50'].shift(7)

                    # RSI & MFI
                    df['RSI'] = df.ta.rsi(length=14)
                    df['MFI'] = df.ta.mfi(length=14)

                    # MACD (Trả về DataFrame 3 cột: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
                    macd = df.ta.macd(fast=12, slow=26, signal=9)
                    df['MACD'] = macd['MACD_12_26_9']
                    df['MACD_hist'] = macd['MACDh_12_26_9']
                    df['MACD_signal'] = macd['MACDs_12_26_9']

                    # Stochastic (Trả về STOCHk_14_3_3, STOCHd_14_3_3)
                    stoch = df.ta.stoch(
                        high='high', low='low', close='close', k=14, d=3, smooth_k=3)
                    df['Stoch_K'] = stoch['STOCHk_14_3_3']
                    df['Stoch_D'] = stoch['STOCHd_14_3_3']

                    # ADX (Trả về ADX_14, DMP_14, DMN_14)
                    adx = df.ta.adx(length=14)
                    df['ADX'] = adx['ADX_14']

                    # Lấy dòng cuối cùng sau khi tính toán xong
                    df_tail = df.tail(1).copy()

                    # Ratios & Trends
                    df_tail['MA9_MA26_ratio'] = df_tail['MA9'] / \
                        df_tail['MA26']
                    df_tail['MA9_MA50_ratio'] = df_tail['MA9'] / \
                        df_tail['MA50']
                    df_tail['MA26_MA50_ratio'] = df_tail['MA26'] / \
                        df_tail['MA50']

                    df_tail['MA9_trend'] = df_tail['MA9'] / \
                        df_tail['MA9_prev7']
                    df_tail['MA26_trend'] = df_tail['MA26'] / \
                        df_tail['MA26_prev7']
                    df_tail['MA50_trend'] = df_tail['MA50'] / \
                        df_tail['MA50_prev7']

                    # Chọn cột kết quả
                    cols_to_keep = [
                        'date', 'volume', 'open', 'close', 'price_change_ratio',
                        'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio',
                        'volume_mean_50_days_ratio', 'MA7', 'MA9', 'MA14', 'MA18',
                        'MA20', 'MA26', 'MA50', 'MA52', 'RSI', 'MACD', 'MACD_signal',
                        'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI', 'MA9_MA26_ratio',
                        'MA9_MA50_ratio', 'MA26_MA50_ratio', 'MA9_trend', 'MA26_trend', 'MA50_trend'
                    ]

                    df_tail = df_tail[cols_to_keep]
                    df_tail['symbol'] = symbol

                    result_df = pd.concat(
                        [result_df, df_tail], ignore_index=True)
                    is_done = True

                except Exception as e:
                    logger.error(
                        f"Error processing symbol: {symbol}. Error: {e}")
                    retry_cnt += 1
                    if retry_cnt >= retry_max:
                        logger.error(
                            f"Max retries reached for symbol: {symbol}. Skipping...")
                        is_done = True
                    else:
                        logger.info("Retrying in 60 seconds...")
                        sleep(60)

        return result_df[result_df['date'] == end_date]
