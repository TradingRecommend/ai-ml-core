from typing import List
import requests
import pandas_ta as ta  # Thay đổi từ talib sang pandas_ta
from datetime import datetime
import pandas as pd

class BuildTopCoinFeature:
    def __init__(self):
        pass

    @staticmethod
    def get_price_volume_binance(date=str, symbol="BTCUSDT", interval="1d", limit=30):
        end_date = datetime.strptime(date, "%Y%m%d")
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        end_time_ms = int(end_date.timestamp() * 1000)

        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit, "endTime": end_time_ms}
        response = requests.get(url, params=params)
        data = response.json()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["date", "close", "volume", "open", "high", "low"]]

    @staticmethod
    def build_top_coin_features(data: List[object]):
        result_df = pd.DataFrame()
        for item in data:
            end_date = item.get('date', '')
            symbol = item.get('symbol', '')
            symbol_usdt = symbol + "USDT"

            # Lấy 150 nến để đảm bảo đủ dữ liệu cho các chỉ báo chu kỳ dài (MA50, MACD)
            df = BuildTopCoinFeature.get_price_volume_binance(end_date, symbol_usdt, "1d", 150)

            if not all(col in df.columns for col in ['open', 'close', 'high', 'low', 'volume']):
                raise ValueError("Dữ liệu thiếu các cột cần thiết")

            # --- TÍNH TOÁN CƠ BẢN ---
            df['price_change_ratio'] = (df['close'] - df['open']) / df['open']

            # Volume Ratios
            for days in [7, 20, 50]:
                df[f'volume_mean_{days}_days'] = df['volume'].rolling(window=days).mean()
                df[f'volume_mean_{days}_days_ratio'] = df['volume'] / df[f'volume_mean_{days}_days']

            # --- PANDAS-TA INDICATORS ---
            
            # Moving Averages
            ma_periods = [7, 9, 14, 18, 20, 26, 50, 52]
            for p in ma_periods:
                df[f'MA{p}'] = df.ta.sma(length=p)

            # Các giá trị shift để tính trend
            df['MA7_prev7'] = df['MA7'].shift(7)
            df['MA9_prev7'] = df['MA9'].shift(7)
            df['MA26_prev7'] = df['MA26'].shift(7)
            df['MA50_prev7'] = df['MA50'].shift(7)

            # RSI & MFI
            df['RSI'] = df.ta.rsi(length=14)
            df['MFI'] = df.ta.mfi(length=14)

            # MACD: Trả về DataFrame (MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
            macd = df.ta.macd(fast=12, slow=26, signal=9)
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']

            # Stochastic: Trả về (STOCHk_14_3_3, STOCHd_14_3_3)
            stoch = df.ta.stoch(high='high', low='low', close='close', k=14, d=3, smooth_k=3)
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']

            # ADX: Trả về (ADX_14, DMP_14, DMN_14)
            adx = df.ta.adx(length=14)
            df['ADX'] = adx['ADX_14']

            # --- TRÍCH XUẤT DÒNG CUỐI & RATIOS ---
            df_tail = df.tail(1).copy()

            df_tail['MA9_MA26_ratio'] = df_tail['MA9'] / df_tail['MA26']
            df_tail['MA9_MA50_ratio'] = df_tail['MA9'] / df_tail['MA50']
            df_tail['MA26_MA50_ratio'] = df_tail['MA26'] / df_tail['MA50']

            df_tail['MA9_trend'] = df_tail['MA9'] / df_tail['MA9_prev7']
            df_tail['MA26_trend'] = df_tail['MA26'] / df_tail['MA26_prev7']
            df_tail['MA50_trend'] = df_tail['MA50'] / df_tail['MA50_prev7']

            # Filter columns
            cols = ['date', 'volume', 'open', 'close', 'price_change_ratio', 'volume_mean_7_days_ratio', 
                    'volume_mean_20_days_ratio', 'volume_mean_50_days_ratio', 'MA7', 'MA9', 'MA14', 'MA18', 
                    'MA20', 'MA26', 'MA50', 'MA52', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                    'Stoch_K', 'Stoch_D', 'ADX', 'MFI', 'MA9_MA26_ratio', 'MA9_MA50_ratio', 
                    'MA26_MA50_ratio', 'MA9_trend', 'MA26_trend', 'MA50_trend']

            df_tail = df_tail[cols]
            df_tail['date'] = pd.to_datetime(df_tail['date']).dt.strftime('%Y%m%d')
            df_tail['symbol'] = symbol
            
            result_df = pd.concat([result_df, df_tail], ignore_index=True)
            
        return result_df