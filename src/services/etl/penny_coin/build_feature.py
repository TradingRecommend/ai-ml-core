from typing import List
import requests
import pandas_ta as ta  # Thay đổi talib sang pandas_ta
from datetime import datetime
import pandas as pd

class BuildPennyCoinFeature:
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
    def build_penny_coin_features(data: List[object]):
        result_df = pd.DataFrame()
        for item in data:
            end_date = item.get('date', '')
            symbol = item.get('symbol', '')
            symbol_usdt = symbol + "USDT"

            # Lấy 150 nến để đảm bảo các chỉ số MA50 và MACD đủ dữ liệu để hội tụ chính xác
            df = BuildPennyCoinFeature.get_price_volume_binance(end_date, symbol_usdt, "1d", 150)

            if not all(col in df.columns for col in ['open', 'close', 'high', 'low', 'volume']):
                raise ValueError("Dữ liệu thiếu các cột cần thiết: 'close', 'volume'")

            # --- TÍNH TOÁN CƠ BẢN ---
            df['price_change_ratio'] = (df['close'] - df['open']) / df['open']

            # Volume Mean Ratios
            for period in [7, 20, 50]:
                df[f'volume_mean_{period}_days'] = df['volume'].rolling(window=period).mean()
                df[f'volume_mean_{period}_days_ratio'] = df['volume'] / df[f'volume_mean_{period}_days']

            # --- PANDAS-TA INDICATORS ---

            # MA Ratios (Giá hiện tại / Đường trung bình)
            # Lưu ý: Giữ nguyên logic tính ratio của bạn thay vì chỉ lấy đường MA
            df['MA7'] = df['close'] / df.ta.sma(length=7)
            df['MA20'] = df['close'] / df.ta.sma(length=20)
            df['MA50'] = df['close'] / df.ta.sma(length=50)

            # RSI & MFI
            df['RSI'] = df.ta.rsi(length=14)
            df['MFI'] = df.ta.mfi(length=14)

            # MACD (Trả về MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
            macd = df.ta.macd(fast=12, slow=26, signal=9)
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']

            # Stochastic (Trả về STOCHk_14_3_3, STOCHd_14_3_3)
            stoch = df.ta.stoch(high='high', low='low', close='close', k=14, d=3, smooth_k=3)
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']

            # ADX (Trả về ADX_14, DMP_14, DMN_14)
            adx = df.ta.adx(length=14)
            df['ADX'] = adx['ADX_14']

            # --- LỌC CỘT VÀ LẤY DÒNG CUỐI ---
            cols_to_keep = [
                'date', 'volume', 'open', 'close', 'price_change_ratio', 
                'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 
                'volume_mean_50_days_ratio', 'MA7', 'MA20', 'MA50', 'RSI',
                'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI'
            ]
            
            df = df[cols_to_keep]
            df_tail = df.tail(1).copy()

            df_tail['date'] = pd.to_datetime(df_tail['date']).dt.strftime('%Y%m%d')
            df_tail['symbol'] = symbol

            result_df = pd.concat([result_df, df_tail], ignore_index=True)
            
        return result_df