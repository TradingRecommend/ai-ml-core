from typing import List
import requests
from vnstock import Vnstock
import talib
from datetime import datetime, timedelta
import pandas as pd

class BuildTopCoinFeature:
    def __init__(self):
        pass

    @staticmethod
    def get_price_volume_binance(date=str, symbol="BTCUSDT", interval="1d", limit=30):
        # Convert end_date to Unix timestamp in milliseconds (end of day)
        end_date = datetime.strptime(date, "%Y%m%d")
        # Set to end of the day (23:59:59.999) to ensure the date is included
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        end_time_ms = int(end_date.timestamp() * 1000)  # Convert to milliseconds

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
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df[["date", "close", "volume", "open", "high", "low"]]


    @staticmethod
    def build_top_coin_features(data: List[object]):
        result_df = pd.DataFrame()
        for item in data:
            # Ngày kết thúc cụ thể
            end_date = item.get('date', '')
            symbol = item.get('symbol', '')
            symbol_usdt = symbol + "USDT"


            # Lấy dữ liệu lịch sử giá cổ phiếu
            df = BuildTopCoinFeature.get_price_volume_binance(end_date, symbol_usdt, "1d", 100)

            # Đảm bảo cột 'close', 'high', 'low', và 'volume' tồn tại trong dữ liệu
            if not all(col in df.columns for col in ['open', 'close', 'high', 'low', 'volume']):
                raise ValueError(
                    "Dữ liệu thiếu các cột cần thiết: 'close', 'volume'")

            # Tính phần trăm tăng/giảm giá
            df['price_change_ratio'] = ((df['close'] - df['open']) / df['open'])

            # Calculate the average volume for the last 7 days
            df['volume_mean_7_days'] = df['volume'].rolling(window=7).mean()
            df['volume_mean_7_days_ratio'] = df['volume'] / df['volume_mean_7_days']

            df['volume_mean_20_days'] = df['volume'].rolling(window=20).mean()
            df['volume_mean_20_days_ratio'] = df['volume'] / \
                df['volume_mean_20_days']

            df['volume_mean_50_days'] = df['volume'].rolling(window=50).mean()
            df['volume_mean_50_days_ratio'] = df['volume'] / \
                df['volume_mean_50_days']

            # Tính các chỉ số kỹ thuật
            df['MA7'] = df['close'] / \
                talib.SMA(df['close'], timeperiod=7)   # MA 7 ngày
            df['MA20'] = df['close'] / \
                talib.SMA(df['close'], timeperiod=20)  # MA 20 ngày
            df['MA50'] = df['close'] / \
                talib.SMA(df['close'], timeperiod=50)  # MA 50 ngày
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)  # RSI 14 ngày
            df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)  # Tính MFI bằng TA-Lib
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
                # Stochastic Oscillator
                df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
            df['ADX'] = talib.ADX(df['high'], df['low'],
                                df['close'], timeperiod=14)  # ADX

            df = df[['date', 'volume', 'open', 'close', 'price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'volume_mean_50_days_ratio', 'MA7', 'MA20', 'MA50', 'RSI',
                    'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI']]
        
            # Get the last row of the DataFrame
            df_tail = df.tail(1)

            df_tail['date'] = pd.to_datetime(df_tail['date']).dt.strftime('%Y%m%d')
            df_tail[['symbol']] = symbol

            result_df = pd.concat([result_df, df_tail], ignore_index=True)
        return result_df


