from typing import List
from vnstock import Vnstock
import talib
from datetime import datetime, timedelta
import pandas as pd

class BuildStockFeature:
    def __init__(self):
        pass

    """
        Build stock features from the provided data.
    """
    @staticmethod
    def build_stock_features(data: List[object]):
        result_df = pd.DataFrame()
        for item in data:
            # Ngày kết thúc cụ thể
            end_date = item.get('date', '')
            symbol = item.get('symbol', '') 

            end_date_obj = datetime.strptime(end_date, '%Y%m%d')

            # Tính ngày bắt đầu (100 ngày trước ngày kết thúc)
            # Lấy khoảng thời gian lớn hơn để đảm bảo đủ dữ liệu
            start_date_obj = end_date_obj - timedelta(days=200)
            start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')

            # Lấy dữ liệu lịch sử giá cổ phiếu
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            df = stock.quote.history(start=start_date, end=end_date, interval='1D')

            # Lấy 100 dòng dữ liệu gần nhất
            df = df[df['time'] <= end_date_obj]
            df = df.tail(100)

            # Đảm bảo cột 'close', 'high', 'low', và 'volume' tồn tại trong dữ liệu
            if not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
                raise ValueError(
                    "Dữ liệu thiếu các cột cần thiết: 'close', 'high', 'low', 'volume'")

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

            df = df[['time', 'volume', 'open', 'close', 'price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 'volume_mean_50_days_ratio', 'MA7', 'MA20', 'MA50', 'RSI',
                    'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI']]
        
            # Get the last row of the DataFrame
            df_tail = df.tail(1)

            df_tail = df_tail.rename(columns={'time': 'date'})
            df_tail['date'] = pd.to_datetime(df_tail['date']).dt.strftime('%Y%m%d')
            df_tail[['symbol']] = symbol

            result_df = pd.concat([result_df, df_tail], ignore_index=True)
        return result_df


