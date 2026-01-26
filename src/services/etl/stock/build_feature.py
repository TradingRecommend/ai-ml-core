from time import sleep
from src.config.logger import Logger
from typing import List
from vnstock import Vnstock
import talib
from datetime import datetime, timedelta
import pandas as pd

pd.options.mode.chained_assignment = None  # Suppress the warning
class BuildStockFeature:
    def __init__(self):
        pass


    """
        Build stock features from the provided data.
    """
    @staticmethod
    def build_stock_features(data: List[object]):
        logger = Logger(BuildStockFeature.__name__)

        result_df = pd.DataFrame()
        for item in data:
            # Ngày kết thúc cụ thể
            end_date = item.get('date', '')
            symbol = item.get('symbol', '')

            logger.info(f"Building features for symbol: {symbol}, date: {end_date}")

            end_date_obj = datetime.strptime(end_date, '%Y%m%d')

            # Tính ngày bắt đầu (100 ngày trước ngày kết thúc)
            # Lấy khoảng thời gian lớn hơn để đảm bảo đủ dữ liệu
            start_date_obj = end_date_obj - timedelta(days=200)
            start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')

            # Lấy dữ liệu lịch sử giá cổ phiếu
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            df = stock.quote.history(
                start=start_date, end=end_date, interval='1D')

            # Lấy 100 dòng dữ liệu gần nhất
            df = df[df['time'] <= end_date_obj]
            df = df.tail(100)

            # Đảm bảo cột 'close', 'high', 'low', và 'volume' tồn tại trong dữ liệu
            if not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
                raise ValueError(
                    "Dữ liệu thiếu các cột cần thiết: 'close', 'high', 'low', 'volume'")

            # Tính phần trăm tăng/giảm giá
            df['price_change_ratio'] = (
                (df['close'] - df['open']) / df['open'])

            # Calculate the average volume for the last 7 days
            df['volume_mean_7_days'] = df['volume'].rolling(window=7).mean()
            df['volume_mean_7_days_ratio'] = df['volume'] / \
                df['volume_mean_7_days']

            df['volume_mean_20_days'] = df['volume'].rolling(window=20).mean()
            df['volume_mean_20_days_ratio'] = df['volume'] / \
                df['volume_mean_20_days']

            df['volume_mean_50_days'] = df['volume'].rolling(window=50).mean()
            df['volume_mean_50_days_ratio'] = df['volume'] / \
                df['volume_mean_50_days']

            # Tính các chỉ số kỹ thuật
            df['MA7'] = talib.SMA(df['close'], timeperiod=7)   # MA 7 ngày
            df['MA7_prev7'] = df['MA7'].shift(7)
            df['MA9'] = talib.SMA(df['close'], timeperiod=9)   # MA 9 ngày
            df['MA9_prev7'] = df['MA9'].shift(7)
            df['MA14'] = talib.SMA(df['close'], timeperiod=14)   # MA 14 ngày
            df['MA18'] = talib.SMA(df['close'], timeperiod=18)   # MA 18 ngày
            df['MA20'] = talib.SMA(df['close'], timeperiod=20)  # MA 20 ngày
            df['MA26'] = talib.SMA(df['close'], timeperiod=26)   # MA 26 ngày
            df['MA26_prev7'] = df['MA26'].shift(7)
            df['MA50'] = talib.SMA(df['close'], timeperiod=50)  # MA 50 ngày
            df['MA50_prev7'] = df['MA50'].shift(7)
            df['MA52'] = talib.SMA(df['close'], timeperiod=52)  # MA 52 ngày
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)  # RSI 14 ngày
            df['MFI'] = talib.MFI(df['high'], df['low'], df['close'],
                                  df['volume'], timeperiod=14)  # Tính MFI bằng TA-Lib
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
                # Stochastic Oscillator
                df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
            df['ADX'] = talib.ADX(df['high'], df['low'],
                                  df['close'], timeperiod=14)  # ADX

            # Get the last row of the DataFrame
            df_tail = df.tail(1)

            df_tail['MA9_MA26_ratio'] = df_tail['MA9'] / df_tail['MA26']
            df_tail['MA9_MA50_ratio'] = df_tail['MA9'] / df_tail['MA50']
            df_tail['MA26_MA50_ratio'] = df_tail['MA26'] / df_tail['MA50']

            # MA7 trend (assuming you're using values from df, not df_tail)
            df_tail['MA9_trend'] = df_tail['MA9'] / df_tail['MA9_prev7']
            df_tail['MA26_trend'] = df_tail['MA26'] / df_tail['MA26_prev7']
            df_tail['MA50_trend'] = df_tail['MA50'] / df_tail['MA50_prev7']

            df_tail = df_tail[['time', 'volume', 'open', 'close', 'price_change_ratio', 'volume_mean_7_days_ratio', 'volume_mean_20_days_ratio', 
                               'volume_mean_50_days_ratio', 'MA7', 'MA9', 'MA14', 'MA18', 'MA20', 'MA26', 'MA50', 'MA52', 'RSI',
                               'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D', 'ADX', 'MFI', 'MA9_MA26_ratio',
                               'MA9_MA50_ratio', 'MA26_MA50_ratio',
                               'MA9_trend', 'MA26_trend', 'MA50_trend'
                               ]]

            df_tail = df_tail.rename(columns={'time': 'date'})
            df_tail['date'] = pd.to_datetime(
                df_tail['date']).dt.strftime('%Y%m%d')
            df_tail[['symbol']] = symbol

            result_df = pd.concat([result_df, df_tail], ignore_index=True)

            sleep(10)
        return result_df
