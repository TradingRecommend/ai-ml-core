from datetime import datetime
import os
import requests

from src.repository.prediction_result import PredictionResultRepository

class StockTelegramNotifier:
    def __init__(self, date):
        self.date = date

        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        self.prediction_result_repository = PredictionResultRepository()

    def send_message(self):
        prediction_result = self.prediction_result_repository.get_by_date_and_type(
            date=self.date,
            type="1",
            prediction_limit=0.94
        )

        stocks = [result.symbol for result in prediction_result]
        date_formatted = datetime.strptime(self.date, "%Y%m%d").strftime("%d/%m/%Y")
        
        if len(stocks) == 0:
            message = (
                f"📢 ** BẢN TIN GỢI Ý CHỨNG KHOÁN - {date_formatted} **\n\n"
                f"Hôm nay chưa có cổ phiếu nào nổi bật đạt tiêu chí khuyến nghị.\n"
                f"#NelsonInvestment #NelsonTradingRecommendation"
            )
        else:
            stock_list = "\n".join(f"- {s}" for s in stocks)
            message = (
                f"🔥 ** BẢN TIN GỢI Ý CHỨNG KHOÁN - {date_formatted} **\n\n"
                f"💎 Cổ phiếu đáng chú ý hôm nay:\n{stock_list}\n\n"
                f"#NelsonInvestment #NelsonTradingRecommendation"
            )

        # === SEND TO TELEGRAM ===
        payload = {"chat_id": self.chat_id, "text": message}

        requests.post(self.api_url, data=payload)
