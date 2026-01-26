# api/cron.py
from datetime import datetime, timedelta, timezone
import os
from http.server import BaseHTTPRequestHandler

from src.services.etl.composite import ETLComposite
from src.services.etl.stock.etl_prediction_feature import ETLPredictionStockFeature
from src.services.noti.stock.telegram import StockTelegramNotifier
from src.services.prediction.composite import PredictionComposite
from src.services.prediction.stock.stock_logistic_prediction import StockLogisticPrediction

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # 1. Security Check
        auth_header = self.headers.get('Authorization')
        if auth_header != f"Bearer {os.environ.get('CRON_SECRET')}":
            self.send_response(401)
            self.end_headers()
            return
        
        hanoi_tz = timezone(timedelta(hours=7))
        date = datetime.now(hanoi_tz).strftime('%Y%m%d')

        try:
            # 2. Trigger your project logic
            # Example: Running an ETL service
            composite_etl = ETLComposite()
            composite_prediction_model = PredictionComposite()
            
            composite_etl.add_operation(ETLPredictionStockFeature(date=date)) 

            composite_prediction_model.add_operation(StockLogisticPrediction(date=date)) 
       
            # Run the ETL composite
            composite_etl.run()

            # Run the prediction model
            composite_prediction_model.run()

            notifier = StockTelegramNotifier(date=date)
            notifier.send_message()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Success".encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode('utf-8'))