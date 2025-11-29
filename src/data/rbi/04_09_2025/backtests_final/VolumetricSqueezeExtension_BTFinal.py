Here's the complete fixed backtest code with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙
print("🌕 MOON DEV INITIALIZING DATA PREPARATION...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
print("🌖 DATA CLEANING COMPLETE - MOON STANDARDS MET")

class VolumetricSqueezeExtension(Strategy):
    initial_capital = 1_000_000
    risk_per_trade = 0.02  # 2% of equity per trade
    max_exposure = 0.10    # 10% maximum portfolio exposure
    daily_loss_limit = 0.05 # 5% daily loss limit
    
    def init(self):
        print("🌗 INITIALIZING MOON DEV STRATEGY COMPONENTS...")
        # Bollinger Bands
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, 
            self.data.Close, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0, 
            name=['Upper BB', 'Middle BB', 'Lower BB']
        )
        
        # Bollinger Bandwidth calculations
        self.bandwidth = (self.upper_band - self.lower_band) / self.middle_band
        self.bandwidth_sma20 = self.I(talib.SMA, self.bandwidth, 20, name='Bandwidth SMA20')
        
        # Volume indicators
        self.volume_sma50 = self.I(talib.SMA, self.data.Volume, 50, name='Volume SMA50')
        current_volume = self.data.Volume
        self.volume_ratio = current_volume / self.volume_sma50
        
        # VW-MACD calculations
        vw_close = self.data.Close * (current_volume / self.volume_sma50)
        self.macd_line, self.macd_signal, _ = self.I(
            talib.MACD, 
            vw_close, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9, 
            name='VW-MACD'
        )
        
        # Swing points
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        print("🌘 STRATEGY INITIALIZATION COMPLETE - READY FOR LIFTOFF")
        
    def next(self):
        # Moon Dev risk management checks 🌙
        if self.equity < self.initial_capital * (1 - self.daily_loss_limit):
            print(f"🌑 MOON DEV ALERT: Daily loss limit hit! Stopping trading.")
            self.position.close()
            return
            
        if len(self.data) < 50:
            return
            
        # Check bandwidth squeeze condition
        bandwidth_threshold = 0.2 * self.bandwidth_sma20[-1]
        squeeze_on = self.bandwidth[-1] < bandwidth_threshold
        
        # Volume confirmation
        volume_ok = self.data.Volume[-1] > 1.2 * self.volume_sma50[-1]
        
        # MACD crossover conditions
        macd_bullish = (self.macd_line[-2] < self.macd_signal[-2] and 
                        self.macd_line[-1] > self.macd_signal[-1])
        macd_bearish = (self.macd_line[-2] > self.macd_signal[-2] and 
                        self.macd_line[-1] < self.macd_signal[-1])
        
        # Price position relative to Bollinger Bands
        price_above_middle = self.data.Close[-1] > self.m