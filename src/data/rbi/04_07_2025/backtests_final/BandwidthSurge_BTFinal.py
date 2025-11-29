Here's the debugged version of your BandwidthSurge strategy with Moon Dev themed improvements! 🌙✨

```python
# 🌙 Moon Dev Backtest Engine: BandwidthSurge Strategy Implementation 🚀
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class BandwidthSurge(Strategy):
    def init(self):
        # 🌌 Cosmic Indicators Configuration
        self.risk_percent = 0.01  # 1% risk per trade
        
        # 🌀 Bollinger Bands Calculation
        def bb_upper(close): 
            return talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[0]
        def bb_lower(close): 
            return talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[2]
        self.upper_band = self.I(bb_upper, self.data.Close)
        self.lower_band = self.I(bb_lower, self.data.Close)
        
        # 📡 Bandwidth Calculation (Upper-Lower)/Middle
        def bb_middle(close):
            return talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[1]
        middle_band = self.I(bb_middle, self.data.Close)
        self.bandwidth = self.I(lambda u, l, m: (u - l) / m, 
                               self.upper_band, self.lower_band, middle_band)
        
        # 🌟 6-Month Low Filter (17280 periods for 15m data)
        self.bandwidth_low = self.I(talib.MIN, self.bandwidth, timeperiod=17280)
        
        # 📈 Volume Surge Indicator
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=4800)  # 50-day SMA
        
        # 🌪️ ADX Trend Filter
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # 🛑 Parabolic SAR Exit
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, acceleration=0.02, maximum=0.2)

    def next(self):
        # 🌙 Core Trading Logic
        if not self.position:
            # 🚀 Entry Conditions
            bandwidth_at_low = self.bandwidth[-1] <= self.bandwidth_low[-1]
            price_breakout = self.data.Close[-1] > self.upper_band[-1]
            volume_surge = self.data.Volume[-1] > (self.volume_sma[-1] * 1.2)
            low_trend_strength = self.adx[-1] < 20
            
            if all([bandwidth_at_low, price_breakout, volume_surge, low_trend_strength]):
                # 💰 Risk Management Calculation
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Close[-1]
                stop_loss = entry_price * 0.98  # 2% stop loss
                position_size = int(round(risk_amount / (entry_price - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌕 MOON DEV ALERT: Cosmic Breakout Detected! 🚀")
                    print(f"Entry: {entry_price:.2f} | Size: {position_size} units | SL: {stop_loss:.2f}")
        
        else:
            # 🌠 Exit Conditions
            if self.sar[-1] > self.data.Close[-1]:
                self.position.close()
                print(f"🌑 SAR Reversal Signal: Returning to Base Station � | Exit Price: {self.data.Close[-1]:.2f}")

# 🛰️ Data Loading & Preparation
try:
    data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv", 
                       parse_dates=['Date'], index_col='Date')
    
    # 🧹 Data Cleaning Ritual
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'