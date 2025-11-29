Here's the fixed and completed backtest code with Moon Dev themed debug prints and proper position sizing:

```python
# 🌙 Moon Dev's Chaikin Divergence Breakout Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# 🚀 Data Preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🌙 Clean and map columns to backtesting.py format
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class ChaikinDivergenceBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # ✨ Indicator Calculations
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, timeperiod=20)
        
        # MACD Histogram Calculation 🌙
        def _macd_hist(close):
            macd, macdsignal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return hist
        self.macd_hist = self.I(_macd_hist, self.data.Close)
        
        # Swing Points and Volatility 🌌
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Volume Analysis 📊
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
    
    def next(self):
        current_close = self.data.Close[-1]
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        
        # 🌙 Position Sizing Calculation
        def calculate_size(entry_price, stop_price):
            risk_amount = self.equity * self.risk_per_trade
            risk_per_share = abs(entry_price - stop_price)
            return int(round(risk_amount / risk_per_share)) if risk_per_share > 0 else 0
        
        # 🚀 Long Entry Logic
        if not self.position:
            # CMF Crossover Check
            cmf_bullish = (self.cmf[-2] <= 0 and self.cmf[-1] > 0)
            
            # MACD Bullish Divergence 🌙
            price_lower_low = current_low < self.data.Low[-2]
            hist_higher_low = self.macd_hist[-1] > self.macd_hist[-2]
            divergence_bull = price_lower_low and hist_higher_low
            
            # Breakout Confirmation ✨
            swing_break = current_close > self.swing_high[-2]
            
            if cmf_bullish and divergence_bull and swing_break:
                # Stop Logic 🌌
                stop_price = min(self.swing_low[-1], current_close - 1.5*self.atr[-1])
                size = calculate_size(current_close, stop_price)
                if size > 0:
                    tp = current_close + 2*(current_close - stop_price)
                    self.buy(size=size, sl=stop_price, tp=tp)
                    print(f"🌙✨🚀 LONG ENTRY @ {current_close:.2f} | Size: {size} units | SL: {stop_price:.2f} | TP: {tp:.2f}")
        
        # 🌑 Short Entry Logic
        if not self.position:
            # CMF Crossunder Check
            cmf_bearish = (self.cmf[-2] >= 0 and self.cmf[-1] < 0)
            
            # MACD Bearish Divergence 🌙
            price_higher_high = current_high > self.data.High[-2]
            hist_lower_high = self.macd_hist[-1] < self.macd_hist[-2]