I'll debug the VoltaicSqueeze strategy while maintaining all the original logic. Here's the fixed version with Moon Dev enhancements:

```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# Moon Dev Data Loading 🌙
print("🌙 Loading cosmic market data...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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
print("✨ Data aligned with lunar cycles!")

class VoltaicSqueeze(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    atr_multiplier = 1.5  # Trailing stop multiplier
    
    def init(self):
        print("🌌 Initializing celestial indicators...")
        # Calculate core indicators using TA-Lib
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands Width
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        bb_width = (upper - lower) / middle
        self.bb_width = self.I(lambda: bb_width, name='BB_WIDTH')
        self.bb_width_avg = self.I(lambda: talib.SMA(bb_width, timeperiod=10), name='BBW_MA10')
        
        # Volume-Weighted MACD
        vw_price = close * self.data.Volume
        macd, macdsignal, macdhist = talib.MACD(vw_price, fastperiod=12, slowperiod=26, signalperiod=9)
        self.vw_macd_hist = self.I(lambda: macdhist, name='VW_MACD_HIST')
        
        # 2-period KST with pandas_ta
        close_series = pd.Series(close)
        kst = ta.kst(close_series, roc1=2, roc2=2, roc3=2, roc4=2, 
                     sma1=2, sma2=2, sma3=2, sma4=2, signal=2)
        self.kst_line = self.I(lambda: kst['KST_2_2_2_2_2_2_2_2'].values, name='KST_LINE')
        self.kst_signal = self.I(lambda: kst['SIGNAL_2_2_2_2_2_2_2_2'].values, name='KST_SIGNAL')
        
        # Volatility measures
        self.atr = self.I(lambda: talib.ATR(high, low, close, timeperiod=14), name='ATR')
        self.swing_high = self.I(lambda: talib.MAX(high, timeperiod=20), name='SWING_HIGH')
        self.swing_low = self.I(lambda: talib.MIN(low, timeperiod=20), name='SWING_LOW')
        print("🌟 Indicators calibrated to cosmic frequencies!")

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 40:
            return
            
        prev_i = len(self.data) - 2
        current_i = len(self.data) - 1
        
        # Calculate entry conditions
        bb_squeeze = (self.bb_width[prev_i] < self.bb_width_avg[prev_i])
        
        macd_surge = False
        if prev_i > 0 and self.vw_macd_hist[prev_i-1] != 0:
            hist_change = ((self.vw_macd_hist[prev_i] - self.vw_macd_hist[prev_i-1]) 
                          / abs(self.vw_macd_hist[prev_i-1]))
            macd_surge = hist_change > 0.2
            
        kst_bullish = (self.kst_line[prev_i] > self.kst_signal[prev_i] and
                      self.kst_line[prev_i-1] <= self.kst_signal[prev