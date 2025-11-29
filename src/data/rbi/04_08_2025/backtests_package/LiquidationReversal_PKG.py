Here's the fixed code with all necessary adjustments and Moon Dev themed improvements: 🌙✨

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data with Moon Dev magic 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names with lunar precision 🌑
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map required columns to proper case with cosmic alignment ✨
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_mapping, inplace=True)

# Convert datetime and set index with time warp precision 🕰️
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquidationReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade - Moon Dev recommended 🌙
    
    def init(self):
        # Funding Rate Indicators 🌙✨
        funding_series = self.data.df['funding_rate'].ffill()
        self.funding_ma = self.I(talib.SMA, funding_series, timeperiod=100, name='Funding MA')
        self.funding_std = self.I(talib.STDDEV, funding_series, timeperiod=100, name='Funding Std')
        
        # Volume Analysis 📈🌕
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Vol MA20')
        
        # Volatility Measures 🌊🌑
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=10, name='ATR MA10')
        
        # Swing Points 🏔️✨
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        
        # Momentum Indicators 🚀🌙
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI14')
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        if len(self.data.Close) < 100:  # Ensure enough lunar cycles 🌑🌒🌓🌔🌕
            return
            
        # Current indicator values with Moon Dev precision 🌙
        funding_z = ((self.data.df['funding_rate'][-1] - self.funding_ma[-1]) 
                    / self.funding_std[-1] if self.funding_std[-1] != 0 else 0)
        vol_spike = self.data.Volume[-1] > self.vol_ma[-1] * 1.5
        atr_condition = self.atr[-1] < self.atr_ma[-1]
        
        # Moon Dev Entry Logic 🌑✨
        if not self.position:
            # Long setup: Extreme negative funding + bullish confirmation
            if funding_z < -2 and vol_spike:
                bullish_engulf = self.engulfing[-1] == 100
                rsi_oversold = self.rsi[-1] < 30
                if bullish_engulf or rsi_oversold:
                    self.enter_long()
                    print(f"🌙🚀 Moon Dev LONG signal activated! Bullish engulf: {bullish_engulf}, RSI: {self.rsi[-1]:.2f}")
            
            # Short setup: Extreme positive funding + bearish confirmation
            elif funding_z > 2 and vol_spike:
                bearish_engulf = self.engulfing[-1] == -100
                rsi_overbought = self.rsi[-1] > 70
                if bearish_engulf or rsi_overbought:
                    self.enter_short()
                    print(f"🌙🌑 Moon Dev SHORT signal activated! Bearish engulf: {bearish_engulf}, R