```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map required columns to proper case
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_mapping, inplace=True)

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquidationReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Funding Rate Indicators 🌙
        funding_series = self.data.df['funding_rate'].ffill()
        self.funding_ma = self.I(talib.SMA, funding_series, 100, name='Funding MA')
        self.funding_std = self.I(talib.STDDEV, funding_series, 100, name='Funding Std')
        
        # Volume Analysis 📈
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20, name='Vol MA20')
        
        # Volatility Measures 🌊
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_ma = self.I(talib.SMA, self.atr, 10, name='ATR MA10')
        
        # Swing Points 🏔️
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing Low')
        
        # Momentum Indicators 🚀
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI14')
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        if len(self.data.Close) < 100:  # Ensure enough data
            return
            
        # Current indicator values 🌙
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
            
            # Short setup: Extreme positive funding + bearish confirmation
            elif funding_z > 2 and vol_spike:
                bearish_engulf = self.engulfing[-1] == -100
                rsi_overbought = self.rsi[-1] > 70
                if bearish_engulf or rsi_overbought:
                    self.enter_short()
        
        # Moon Dev Exit Logic 🌑🎯
        else:
            if atr_condition:
                self.position.close()
                print(f"🌙✨ Volatility contraction! Closing position at {self.data.Close[-1]:.2f}")

    def enter_long(self):
        price = self.data.Close[-1]
        sl = self.swing_low[-1]
        risk = price - sl
        if risk <= 0: return
        
        size = int(round((self.equity * self.risk_percent) / risk))
        tp = price + 2*risk  # 2:1 reward
        self.buy(size=size, sl=sl, tp=tp)
        print(f"