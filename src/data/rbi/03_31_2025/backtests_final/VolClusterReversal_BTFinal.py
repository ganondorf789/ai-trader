Here's the fixed and completed code with Moon Dev themed debug prints and proper position sizing:

```python
# 🌙 Moon Dev's VolClusterReversal Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# Clean and prepare data
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

class VolClusterReversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    rsi_period = 14
    swing_window = 20
    vwma_length = 20
    atr_period = 14
    
    def init(self):
        # 🌙 Core Indicators
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period, name='RSI')
        
        # 🌙 Volatility-adjusted Levels
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window, name='Swing_Low')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # 🌙 Liquidation Clusters (Volatility-adjusted)
        self.long_trigger = self.I(lambda: self.swing_low - self.atr*1.5, name='Long_Trigger')
        self.short_trigger = self.I(lambda: self.swing_high + self.atr*1.5, name='Short_Trigger')
        
        # 🌙 Volume Divergence System
        def delta_vol(close, open, vol):
            return np.where(close > open, vol, -vol)
        self.delta = self.I(delta_vol, self.data.Close, self.data.Open, self.data.Volume, name='Delta')
        self.delta_ma = self.I(talib.SMA, self.delta, 20, name='Delta_MA')
        
        # 🌙 Volume-Weighted MA
        self.vwma = self.I(ta.vwma, self.data.Close, self.data.Volume, length=self.vwma_length, name='VWMA')

    def next(self):
        price = self.data.Close[-1]
        equity = self.equity
        
        # 🌙 Avoid over-trading
        if len(self.trades) >= 1:
            return
            
        # 🌙 Long Entry Logic
        if (price <= self.long_trigger[-1] and
            self.rsi[-1] < 35 and
            self.delta[-1] < self.delta_ma[-1] and
            price > self.vwma[-1]):
            
            sl = self.swing_low[-1] - self.atr[-1]*0.5
            tp = price + (price - sl)*2  # 1:2 RR
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / (price - sl)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl, tp=tp)
                print(f"🌙✨ LONG Signal! Size: {position_size} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} 🚀")

        # 🌙 Short Entry Logic
        elif (price >= self.short_trigger[-1] and
              self.rsi[-1] > 65 and
              self.delta[-1] > self.delta_ma[-1] and
              price < self.vwma[-1]):
            
            sl = self.swing_high[-1] + self.atr[-1]*0.5
            tp = price - (sl - price)*2  # 1:2 RR
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / (sl - price)))
            
            if position_size >