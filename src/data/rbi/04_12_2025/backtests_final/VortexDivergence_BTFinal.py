Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# Clean and prepare data 🌙✨
print("🌙 Initializing Moon Dev Backtest Environment...")
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime']
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VortexDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        print("🌙 Initializing Moon Dev Indicators...")
        # Vortex Indicator 🌪️
        self.vi_plus, self.vi_minus = self.I(
            ta.vortex,
            self.data.High, self.data.Low, self.data.Close,
            length=14, name='Vortex'
        )
        
        # Chaikin Money Flow 💰
        self.cmf = self.I(
            ta.cmf,
            self.data.High, self.data.Low, self.data.Close, self.data.Volume,
            length=20, name='CMF'
        )
        self.cmf_sma = self.I(
            talib.SMA, self.cmf,
            timeperiod=20, name='CMF_SMA'
        )
        
        # ADX for trend strength 📈
        self.adx = self.I(
            talib.ADX,
            self.data.High, self.data.Low, self.data.Close,
            timeperiod=14, name='ADX'
        )
        
        # Swing levels for stop loss 🛑
        self.swing_low = self.I(
            talib.MIN, self.data.Low,
            timeperiod=5, name='Swing_Low'
        )
        self.swing_high = self.I(
            talib.MAX, self.data.High,
            timeperiod=5, name='Swing_High'
        )
        print("🌙 Moon Dev Indicators Initialized Successfully! ✨")

    def next(self):
        if len(self.data) < 40:  # Ensure enough data for all indicators
            return
            
        # Current values 📊
        price = self.data.Close[-1]
        adx = self.adx[-1]
        cmf = self.cmf[-1]
        cmf_prev = self.cmf[-2]
        cmf_sma = self.cmf_sma[-1]
        
        # Exit conditions check 🚪
        if self.position:
            if self.position.is_long and cmf < cmf_sma:
                self.position.close()
                print(f"🌙🌑 MOON DEV LONG EXIT | CMF crossed below SMA: {cmf:.2f} < {cmf_sma:.2f}")
            elif self.position.is_short and cmf > cmf_sma:
                self.position.close()
                print(f"🌙🌕 MOON DEV SHORT EXIT | CMF crossed above SMA: {cmf:.2f} > {cmf_sma:.2f}")
            return
            
        # Entry conditions check 🔍
        # Long entry logic 🟢
        if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
            (self.data.Low[-1] < self.data.Low[-2] and cmf > cmf_prev) and
            (adx > 25)):
            
            sl_price = self.swing_low[-1]
            risk_amount = self.equity * self.risk_percent
            position_size = int(round(risk_amount / (price - sl_price)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price)
                print(f"🌙🚀 MOON DEV LONG ENTRY! Size: {position_size} | SL: {sl_price:.2f} ✨")
                print(f"🌙📊 VI+ crossed VI- | ADX: {adx:.1f} | CMF Bullish Divergence")
        
        # Short entry logic 🔴
        elif ((self.vi_minus[-2] < self.vi_plus[-