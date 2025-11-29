Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidationSpike(Strategy):
    risk_pct = 0.01
    swing_period = 20
    atr_period = 14
    volume_multiplier = 1.5
    
    def init(self):
        # Core indicators
        self.recent_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.recent_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_sma = self.I(talib.SMA, self.atr, 5)
        self.oi_roc = self.I(talib.ROC, self.data.openinterest, 1)
        
        print("🌙 Lunar Indicators Activated! ✨ SMA20 | ATR14 | OI Momentum")
        print("🚀 Moon Dev Package AI initialized with pure TA-Lib indicators")

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if not self.position:
            # Long entry constellation 🌌
            if (price > self.recent_high[-1] and
                self.data.Volume[-1] > self.volume_sma[-1] * self.volume_multiplier and
                self.oi_roc[-1] > 0):
                
                sl = self.recent_low[-1]
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (price - sl)))
                
                self.buy(size=position_size, sl=sl, 
                        tag=f"🚀 MOON SHOT | SL: {sl:.2f}")
                print(f"🌕🌕 BUY SIGNAL! Size: {position_size} | Risk: {self.risk_pct*100}%")
                print("✨ Bullish crossover detected: Price breaking recent high with volume confirmation")

            # Short entry constellation 🌑
            elif (price < self.recent_low[-1] and
                  self.data.Volume[-1] > self.volume_sma[-1] * self.volume_multiplier and
                  self.oi_roc[-1] < 0):
                  
                sl = self.recent_high[-1]
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (sl - price)))
                
                self.sell(size=position_size, sl=sl,
                         tag=f"💎 DIAMOND HAND | SL: {sl:.2f}")
                print(f"🌑🌑 SELL SIGNAL! Size: {position_size} | Risk: {self.risk_pct*100}%")
                print("🌘 Bearish crossover detected: Price breaking recent low with volume confirmation")

        else:
            # Exit conditions for long positions
            if self.position.is_long:
                if price < self.recent_high[-1]:
                    self.position.close(tag="🌌 Liquidation Zone Breach")
                    print("🌗 Long position closed: Price fell below recent high")
                elif self.atr[-1] < self.atr_sma[-1]:
                    self.position.close(tag="📉 ATR Momentum Loss")
                    print("🌓 Long position closed: ATR momentum weakening")
                    
            # Exit conditions for short positions        
            elif self.position.is_short:
                if price > self.recent_low[-1]:
                    self.position.close(tag="🌌 Liquidation Zone Breach")
                    print("🌖 Short position closed: Price rose above recent low")
                elif self.atr