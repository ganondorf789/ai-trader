```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
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

        else:
            # Exit conditions for long positions
            if self.position.is_long:
                if price < self.recent_high[-1]:
                    self.position.close(tag="🌌 Liquidation Zone Breach")
                elif self.atr[-1] < self.atr_sma[-1]:
                    self.position.close(tag="📉 ATR Momentum Loss")
                    
            # Exit conditions for short positions        
            elif self.position.is_short:
                if price > self.recent_low[-1]:
                    self.position.close(tag="🌌 Liquidation Zone Breach")
                elif self.atr[-1] < self.atr_sma[-1]:
                    self.position.close(tag="📉 ATR Momentum Loss")

            # Universal trailing stop
            for trade in self.trades:
                if trade.is_long:
                    trail_price = max(self.data.High[trade.entry_bar:self.current_bar+1]) - 2*atr
                    if self.data.Low[-1] < trail_price:
                        trade.close(tag="🌠 Trailing Stop")
                else:
                    trail_price = min(self.data.Low[t