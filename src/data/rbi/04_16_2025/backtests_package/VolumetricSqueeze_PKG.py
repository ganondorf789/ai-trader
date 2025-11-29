Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT 🌙
import pandas as pd
import talib
import pandas_ta as pd_ta
import numpy as np
from backtesting import Backtest, Strategy

# 🚀 DATA PREPARATION
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🌙 Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolumetricSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% equity risk
    consecutive_losses = 0
    
    def init(self):
        # 🌟 INDICATOR CALCULATIONS
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.data.Close, 
                                          timeperiod=20, nbdevup=2, nbdevdn=2)
        self.I(lambda: upper, name='BB_upper')
        self.I(lambda: middle, name='BB_middle')
        self.I(lambda: lower, name='BB_lower')
        
        # BB Width Percentile
        bb_width = (upper - lower) / middle * 100
        bb_width_pct = pd_ta.percentile(bb_width, length=20)
        self.I(lambda: bb_width_pct, name='BB_width_pct')
        
        # Klinger Volume Oscillator
        kvo, kvo_signal = pd_ta.kvo(self.data.High, self.data.Low, self.data.Close,
                                  self.data.Volume, fast=34, slow=55, signal=13)
        self.I(lambda: kvo, name='KVO')
        self.I(lambda: kvo_signal, name='KVO_signal')
        
        # ATR and Volume MA
        self.I(lambda: talib.ATR(self.data.High, self.data.Low, self.data.Close, 14), name='ATR')
        self.I(lambda: talib.SMA(self.data.Volume, 20), name='Volume_MA')
        
        # Trend Filter (200 SMA)
        self.I(lambda: talib.SMA(self.data.Close, 200), name='SMA200')
        
        # Swing Highs/Lows
        self.I(lambda: talib.MAX(self.data.High, 20), name='Swing_high')
        self.I(lambda: talib.MIN(self.data.Low, 20), name='Swing_low')

    def next(self):
        # 🌙 MOON DEV TRADING LOGIC
        if self.consecutive_losses >= 3:
            print("🌧️ 3 Consecutive Losses - Trading Halted!")
            return

        price = self.data.Close[-1]
        upper_band = self.data.BB_upper[-1]
        lower_band = self.data.BB_lower[-1]
        
        # 🚨 Entry Conditions
        if not self.position:
            # Volatility Squeeze
            if self.data.BB_width_pct[-1] < 20:
                # Bullish Setup
                if (price > upper_band and 
                    self.data.Volume[-1] > self.data.Volume_MA[-1] and
                    self.data.Close[-1] > self.data.SMA200[-1]):
                    
                    # Check KVO Divergence
                    if (self.data.Low[-1] < self.data.Low[-2] and
                        self.data.KVO[-1] > self.data.KVO[-2]):
                        
                        # Risk Management
                        stop_loss = min(self.data.Swing_low[-1], self.data.BB_middle[-1])
                        risk_amount = self.equity * self.risk_per_trade
                        risk_per_share = price - stop_loss
                        
                        if risk_per_share > 0:
                            size = int(round(risk_amount / risk_per_share))
                            self.buy(size=size, sl=stop_loss, 
                                   tp=price + 1.5*self.data.ATR[-1])
                            print(f"🌕 MOON DEV LONG ENTRY 🌕 | Size: {size} | SL: {stop_loss:.2f}")

                # Bearish Setup