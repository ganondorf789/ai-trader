```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and prepare data 🌙
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

class SqueezeBreakout(Strategy):
    risk_percent = 0.01  # 🌙 1% risk per trade
    
    def init(self):
        # Keltner Channel Components ✨
        self.ema20 = self.I(talib.EMA, self.data.Close, 20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.upper_band = self.I(lambda ema, atr: ema + 2*atr, self.ema20, self.atr20)
        self.lower_band = self.I(lambda ema, atr: ema - 2*atr, self.ema20, self.atr20)
        
        # Volatility Squeeze Detection 🌙
        self.channel_width = self.I(lambda u, l: u - l, self.upper_band, self.lower_band)
        self.width_low = self.I(talib.MIN, self.channel_width, 20)
        
        # Volume Analysis 📉
        self.vol_sma5 = self.I(talib.SMA, self.data.Volume, 5)
        self.vol_sma20 = self.I(talib.SMA, self.data.Volume, 20)
        
        print("🌙 Moon Dev Strategy Activated! Ready to catch those breakouts! 🚀")

    def next(self):
        price = self.data.Close[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        
        # Moon Dev Debug Prints ✨
        if len(self.data) % 100 == 0:
            print(f"🌙 Bar {len(self.data)} | Price: {price:.2f} | Width: {self.channel_width[-1]:.2f} ✨")
        
        # Entry Logic 🚀
        if not self.position:
            # Volatility Squeeze Condition
            squeeze_active = self.channel_width[-1] <= self.width_low[-1]
            
            # Volume Downtrend Condition
            volume_downtrend = self.vol_sma5[-1] < self.vol_sma20[-1]
            
            if squeeze_active and volume_downtrend:
                # Long Entry 📈
                if price > upper:
                    risk = abs(upper - lower)
                    if risk > 0:
                        size = int(round((self.broker.equity * self.risk_percent) / risk))
                        self.buy(size=size, sl=lower)
                        print(f"🚀🌙 BLASTOFF LONG! Size: {size} | Entry: {price:.2f} | SL: {lower:.2f} ✨")
                
                # Short Entry 📉
                elif price < lower:
                    risk = abs(upper - lower)
                    if risk > 0:
                        size = int(round((self.broker.equity * self.risk_percent) / risk))
                        self.sell(size=size, sl=upper)
                        print(f"🌙💫 SHORT STARDUST! Size: {size} | Entry: {price:.2f} | SL: {upper:.2f} 🚀")

        # Exit Logic ✨
        else:
            if self.position.is_long and price < upper:
                self.position.close()
                print(f"🌙✨ LONG MOONING EXIT: {price:.2f} | Profit: {self.position.pl:.2f} 🚀")
            elif self.position.is_short and price > lower:
                self.position.close()
                print(f"🌙✨ SHORT LANDING EXIT: {price:.2f} | Profit: {self.position.pl:.2f} 📉")

# Launch Backtest 🌙
bt = Backtest(data, Squeeze