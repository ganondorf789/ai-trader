Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

```python
from backtesting import Backtest, Strategy
import talib
import pandas as pd
import numpy as np

# Load and preprocess data 🌙
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

class VolumetricBreakout(Strategy):
    def init(self):
        # Moon Dev Indicators Setup ✨
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_MA_20')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI_14')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_14')
        
        # Dynamic Bollinger Bands with ATR Width 🌗
        self.upper_band = self.I(
            lambda c,h,l: talib.SMA(c, timeperiod=20) + 2.5*talib.ATR(h,l,c, timeperiod=14),
            self.data.Close, self.data.High, self.data.Low,
            name='UPPER_BAND'
        )
        self.lower_band = self.I(
            lambda c,h,l: talib.SMA(c, timeperiod=20) - 2.5*talib.ATR(h,l,c, timeperiod=14),
            self.data.Close, self.data.High, self.data.Low,
            name='LOWER_BAND'
        )

    def next(self):
        # Moon Dev Safety Check 🌙
        if len(self.data.Close) < 20 or len(self.rsi) < 2:
            return

        # Current Market Conditions 🌌
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.volume_ma[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr = self.atr[-1]

        # Moon Dev Debug Feed 🚨
        print(f"🌙 [CANDLE] Price: {price:.2f} | Vol: {vol:.2f}/{vol_ma:.2f} | RSI: {rsi_now:.1f}")

        # Entry Constellation Alignment 🌠
        if not self.position:
            # Long Entry Protocol 🚀
            if price > upper and vol > 1.5*vol_ma:
                risk = 0.01 * self.equity
                sl = price - 1.5*atr
                size = int(round(risk/(price - sl)))
                print(f"🚀🌕 [LONG] Breakout at {price:.2f} | Size: {size}")
                self.buy(size=size, sl=sl)
            
            # Short Entry Protocol 🌑
            elif price < lower and vol > 1.5*vol_ma:
                risk = 0.01 * self.equity
                sl = price + 1.5*atr
                size = int(round(risk/(sl - price)))
                print(f"🌑🌘 [SHORT] Breakdown at {price:.2f} | Size: {size}")
                self.sell(size=size, sl=sl)

        # Exit Protocol Cluster 🌟
        else:
            # Long Exit - RSI Reversion ✨
            if self.position.is_long and rsi_prev >= 30 and rsi_now < 30:
                print(f"🌙🌟 [CLOSE LONG] RSI Drop {rsi_prev:.1f}→{rsi_now:.1f}")
                self.position.close()
            
            # Short Exit - RSI Reversion ✨
            elif self.position.is_short and rsi_prev <= 70 and rsi_now > 70:
                print(f"🌙🌟 [CLOSE SHORT] RSI Rise {