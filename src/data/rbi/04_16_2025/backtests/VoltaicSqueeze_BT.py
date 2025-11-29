Here's the complete backtesting implementation for the VoltaicSqueeze strategy:

```python
# 🌙 MOON DEV BACKTESTING SUITE - VOLTAIC SQUEEZE STRATEGY 🚀

from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import numpy as np

# ====================== DATA PREPARATION ====================== #
print("🌙 Preparing lunar market data...")
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse and align cosmic data streams
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Configure spacetime coordinates
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
print("✨ Data aligned with cosmic standards! 🪐")

# ====================== CELESTIAL STRATEGY ====================== #
class VoltaicSqueeze(Strategy):
    bb_period = 20
    bb_dev = 2
    kc_period = 20
    kc_dev = 2
    volume_ma_period = 20
    adx_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌌 Cosmic Indicator Initialization
        c = self.data.Close
        h = self.data.High
        l = self.data.Low
        v = self.data.Volume
        
        # Bollinger Bands (20,2)
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            lambda: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev)[0:3],
            name='BBANDS'
        )
        
        # Keltner Channels (EMA20 + 2*ATR20)
        ema = talib.EMA(c, self.kc_period)
        atr = talib.ATR(h, l, c, self.kc_period)
        self.upper_kc = self.I(lambda: ema + self.kc_dev*atr, name='UpperKC')
        self.lower_kc = self.I(lambda: ema - self.kc_dev*atr, name='LowerKC')
        
        # Volume Momentum
        self.volume_ma = self.I(talib.SMA, v, self.volume_ma_period, name='VolumeMA')
        
        # Trend Strength
        self.adx = self.I(talib.ADX, h, l, c, self.adx_period, name='ADX')
        
        print("🛸 Indicators charged with quantum energy! ⚡")

    def next(self):
        # 🪐 Current Cosmic Readings
        price = self.data.Close[-1]
        upper_bb = self.upper_bb[-1]
        lower_bb = self.lower_bb[-1]
        upper_kc = self.upper_kc[-1]
        lower_kc = self.lower_kc[-1]
        volume = self.data.Volume[-1]
        vol_ma = self.volume_ma[-1]
        adx = self.adx[-1]

        # 🌙 Squeeze Detection Protocol
        squeeze_on = (upper_bb < upper_kc) and (lower_bb > lower_kc)
        
        if not self.position:
            # 🚀 Long Entry Sequence
            if all([
                squeeze_on,
                price > upper_bb,
                volume > vol_ma,
                adx > 25
            ]):
                sl = lower_bb
                risk_amount = self.equity * self.risk_pct
                risk_distance = price - sl
                position_size = int(round(risk_amount / risk_distance))
                
                print(f"🌕 BULLISH BREAKOUT! | Price: {price:.2f} | Size: {position_size} | SL: {sl:.2f} 🚀")
                self.buy(size=position_size, sl=sl)
            
            # 🌑 Short Entry Sequence
            elif all([
                squeeze_on,
                price < lower_bb,
                volume > vol_ma,
                adx > 25
            ]):
                sl = upper_bb
                risk_amount