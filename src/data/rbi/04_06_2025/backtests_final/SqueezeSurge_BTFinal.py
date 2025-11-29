I'll fix the code while maintaining the original strategy logic and adding Moon Dev themed debug prints. Here's the complete fixed version:

```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# 🌙 MOON DEV DATA PREPARATION ✨
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names with lunar precision
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns like shooting stars
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Map to required columns with celestial alignment
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['DateTime'] = pd.to_datetime(data['datetime'])
    data.set_index('DateTime', inplace=True)
    print("🌑🌒🌓🌔🌕🌖🌗🌘 MOON DEV DATA LOADED SUCCESSFULLY ✨")
    return data

data = load_data('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

class SqueezeSurge(Strategy):
    risk_pct = 0.01  # 1% risk per trade (moon dust allocation)
    
    def init(self):
        # 🌙 BOLLINGER BAND CALCULATIONS (CELESTIAL BANDS)
        self.upper_band = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # 🌟 VOLUME SURGE INDICATOR (LUNAR TIDES)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # ✨ MACD HISTOGRAM (COSMIC ENERGY FLOW)
        self.macd_hist = self.I(lambda c: talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[2], self.data.Close)
        
        # 🔭 BOLLINGER BANDWIDTH (GRAVITATIONAL SQUEEZE)
        self.bandwidth = self.I(lambda u, l: u - l, self.upper_band, self.lower_band)
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, timeperiod=20)
        
        print("🌌 MOON DEV INDICATORS INITIALIZED - READY FOR LIFTOFF 🚀")

    def next(self):
        if len(self.data.Close) < 20:
            print("🌑 GATHERING MOON DATA - NOT ENOUGH CANDLES YET")
            return
        
        # 🌙 CURRENT VALUES (CELESTIAL SNAPSHOT)
        close = self.data.Close[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        bandwidth = self.bandwidth[-1]
        min_bw = self.min_bandwidth[-1]
        volume = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        hist = self.macd_hist[-1]
        prev_hist = self.macd_hist[-2] if len(self.macd_hist) > 1 else 0
        
        # 🚀 ENTRY CONDITIONS (GRAVITATIONAL ALIGNMENT)
        squeeze = bandwidth <= min_bw
        vol_surge = volume > 1.5 * vol_sma
        
        if not self.position:
            # 🌕 LONG ENTRY (MOONSHOT OPPORTUNITY)
            if all([
                squeeze,
                close > upper,
                vol_surge,
                hist > 0,
                hist > prev_hist  # Bullish momentum crossover replacement
            ]):
                midpoint = (upper + lower) / 2
                risk_per_share = close - midpoint
                if risk_per_share > 0:
                    size = int(round((self.risk_pct * self.equity) / risk_per_share))
                    if size > 0:  # Ensure we don't trade zero units
                        tp = close + 2 * risk_per_share