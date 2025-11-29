Here's the fixed code with all backtesting.lib imports removed and proper replacements implemented:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

# Data preparation
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

class LiquidationBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate Bollinger Bands
        self.upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, plot=False)[0]
        self.middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, plot=False)[1]
        self.lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, plot=False)[2]
        
        # Calculate Bollinger Band Width
        self.band_width = self.I(lambda: (self.upper - self.lower) / self.middle, plot=False)
        self.band_width_low = self.I(talib.MIN, self.band_width, timeperiod=35040, plot=False)  # 1-year low
        
        # Volatility metrics
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, plot=False)
        
        # Synthetic funding rate anomaly (z-score)
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20, plot=False)
        self.std20 = self.I(talib.STDDEV, self.data.Close, timeperiod=20, plot=False)
        self.funding_z = self.I(lambda: (self.data.Close - self.sma20) / self.std20, plot=False)
        
        # Synthetic liquidation clusters (volume spikes + price action)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, plot=False)
        self.vol_spike = self.I(lambda: self.data.Volume > 2 * self.vol_sma, plot=False)
        self.price_change = self.I(lambda: self.data.Close.pct_change(), plot=False)
        
    def next(self):
        if len(self.data) < 35040 or len(self.data.Close) < 20:  # Ensure enough data
            return
            
        current_low = self.band_width[-1]
        yearly_low = self.band_width_low[-1]
        atr = self.atr[-1]
        price = self.data.Close[-1]
        
        # Funding rate anomaly detection
        funding_anomaly = abs(self.funding_z[-1]) > 2
        
        # Liquidation cluster detection
        long_liq = (self.price_change[-1] < -0.01) and self.vol_spike[-1]
        short_liq = (self.price_change[-1] > 0.01) and self.vol_spike[-1]
        
        # Entry conditions
        if current_low < yearly_low and funding_anomaly:
            if not self.position:
                # Determine dominant liquidation direction
                if short_liq:  # Short squeeze scenario
                    stop_loss = self.lower[-1] - 1.5 * atr
                    risk = price - stop_loss
                    position_size = int(round((self.risk_per_trade * self.equity) / risk))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss, tp=price + 2 * atr)
                        print(f"🌙✨ MOON DEV LONG SIGNAL! 🚀 Buying {position_size} units at {price:.2f}")
                
                elif long_liq:  # Long squeeze scenario