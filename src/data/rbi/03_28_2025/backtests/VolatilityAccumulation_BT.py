```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VOLATILITY ACCUMULATION STRATEGY 🚀

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta

class VolatilityAccumulation(Strategy):
    def init(self):
        # 🌙 CALCULATE REQUIRED INDICATORS USING TA-LIB
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        
        # Volatility measurements (20-day lookback = 20*96 15m bars)
        lookback_3day = 3 * 96
        lookback_20day = 20 * 96
        lookback_50day = 50 * 96
        
        # 3-day price percentage change
        self.roc_3day = self.I(talib.ROC, self.data.Close, lookback_3day)
        
        # Historical volatility (std dev of 3-day returns over 20 days)
        self.std_dev_20 = self.I(talib.STDDEV, self.roc_3day, lookback_20day)
        
        # Volatility filter (50-day SMA of 20-day volatility)
        self.vol_filter = self.I(talib.SMA, self.std_dev_20, lookback_50day)
        
        # Mean reversion level (20-day SMA)
        self.sma_20 = self.I(talib.SMA, self.data.Close, lookback_20day)
        
        print("🌙 MOON DEV INDICATORS INITIALIZED ✨")

    def next(self):
        current_bar = len(self.data) - 1
        
        # 🌙 VOLATILITY FILTER CHECK
        vol_filter_active = self.std_dev_20[-1] > self.vol_filter[-1]
        
        if not self.position:
            # 🌌 ENTRY CONDITIONS
            price_drop_threshold = -2 * self.std_dev_20[-1]
            obv_confirmation = self.obv[-1] > self.obv[-3*96]  # 3-day OBV comparison
            
            if (vol_filter_active and
                self.roc_3day[-1] < price_drop_threshold and
                obv_confirmation):
                
                # 🎯 RISK MANAGEMENT CALCULATIONS
                entry_price = self.data.Close[-1]
                close_3day_ago = self.data.Close[-3*96]
                price_drop = (close_3day_ago - entry_price)
                stop_loss_price = entry_price - 1.5 * price_drop
                risk_per_share = entry_price - stop_loss_price
                
                # 💵 POSITION SIZING WITH MOON DEV SAFETY
                risk_percent = 0.01  # 1% risk
                risk_amount = self.equity * risk_percent
                position_size = int(round(risk_amount / risk_per_share))
                
                # 🚀 ENTRY WITH STOP LOSS
                self.buy(size=position_size, sl=stop_loss_price)
                print(f"🌙 MOON DEV ENTRY ✨ | Price: {entry_price:.2f} | Size: {position_size} | SL: {stop_loss_price:.2f}")
        
        else:
            # 🌕 EXIT CONDITIONS
            exit_primary = self.data.Close[-1] > self.sma_20[-1]
            exit_secondary = self.obv[-1] < self.obv[-2]
            exit_time = current_bar - self.position.entry_bar >= 5*96  # 5-day timeout
            
            if exit_primary or exit_secondary or exit_time:
                self.position.close()
                reason = ("PRIMARY" if exit_primary else 
                         "SECONDARY" if exit_secondary else "TIME")
                print(f"🚀 MOON DEV EXIT ✨ | Price: {self.data.Close[-1]:.2f} | Reason: {reason}")

# 🌙 DATA PREPARATION
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'])

# 🧹 DATA CLEANING
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🗺 COLUMN MAPPING
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close