import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

class VolumetricSqueeze(Strategy):
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Bollinger Bands
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # Bandwidth calculations
        self.bandwidth = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, timeperiod=20)
        
        # Volume Percentile
        self.volume_pct = self.I(ta.percentile_rank, self.data.Volume, 100)
        
        # Trackers
        self.entry_price = None
        self.entry_bandwidth = None

    def next(self):
        if not self.position:
            # Detect squeeze and volume spike
            if (self.bandwidth[-1] <= self.bandwidth_min[-1]) and (self.volume_pct[-1] > 70):
                # Long condition
                if self.data.Close[-1] > self.bb_upper[-1]:
                    risk_percent = 0.01
                    risk_amount = self.equity * risk_percent
                    entry_price = self.data.Close[-1]
                    stop_loss = entry_price - (self.bb_upper[-1] - self.bb_lower[-1])
                    risk_per_share = entry_price - stop_loss
                    if risk_per_share <= 0:
                        return
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        self.entry_price = entry_price
                        self.entry_bandwidth = self.bandwidth[-1]
                        print(f"🚀 Moon Dev Long Signal: Entry {entry_price}, Size {position_size} ✨")
                
                # Short condition
                elif self.data.Close[-1] < self.bb_lower[-1]:
                    risk_percent = 0.01
                    risk_amount = self.equity * risk_percent
                    entry_price = self.data.Close[-1]
                    stop_loss = entry_price + (self.bb_upper[-1] - self.bb_lower[-1])
                    risk_per_share = stop_loss - entry_price
                    if risk_per_share <= 0:
                        return
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_loss)
                        self.entry_price = entry_price
                        self.entry_bandwidth = self.bandwidth[-1]
                        print(f"🌑 Moon Dev Short Signal: Entry {entry_price}, Size {position_size} ✨")
        else:
            # Check 2x bandwidth exit
            if self.position.is_long:
                exit_price = self.entry_price - 2 * self.entry_bandwidth
                if self.data.Close[-1] < exit_price:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit: Long closed at {self.data.Close[-1]} (2x bandwidth) 🛑")
                    self.entry_price = None
                    self.entry_bandwidth = None
            elif self.position.is_short:
                exit_price = self.entry_price + 2 * self.entry_bandwidth
                if self.data.Close[-1] > exit_price:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit: Short closed at {self.data.Close[-1]} (2x bandwidth) 🛑")
                    self.entry_price = None
                    self.entry_bandwidth = None

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data