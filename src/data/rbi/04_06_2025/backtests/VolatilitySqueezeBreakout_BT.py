```python
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

class VolatilitySqueezeBreakout(Strategy):
    def init(self):
        # 🌙 Core Indicators
        self.sma = self.I(talib.SMA, self.data.Close, 20, name='SMA 20')
        self.std = self.I(talib.STDDEV, self.data.Close, 20, name='STD 20')
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing Low')
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        
        # 🌌 Historical tracking
        self.bandwidth_history = []
        self.obv_z_history = []

    def next(self):
        # ✨ Wait for sufficient data
        if len(self.data) < 20:
            return

        # 🚀 Calculate current volatility metrics
        current_sma = self.sma[-1]
        current_std = self.std[-1]
        upper_bb = current_sma + 2*current_std
        lower_bb = current_sma - 2*current_std
        bandwidth = (upper_bb - lower_bb)/current_sma
        
        # 🌙 Bandwidth percentile calculation
        if len(self.bandwidth_history) > 0:
            percentile = 100 * sum(b < bandwidth for b in self.bandwidth_history)/len(self.bandwidth_history)
        else:
            percentile = 0
        self.bandwidth_history.append(bandwidth)

        # 🌊 Volume oscillator Z-score
        current_obv = self.obv[-1]
        obv_window = self.obv[-20:]
        if len(obv_window) >= 20:
            z_score = (current_obv - np.mean(obv_window))/np.std(obv_window) if np.std(obv_window) != 0 else 0
        else:
            z_score = 0

        # 🚨 Entry Conditions
        if not self.position:
            # 🌙 Volatility squeeze + volume spike
            if (percentile < 10) and (abs(z_score) > 2):
                # 🚀 Long Entry
                if self.data.Close[-1] > upper_bb:
                    risk_amount = self.equity * 0.01
                    stop_loss = self.swing_low[-1]
                    risk_per_share = self.data.Close[-1] - stop_loss
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount/risk_per_share))
                        if position_size > 0:
                            take_profit = self.data.Close[-1] + 2*risk_per_share
                            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                            print(f"🚀🌙 LONG: {self.data.Close[-1]} | Size: {position_size} | SL: {stop_loss} | TP: {take_profit}")
                
                # 🌑 Short Entry
                elif self.data.Close[-1] < lower_bb:
                    risk_amount = self.equity * 0.01
                    stop_loss = self.swing_high[-1]
                    risk_per_share = stop_loss - self.data.Close[-1]
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount/risk_per_share))
                        if position_size > 0:
                            take_profit = self.data.Close[-1] - 2*risk_per_share
                            self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                            print(f"🌑🌙 SHORT: {self.data.Close[-1]} | Size: {position_size} | SL: {stop_loss} | TP: {take_profit}")

# 🌙 Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

#