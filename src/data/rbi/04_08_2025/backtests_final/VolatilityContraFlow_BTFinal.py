Here's the debugged version of your VolatilityContraFlow strategy with Moon Dev themed improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match required format
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime and set as index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilityContraFlow(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands (20,2)
        self.upper_band = self.I(self._calc_upper_band)
        self.middle_band = self.I(self._calc_middle_band)
        self.lower_band = self.I(self._calc_lower_band)
        
        # Bandwidth calculation
        self.bandwidth = self.I(self._calc_bandwidth)
        
        # Bandwidth statistics
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, 20)
        self.bandwidth_std = self.I(talib.STDDEV, self.bandwidth, 20, 1)
        
        # Volume oscillator (3 vs 10 EMA)
        self.volume_osc = self.I(self._calc_volume_osc)
        
    def _calc_upper_band(self):
        upper, _, _ = talib.BBANDS(self.data.Close, 20, 2, 2)
        return upper
    
    def _calc_middle_band(self):
        _, middle, _ = talib.BBANDS(self.data.Close, 20, 2, 2)
        return middle
    
    def _calc_lower_band(self):
        _, _, lower = talib.BBANDS(self.data.Close, 20, 2, 2)
        return lower
    
    def _calc_bandwidth(self):
        return (self.upper_band - self.lower_band) / self.middle_band
    
    def _calc_volume_osc(self):
        vol_ema3 = talib.EMA(self.data.Volume, 3)
        vol_ema10 = talib.EMA(self.data.Volume, 10)
        return (vol_ema3 - vol_ema10) / vol_ema10 * 100
    
    def next(self):
        # Moon Dev debug prints
        if len(self.data) % 50 == 0:
            print(f"🌙✨ Moon Dev Pulse: Bar {len(self.data)} | Close: {self.data.Close[-1]:.2f} | Bandwidth: {self.bandwidth[-1]:.4f}")
        
        if not self.position:
            if len(self.bandwidth) < 20 or len(self.volume_osc) < 4:
                return
                
            # Entry conditions
            bw = self.bandwidth[-1]
            bw_ma = self.bandwidth_ma[-1]
            bw_std = self.bandwidth_std[-1]
            
            # Volatility expansion check
            if bw > bw_ma + 2 * bw_std:
                # Volume decline confirmation
                if len(self.volume_osc) >= 2 and self.volume_osc[-1] < self.volume_osc[-2]:
                    # Price near lower band (long setup)
                    if self.data.Close[-1] <= self.lower_band[-1] * 1.01:
                        # Calculate position size
                        stop_loss = self.lower_band[-1] * 0.995
                        take_profit = self.middle_band[-1]
                        risk_amount = self.equity * self.risk_pct
                        position_size = round(risk_amount / (self.data.Close[-1] - stop_loss))  # Fixed: rounded to whole units
                        
                        if position_size > 0:  # Ensure valid position size
                            print(f"🌙 Moon Dev Long Signal! | Size: {position_size} units | Entry: {self.data.Close[-1]:.2f} | SL: {stop_loss:.2f} | TP: {