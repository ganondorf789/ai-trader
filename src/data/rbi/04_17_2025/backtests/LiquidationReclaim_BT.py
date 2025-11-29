Here's the implementation of the LiquidationReclaim strategy for backtesting.py:

```python
# 🌙✨ Moon Dev Backtest AI Generated Code 🚀

# 1. Imports
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
import numpy as np

# 2. Data Preparation
def prepare_data(path):
    # Load and clean data
    data = pd.read_csv(path, parse_dates=['datetime'])
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Rename columns to proper case
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns=column_map, inplace=True)
    data.set_index('datetime', inplace=True)
    return data

# 3. Strategy Implementation
class LiquidationReclaim(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    
    def init(self):
        # 🌙✨ Calculate Indicators using TA-Lib
        # Liquidation Cluster Detection (Z-Score on Hourly Volume)
        self.hourly_volume = self.I(talib.SUM, self.data.Volume, 4, name='Hourly_Vol')
        self.hourly_mean = self.I(talib.SMA, self.hourly_volume, 80, name='Hourly_Mean')
        self.hourly_std = self.I(talib.STDDEV, self.hourly_volume, 80, name='Hourly_Std')
        
        # VWAP Calculation
        def calc_vwap(h, l, c, v):
            typical = (h + l + c) / 3
            cumulative_tpv = typical * v
            cumulative_vol = v
            return cumulative_tpv.cumsum() / cumulative_vol.cumsum()
        self.vwap = self.I(calc_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, name='VWAP')
        
        # Bollinger Bands (20-period)
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            lambda close: talib.BBANDS(close, 20, 2, 2),
            self.data.Close,
            name=['Upper_BB', 'Middle_BB', 'Lower_BB']
        )
        
        # ATR for trailing stop
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # Swing Low detection
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        
        print("🌙✨ Strategy Indicators Initialized!")
    
    def next(self):
        # 🌙✨ Entry Logic
        if not self.position:
            # Calculate Z-Score for liquidation clusters
            current_vol = self.hourly_volume[-1]
            mean_vol = self.hourly_mean[-1]
            std_vol = self.hourly_std[-1]
            z_score = (current_vol - mean_vol)/std_vol if std_vol != 0 else 0
            
            # Entry conditions
            liquidation_cond = z_score > 2
            vwap_cond = self.data.Close[-1] > self.vwap[-1]
            volume_cond = (self.data.Volume[-1] > self.data.Volume[-2] and 
                          self.data.Volume[-2] > self.data.Volume[-3])
            
            if liquidation_cond and vwap_cond and volume_cond:
                # Risk management calculations
                stop_loss = min(self.swing_low[-1], self.vwap[-1])
                risk_per_share = self.data.Close[-1] - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = self.risk_per_trade * self.equity
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    # Execute trade
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙✨🚀 MOON DEV LONG ENTRY | Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {stop_loss:.2f}")
        
        # 🌙✨ Exit Logic
        if self.position: