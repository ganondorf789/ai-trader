```python
# 🌙 Moon Dev's Volumetric Band Squeeze Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data Preparation 🌐
def load_data(path):
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    # Clean columns
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

class VolumetricBandSqueeze(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    swing_period = 5
    
    def init(self):
        # 🌗 Indicator Calculation Phase
        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                   timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # Band Width Calculation
        self.band_width = (self.upper - self.lower) / self.middle
        self.band_width_sma = self.I(talib.SMA, self.band_width, 20)
        
        # OBV System
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.obv_ema = self.I(talib.EMA, self.obv, 21)
        
        # Volume Filter
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Swing Detection
        self.price_lows = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.obv_highs = self.I(talib.MAX, self.obv, self.swing_period)
        self.price_highs = self.I(talib.MAX, self.data.High, self.swing_period)
        self.obv_lows = self.I(talib.MIN, self.obv, self.swing_period)
        
    def next(self):
        # 🌑 Current Market Conditions
        contraction = self.band_width[-1] < self.band_width_sma[-1]
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1]
        
        # 🛑 Exit Conditions
        if self.position:
            # Emergency OBV Exit
            if self.position.is_long and crossover(self.obv_ema, self.obv):
                print(f"🌙✨ EMERGENCY EXIT LONG @ {self.data.Close[-1]:.2f}")
                self.position.close()
            elif self.position.is_short and crossover(self.obv, self.obv_ema):
                print(f"🌙✨ EMERGENCY EXIT SHORT @ {self.data.Close[-1]:.2f}")
                self.position.close()
            
            # Band Touch Exit
            if self.position.is_long and self.data.Low[-1] <= self.lower[-1]:
                print(f"🌕✅ LONG EXIT @ {self.lower[-1]:.2f}")
                self.position.close()
            elif self.position.is_short and self.data.High[-1] >= self.upper[-1]:
                print(f"🌑✅ SHORT EXIT @ {self.upper[-1]:.2f}")
                self.position.close()
        else:
            # 🚀 Entry Signals
            # Bullish Divergence
            bull_div = (self.price_lows[-1] > self.price_lows[-2] and 
                       self.obv_highs[-1] > self.obv_highs[-2])
            
            # Bearish Divergence
            bear_div = (self.price_highs[-1] < self.price_highs[-2] and 
                       self.obv_lows[-1] < self.obv_lows[-2])
            
            # Long Entry
            if (contraction and bull_div and
                self.data.Close[-1] > self.upper[-1] and
                self.obv[-1] > self.obv_ema[-1] and
                volume_ok):
                
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Open[-1]
                stop_loss = self.lower[-1]
                risk_per_share = entry_price - stop_loss
                
                if risk_per_