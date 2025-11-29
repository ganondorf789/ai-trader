```python
import numpy as np
import pandas as pd
from backtesting import Strategy, Backtest
import talib

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class BandSurgeBreakout(Strategy):
    slope_threshold = 0.5  # Slope parameter for Keltner Channel
    
    def init(self):
        # 🌙✨ Indicator initialization with TA-Lib
        # Bollinger Bands
        self.upper_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.middle_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.lower_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # Keltner Channel
        self.middle_kc = self.I(talib.EMA, self.data.Close, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.upper_kc = self.I(lambda ema, atr: ema + 1.5 * atr, self.middle_kc, self.atr)
        self.lower_kc = self.I(lambda ema, atr: ema - 1.5 * atr, self.middle_kc, self.atr)
        
        # Bollinger Width calculations
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.upper_bb, self.lower_bb, self.middle_bb)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, 20)
        
        # Keltner Slope calculation
        self.kc_slope = self.I(talib.LINEARREG_SLOPE, self.middle_kc, 5)
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        print("🌙✨ Lunar indicators activated! Ready for launch! 🚀")

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 20:
            return
            
        # Previous period values
        prev_idx = -2  # Previous completed bar
        close = self.data.Close[prev_idx]
        volume = self.data.Volume[prev_idx]
        
        # 🌙✨ Entry conditions check
        bb_contraction = self.bb_width[prev_idx] < self.bb_width_ma[prev_idx]
        volume_surge = volume > 1.5 * self.volume_ma[prev_idx]
        kc_slope = self.kc_slope[prev_idx]
        
        # Long conditions
        if (bb_contraction and volume_surge and 
            close > self.upper_bb[prev_idx] and 
            kc_slope > self.slope_threshold):
            
            # 🌙 Risk management calculations
            entry_price = self.data.Open[-1]
            stop_price = self.middle_bb[prev_idx]
            bb_range = self.upper_bb[prev_idx] - self.lower_bb[prev_idx]
            take_profit = entry_price + 1.5 * bb_range
            
            risk_per_share = entry_price - stop_price
            if risk_per_share <= 0:
                print("🌙⚠️ Abort launch! Negative risk detected!")
                return
                
            position_size = int(round((self.equity * 0.01) / risk_per_share))
            
            if position_size > 0:
                print(f"🌙🚀 LIFT OFF! Long {position_size} @ {entry_price:.2f}")
                self.buy(size=position_size, sl=stop_price