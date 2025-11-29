I'll fix the code by removing all `backtesting.lib` imports and replacing any related functions with proper alternatives. Here's the corrected version with Moon Dev enhancements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# ========== MOON DEV DATA PREPARATION ========== 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
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

class MomentumSqueeze(Strategy):
    squeeze_threshold = 0.05  # Band width threshold for squeeze
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # ========== MOON DEV INDICATORS ========== 📈
        # Bollinger Bands
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # Band Width calculation
        self.band_width = self.I(
            lambda u, m, l: (u - l)/m, 
            self.upper_band, self.middle_band, self.lower_band,
            name='BandWidth'
        )
        
        # OBV and its EMA
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.ema_obv = self.I(talib.EMA, self.obv, timeperiod=20)
        
        # Swing Low using MIN
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("🌙✨ Moon Dev Indicators Initialized! Ready for launch! 🚀")

    def next(self):
        # ========== MOON DEV TRADE LOGIC ========== ⚡
        current_bar = len(self.data) - 1
        
        # Entry Conditions
        if not self.position:
            if current_bar < 20:  # Ensure enough data
                return
                
            # Check previous bar conditions
            prev_bar = current_bar - 1
            
            # 1. Bollinger Squeeze
            squeeze = self.band_width[prev_bar] < self.squeeze_threshold
            
            # 2. OBV Cross Above EMA (replaced crossover with direct comparison)
            obv_cross = (self.obv[prev_bar-1] < self.ema_obv[prev_bar-1]) and \
                        (self.obv[prev_bar] > self.ema_obv[prev_bar])
            
            # 3. Price Breakout
            price_breakout = self.data.Close[prev_bar] > self.upper_band[prev_bar]
            
            if squeeze and obv_cross and price_breakout:
                # Moon Dev Risk Management 🌙
                entry_price = self.data.Open[current_bar]
                stop_loss = min(self.swing_low[prev_bar], self.middle_band[prev_bar])
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * self.risk_per_trade) / risk_per_share))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🌙🚀 LIFTOFF! Long {position_size} units at {entry_price}")
                        print(f"    🛡️ Stop Loss: {stop_loss} ({round((entry_price-stop_loss)/entry_price*100,2)}% risk)")

        # Exit Conditions
        else:
            # Check OBV Cross Below EMA (replaced crossover with direct comparison)
            if current_bar < 1:
                return
                
            obv_cross_below = (self.obv[current_bar-2] > self.ema_obv[current_bar-2]) and \
                             (self.obv[current_bar-1] < self.ema_obv[current_bar-1])
            
            if obv_cross