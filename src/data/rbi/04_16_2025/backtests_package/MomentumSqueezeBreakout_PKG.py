I'll fix the code by removing all `backtesting.lib` imports and replacing crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed improvements:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🚀
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# DATA PREPARATION 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class MomentumSqueezeBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    atr_multiplier = 1.5
    cmo_period = 20
    keltner_period = 20
    width_ma_period = 20
    max_bars_held = 5  # 🌗 Moons phases count
    
    def init(self):
        # 🌠 CELESTIAL INDICATORS CALCULATION
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period, name='CMO')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.keltner_period, name='ATR')
        self.ema = self.I(talib.EMA, self.data.Close, self.keltner_period, name='EMA')
        
        # 🌗 Keltner Channel Width calculations
        self.width = self.I(lambda atr: 3 * atr, self.atr, name='Width')
        self.width_ma = self.I(talib.SMA, self.width, self.width_ma_period, name='Width MA')
        
        self.entry_bar = 0  # 🌑 New moon tracker

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # 🌌 CURRENT INDICATOR VALUES
        current_ema = self.ema[-1]
        current_atr = self.atr[-1]
        upper_band = current_ema + self.atr_multiplier * current_atr
        lower_band = current_ema - self.atr_multiplier * current_atr
        current_cmo = self.cmo[-1]
        prev_cmo = self.cmo[-2] if len(self.cmo) > 1 else current_cmo
        
        # 🌠 EXIT CONDITIONS FIRST
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar
            
            if self.position.is_long:
                # 🌕 LONG EXIT CONDITIONS
                exit_condition = (
                    (self.width[-1] > self.width_ma[-1]) or
                    (current_low < lower_band) or
                    (bars_held >= self.max_bars_held)
                )
                
                if exit_condition:
                    self.position.close()
                    print(f"🌙✨ LONG CLOSED | Reason: {'Width Expansion' if self.width[-1] > self.width_ma[-1] else 'Trailing Stop' if current_low < lower_band else 'Time Expiry'} 🚀")
            
            else:  # Short position
                # 🌑 SHORT EXIT CONDITIONS
                exit_condition = (
                    (self.width[-1] > self.width_ma[-1]) or
                    (current_high > upper_band) or
                    (bars_held >= self.max_bars_held))
                
                if exit_condition:
                    self.position.close()
                    print(f"🌙✨ SHORT CLOSED | Reason: {'Width Expansion' if self.width[-1] > self.width_ma[-1] else 'Trailing Stop' if current_high > upper_band else 'Time Expiry'} 🌘")

        # 🌠 ENTRY CONDITIONS
        if not self.position:
            # 🌕 LONG ENTRY
            long_cond = (prev_cmo < 50 and current_cmo > 50) and \  # Manual crossover implementation
                       (self.width[-1] < self.width_ma[-1]) and \
                       (current_close > upper_band)
            
            if long_cond:
                risk_amount = self.equ