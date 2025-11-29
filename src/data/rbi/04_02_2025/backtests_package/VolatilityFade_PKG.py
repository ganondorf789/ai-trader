Here's the fixed code with all indicator packages properly implemented and Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's VolatilityFade Backtest Script 🚀✨

import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# =====================
# DATA PREPARATION 🌌
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert to datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# STRATEGY CLASS 🌗
# =====================
class VolatilityFade(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌗 Indicator Calculations (Moon Dev Certified Indicators)
        print("🌙 Initializing Moon Dev's Volatility Fade Indicators...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=96, name='ATR_1D')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=1920, name='Volume_SMA_20D')
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, anchor='D', name='VWAP')
        
        # 🌑 Trade tracking
        self.stop_loss = None
        self.take_profit = None
        print("✨ Moon Dev Indicators Successfully Initialized! 🚀")

    def next(self):
        # 🌓 Skip if not enough data
        if len(self.data) < 1920:
            print("🌑 Waiting for sufficient data (1920 candles required)...")
            return

        current_idx = len(self.data) - 1
        prev_close = self.data.Close[-2] if current_idx > 0 else None
        current_atr = self.atr[-1]
        volume_avg = self.volume_sma[-1]

        # 🌕 Entry Logic
        if not self.position and prev_close:
            # Long entry condition: 3x ATR drop + volume surge
            if (self.data.Low[-1] <= (prev_close - 3*current_atr)) and (self.data.Volume[-1] > 1.5*volume_avg):
                print("🌖 Moon Dev Long Signal Detected! (3x ATR Drop + Volume Surge)")
                self.calculate_position_size('long', current_atr)
            
            # Short entry condition: 3x ATR spike + volume surge
            elif (self.data.High[-1] >= (prev_close + 3*current_atr)) and (self.data.Volume[-1] > 1.5*volume_avg):
                print("🌒 Moon Dev Short Signal Detected! (3x ATR Spike + Volume Surge)")
                self.calculate_position_size('short', current_atr)

        # 🌑 Exit Logic
        if self.position:
            # Stop loss check
            if (self.position.is_long and self.data.Low[-1] <= self.stop_loss) or \
               (self.position.is_short and self.data.High[-1] >= self.stop_loss):
                self.position.close()
                print(f"🚨 Moon Dev Stop Loss Activated! ({'Long' if self.position.is_long else 'Short'}) 🌧️")

            # Take profit check
            elif (self.position.is_long and self.data.High[-1] >= self.take_profit) or \
                 (self.position.is_short and self.data.Low[-1] <= self.take_profit):
                self.position.close()
                print(f"🎯 Moon Dev Profit Target Reached! ({'Long' if self.position.is_long else 'Short'}) 🚀")

            # EOD close check
            elif self.data.index[-1].time().hour == 23 and self.data.index[-1].minute == 45:
                self.position.close()
                print(f"🌙 Moon Dev End-of-Day Close! 🕒")

    def calculate_position_size(self, direction, at