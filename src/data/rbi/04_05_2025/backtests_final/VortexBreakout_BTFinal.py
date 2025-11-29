Here's the fixed backtest code with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import pandas_ta as ta
import talib
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to Backtesting.py format
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexBreakout(Strategy):
    vi_period = 14
    ci_period = 14
    atr_period = 14
    volume_lookback = 20
    risk_pct = 0.01
    atr_multiplier = 1.5

    def init(self):
        # Calculate Vortex Indicator 🌪️
        vi_plus, vi_minus = ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.vi_period
        )
        self.vi_plus = self.I(lambda x: x, vi_plus, name='VI+')
        self.vi_minus = self.I(lambda x: x, vi_minus, name='VI-')

        # Choppiness Index 🪓
        ci = ta.chop(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.ci_period
        )
        self.ci = self.I(lambda x: x, ci, name='CI')
        self.max_ci = self.I(talib.MAX, self.ci, timeperiod=14, name='MAX_CI')

        # ATR for volatility 🌡️
        self.atr = self.I(talib.ATR,
            self.data.High, self.data.Low, self.data.Close,
            timeperiod=self.atr_period,
            name='ATR'
        )

        # Volume threshold calculation 📈
        volume_90th = self.data.Volume.rolling(self.volume_lookback).quantile(0.9).shift(1)
        self.volume_90th = self.I(lambda x: x, volume_90th, name='VOL_90TH')

    def next(self):
        # Wait for indicators to warm up ⏳
        if len(self.data) < max(self.ci_period+14, self.volume_lookback+1):
            return

        # Trade management 🌙
        if self.position:
            self.update_trailing_stop()
            self.check_emergency_exit()
        else:
            self.check_entries()

    def check_entries(self):
        # Long entry conditions ✅
        vi_bullish = self.vi_plus[-1] > self.vi_minus[-1]
        ci_breakout = (self.ci[-1] < 38.2) and (self.max_ci[-1] > 61.8)
        volume_surge = self.data.Volume[-1] > self.volume_90th[-1]

        if vi_bullish and ci_breakout and volume_surge:
            # Calculate position size 🧮
            risk_amount = self.equity * self.risk_pct
            stop_loss = self.data.Close[-1] - (self.atr[-1] * self.atr_multiplier)  # Price level
            position_size = int(round(risk_amount / (self.data.Close[-1] - stop_loss)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss)
                print(f"🌙✨ MOONSHOT LONG! ✨ {self.data.Close[-1]} | Size: {position_size} units | SL: {stop_loss:.2f} 🚀")
            else:
                print("🌙 WARNING: Position size too small - skipping trade")

    def update_trailing_stop(self):
        for trade in self.trades:
            if trade.is_long:
                # Update trailing stop 🌕
                peak = max(self.data.High[trade.entry_bar:-1], default=trade.entry_price)
                new_sl = peak - (self.atr