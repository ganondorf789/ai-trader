Here's the fixed code with all `backtesting.lib` references removed and proper Moon Dev themed debug prints added:

```python
from backtesting import Backtest, Strategy
import talib
import pandas as pd

class NocturnalReversal(Strategy):
    def init(self):
        # 🌙 Data Cleansing Ritual
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        self.data = self.data.drop(columns=[col for col in self.data.columns if 'unnamed' in col])
        self.data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        # 🌌 Cosmic Indicators
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_upper', index=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_lower', index=2)
        
        # ATR & Volume
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_sma = self.I(talib.SMA, self.atr, 20)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)

    def next(self):
        # 🌑 Require at least 3 days of cosmic data
        if len(self.data) < 3:
            return

        # 🕰️ Time Portal Check (20:00-04:00 UTC)
        def is_overnight(t):
            return t.hour >= 20 or t.hour < 4
        
        # Check previous two candles
        prev1_time = self.data.index[-2].time()
        prev2_time = self.data.index[-3].time()
        if not (is_overnight(prev1_time) and is_overnight(prev2_time)):
            return

        # 🕯️ Hanging Man Pattern Detection
        prev1_bearish = self.data.Close[-2] < self.data.Open[-2]
        prev2_bearish = self.data.Close[-3] < self.data.Open[-3]
        if not (prev1_bearish and prev2_bearish):
            return

        # 📏 Size Amplification Check
        prev1_range = self.data.High[-2] - self.data.Low[-2]
        prev2_range = self.data.High[-3] - self.data.Low[-3]
        if prev1_range < 1.1 * prev2_range:
            return

        # 🌗 Range-Bound Confirmation
        current_close = self.data.Close[-2]
        if current_close > self.bb_upper[-2] or current_close < self.bb_lower[-2]:
            return

        # 🌊 Volatility Filter
        if self.atr[-2] >= self.atr_sma[-2]:
            return

        # 📈 Volume Validation
        if self.data.Volume[-2] < self.volume_sma[-2]:
            return

        # 🚀 Launch Sequence Initiation
        entry_price = self.data.Open[-1]
        stop_loss = self.data.Low[-2]
        take_profit = self.bb_upper[-1]

        # 💰 Risk Management Constellation
        risk_pct = 0.01
        risk_amount = self.equity * risk_pct
        risk_distance = entry_price - stop_loss
        
        # 🛑 Abort Conditions
        if risk_distance <= 0:
            return
        if risk_distance/entry_price > 0.02:
            print(f"🌙⚠️ Black Hole Alert! SL too wide: {risk_distance/entry_price*100:.1f}%")
            return

        position_size = int(round(risk_amount / risk_distance))
        if position_size <= 0:
            return

        # 🌕 Lunar Trade Execution
        if not self.position:
            self.buy(size=position_size)
            self.sell(size=position_size, exectype=bt.Order.Stop, price=stop_loss)
            self.sell(size=position_size, exectype=bt.Order.Limit, price=take_profit)
            print(f"🌙✨🚀 MOONSHOT ACTIVATED! Long {position_size} @ {entry_price}")
            print(f"   🛡️SL: {stop_loss} | 🎯TP: {take_profit}")

        # 💥 Emergency Escape Hatch
        if