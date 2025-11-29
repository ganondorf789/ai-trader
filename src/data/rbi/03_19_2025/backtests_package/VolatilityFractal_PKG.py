I'll help fix the code by removing the `backtesting.lib` import and ensuring all indicators are properly implemented using TA-Lib. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
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

class VolatilityFractal(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Define indicators with proper TA-Lib calculations
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, 20, 2, 2, 0)[0], self.data.Close)
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, 20, 2, 2, 0)[2], self.data.Close)
        self.cmo = self.I(talib.CMO, self.data.Close, 14)
        self.swing_high = self.I(talib.MAX, self.data.High, 5)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.bandwidth = []

    def next(self):
        # Skip early bars without sufficient data
        if len(self.data.Close) < 50:
            return

        # Calculate current indicators
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_volume = self.data.Volume[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        cmo = self.cmo[-1]
        cmo_prev = self.cmo[-2]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        volume_sma = self.volume_sma[-1]

        # Calculate Bollinger Bandwidth
        current_bandwidth = bb_upper - bb_lower
        self.bandwidth.append(current_bandwidth)
        sma_bandwidth = pd.Series(self.bandwidth).rolling(50).mean().iloc[-1]

        # 1. Entry Conditions 🌙
        squeeze = current_bandwidth < sma_bandwidth * 0.5
        cmo_break = cmo_prev <= 80 and cmo > 80  # Replaced crossover with direct comparison
        price_break = current_close > bb_upper
        volume_conf = current_volume > volume_sma

        # Moon Dev debug prints
        if squeeze: print(f"🌙 SQUEEZE DETECTED! Bandwidth: {current_bandwidth:.2f}")
        if cmo_break: print(f"🚀 CMO ROCKETING! {cmo:.2f} > 80")
        if price_break: print(f"💎 PRICE BREAKOUT! {current_close:.2f} > BB_Upper")
        if volume_conf: print(f"📈 VOLUME SURGE! {current_volume:.2f} > SMA")

        # 2. Execute Entry ✨
        if not self.position and all([squeeze, cmo_break, price_break, volume_conf]):
            # Calculate stop loss and position size
            stop_price = max(bb_lower, swing_low) - 0.001 * current_close
            risk_amount = self.equity * self.risk_percent
            position_size = int(round(risk_amount / (current_close - stop_price)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_price, tp=swing_high)
                print(f"🚀🌙 LONG LAUNCH! Size: {position_size} @ {current_close:.2f}")
                print(f"    🛡️ SL: {stop_price:.2f} | 🎯 TP: {swing_high:.2f}")

        # 3. Exit Conditions 🔄
        if self.position:
            # Take profit at fractal resistance
            if current