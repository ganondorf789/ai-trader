Here's the fixed code with all `backtesting.lib` imports and functions removed, and proper Moon Dev themed debug prints added:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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

class VolatilityAnchor(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    atr_period = 14
    swing_window = 20
    atr_ma_period = 20
    sl_multiplier = 1.5
    tp_multiplier = 2

    def init(self):
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, self.atr_ma_period)
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        # Candlestick patterns
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        # Moon Dev debug prints 🌙
        print(f"\n🌙 MOON DEV UPDATE 🌙\nPrice: {current_close:.2f} | ATR: {current_atr:.2f}\nSwing High: {swing_high:.2f} | Swing Low: {swing_low:.2f}")

        if self.position:
            return

        # Breakout logic 🚀
        if current_close > swing_high and current_atr > atr_ma:
            sl = current_close - self.sl_multiplier * current_atr
            risk = current_close - sl
            position_size = int(round((self.risk_pct * self.equity) / risk))
            if position_size > 0:
                print(f"🚀 BREAKOUT LONG! Size: {position_size}")
                self.buy(size=position_size, sl=sl, tp=current_close + self.tp_multiplier * current_atr)

        elif current_close < swing_low and current_atr > atr_ma:
            sl = current_close + self.sl_multiplier * current_atr
            risk = sl - current_close
            position_size = int(round((self.risk_pct * self.equity) / risk))
            if position_size > 0:
                print(f"🚀 BREAKOUT SHORT! Size: {position_size}")
                self.sell(size=position_size, sl=sl, tp=current_close - self.tp_multiplier * current_atr)

        # Reversal logic 🔄
        bullish_rev = (self.engulfing[-1] > 0 or self.hammer[-1] > 0) and \
                    current_close > swing_low and current_atr < atr_ma
                    
        bearish_rev = (self.engulfing[-1] < 0 or self.hammer[-1] < 0) and \
                    current_close < swing_high and current_atr < atr_ma

        if bullish_rev:
            sl = swing_low - 0.5 * current_atr
            risk = current_close - sl
            position_size = int(round((self.risk_pct * self.equity) / risk))
            if