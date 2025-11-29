I'll help fix the code and add Moon Dev themed debug prints. Here's the corrected version:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class FractalMomentumArb(Strategy):
    risk_pct = 0.01
    atr_multiplier = 2.0
    rr_ratio = 2
    fractal_period = 5
    volume_spike_multiplier = 1.5
    
    def init(self):
        # Fractal indicators 🌌
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.fractal_period, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.fractal_period, name='SWING LOW')
        
        # Momentum trio ✨
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        self.macd = self.I(talib.MACD, self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9, name='MACD')
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        
        # Risk management tools 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL SMA')

    def next(self):
        # Skip early bars where indicators are NaN 🌙
        if len(self.data.Close) < 20:
            return

        # Current market conditions 📊
        price = self.data.Close[-1]
        vol_spike = self.data.Volume[-1] > self.vol_sma[-1] * self.volume_spike_multiplier
        rsi_above_50 = self.rsi[-1] > 50
        
        # MACD crossover detection without backtesting.lib 🌗
        macd_line = self.macd[0]
        signal_line = self.macd[1]
        macd_cross = (macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1])
        macd_crossunder = (macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1])

        # Fractal breakout signals 🌗
        bullish_break = price > self.swing_high[-1]
        bearish_break = price < self.swing_low[-1]

        # Moon Dev position sizing calculator 🌕
        def calculate_size(entry_price, stop_price):
            risk_amount = self.equity * self.risk_pct
            risk_per_unit = abs(entry_price - stop_price)
            return int(round(risk_amount / risk_per_unit))

        # Long entry constellation 🌠
        if not self.position and bullish_break and vol_spike and rsi_above_50 and macd_cross:
            sl = self.swing_low[-1]
            tp = price + (price - sl) * self.rr_ratio
            size = calculate_size(price, sl)
            if size > 0:
                self.buy(size=size, sl=sl, tp=tp)
                print(f"🌙🚀 BULLISH BREAKOUT! Entry: {price:.2f}, Size: {size}, Moon Target: {tp:.2f}")

        # Short entry constellation 🌌
        elif not self.position and bearish_break and vol_spike and not rsi_above_50 and macd_crossunder:
            sl = self.swing_high[-1]
            tp = price - (sl - price) * self.rr_ratio
            size = calculate_size(price, sl)
            if size > 0:
                self.sell(size=size