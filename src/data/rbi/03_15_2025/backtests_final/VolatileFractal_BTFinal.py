Let me help you fix and complete the code while maintaining the original strategy logic. Here's the corrected and completed version with Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

# Data handling
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatileFractal(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2     # Risk-reward ratio
    atr_window = 14
    atr_multiplier = 1.5
    ema_short = 50
    ema_long = 200
    
    def init(self):
        # Fractal indicators (using pandas_ta)
        fractals = self.I(ta.fractals, self.data.High, self.data.Low)
        self.fractal_high = fractals['FU']
        self.fractal_low = fractals['FD']
        
        # Volatility indicator (ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_window)
        
        # Trend indicators
        self.ema_short = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_short)
        self.ema_long = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_long)
        
        self.entry_atr = None  # Track ATR at entry

    def next(self):
        price = self.data.Close[-1]
        current_atr = self.atr[-1]
        
        # Moon Dev debug prints 🌙
        print(f"🌙 Current Price: {price:.2f} | ATR: {current_atr:.2f} | EMA50: {self.ema_short[-1]:.2f} | EMA200: {self.ema_long[-1]:.2f}")

        if not self.position:
            # Long entry condition
            if self.fractal_low[-1] and (price > self.ema_long[-1]):
                if current_atr > self.atr[-self.atr_window:].mean():
                    risk_amount = self.equity * self.risk_pct
                    sl = self.data.Low[-1] * 0.99
                    risk_per_share = price - sl
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        tp = price + (risk_per_share * self.rr_ratio)
                        self.buy(size=position_size, sl=sl, tp=tp)
                        self.entry_atr = current_atr
                        print(f"🚀 BULLISH BREAKOUT! Long {position_size} shares | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

            # Short entry condition
            elif self.fractal_high[-1] and (price < self.ema_long[-1]):
                if current_atr > self.atr[-self.atr_window:].mean():
                    risk_amount = self.equity * self.risk_pct
                    sl = self.data.High[-1] * 1.01
                    risk_per_share = sl - price
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        tp = price - (risk_per_share * self.rr_ratio)
                        self.sell(size=position_size, sl=sl, tp=tp)
                        self.entry_atr = current_atr
                        print(f"🔻 BEARISH BREAKDOWN! Short {position_size} shares | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

# Run backtest
bt = Backtest(data, VolatileFractal, cash=10000, commission=.002)
stats = bt.run()
print("🌙✨ Backtest Complete! Moon Dev Strategy Results:")
print(stats)