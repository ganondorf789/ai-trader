Here's the complete backtesting implementation for the VolatilityBreakout strategy:

```python
# 🌙 Moon Dev's Volatility Breakout Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# ========================
# DATA PREPARATION ✨
# ========================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Format columns for backtesting.py
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ========================
# STRATEGY IMPLEMENTATION 🚀
# ========================
class VolatilityBreakout(Strategy):
    atr_period = 14
    rsi_period = 14
    lookback = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Volatility Indicators
        self.recent_high = self.I(talib.MAX, self.data.High, timeperiod=self.lookback)
        self.recent_low = self.I(talib.MIN, self.data.Low, timeperiod=self.lookback)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # 🌙 Momentum Indicator
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        
    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        high_band = self.recent_high[-1]
        low_band = self.recent_low[-1]
        
        # 🌙 Moon Dev Debugging
        print(f"🌙 [BAR] Close: {price:.2f} | RSI: {rsi:.1f} | ATR: {atr:.2f} | High Band: {high_band:.2f} | Low Band: {low_band:.2f}")
        
        if not self.position:
            # Long Entry 🚀
            if price > high_band and rsi > 50:
                risk_amount = self.risk_pct * self.equity
                if atr <= 0: return
                units = int(round(risk_amount / atr))
                
                if units > 0:
                    sl = price - atr
                    tp = price + 1.5 * atr
                    print(f"🚀 [LONG] Size: {units} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                    self.buy(size=units, sl=sl, tp=tp)
            
            # Short Entry 🌑
            elif price < low_band and rsi < 50:
                risk_amount = self.risk_pct * self.equity
                if atr <= 0: return
                units = int(round(risk_amount / atr))
                
                if units > 0:
                    sl = price + atr
                    tp = price - 1.5 * atr
                    print(f"🌑 [SHORT] Size: {units} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                    self.sell(size=units, sl=sl, tp=tp)
        else:
            # Exit on RSI Reversal 🛑
            if self.position.is_long and rsi < 50:
                print(f"🛑 [EXIT LONG] RSI Reversal at {rsi:.1f}")
                self.position.close()
            elif self.position.is_short and rsi > 50:
                print(f"🛑 [EXIT SHORT] RSI Reversal at {rsi:.1f}")
                self.position.close()

# ========================
# BACKTEST EXECUTION ✨
# ========================
bt = Backtest(data, VolatilityBreakout, cash=1_000_000, trade_on_close=True)
stats = bt.run()

#