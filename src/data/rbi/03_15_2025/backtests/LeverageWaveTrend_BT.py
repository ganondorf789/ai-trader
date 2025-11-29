Here's the complete backtesting implementation for the LeverageWaveTrend strategy:

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib as ta
import numpy as np

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LeverageWaveTrend(Strategy):
    risk_pct = 0.02  # 2% risk per trade
    atr_period = 14
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    ewo_short = 5
    ewo_long = 34
    max_leverage = 3

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Core indicators
        self.macd, self.macd_signal, _ = self.I(ta.MACD, close, 
                                               self.macd_fast, self.macd_slow, self.macd_signal)
        self.rsi = self.I(ta.RSI, close, self.rsi_period)
        self.atr = self.I(ta.ATR, high, low, close, self.atr_period)
        
        # Elliott Wave Oscillator
        sma5 = self.I(ta.SMA, close, self.ewo_short)
        sma34 = self.I(ta.SMA, close, self.ewo_long)
        self.ewo = self.I(lambda s5, s34: s5 - s34, sma5, sma34)
        
        # Swing points detection
        self.swing_high = self.I(ta.MAX, high, 20)
        self.swing_low = self.I(ta.MIN, low, 20)
        
        self.trade_counter = 0

    def next(self):
        if self.position:
            self.update_trailing_stop()
            return

        if self.check_entry_conditions():
            self.execute_trade()

    def check_entry_conditions(self):
        # Basic trend filter
        price_above_swing = self.data.Close[-1] > self.swing_high[-2]
        ewo_bullish = self.ewo[-1] > self.ewo[-2] > self.ewo[-3]
        macd_bullish = self.macd[-1] > self.macd_signal[-1] and self.macd[-1] > 0
        rsi_neutral = 40 <= self.rsi[-1] <= 60
        
        return all([price_above_swing, ewo_bullish, macd_bullish, rsi_neutral])

    def execute_trade(self):
        # Risk calculations
        atr_value = self.atr[-1]
        stop_loss = 1.5 * atr_value
        risk_amount = self.risk_pct * self.equity
        base_size = risk_amount / stop_loss
        
        # Apply dynamic leverage
        leverage = min(self.max_leverage, (60 - abs(50 - self.rsi[-1]))  # RSI-based leverage
        position_size = int(round(base_size * leverage))
        
        if position_size <= 0:
            return

        # Entry execution
        entry_price = self.data.Close[-1]
        sl_price = entry_price - stop_loss
        tp_price = entry_price + (2.618 * stop_loss)  # Fib extension target
        
        self.buy(size=position_size, sl=sl_price, tp=tp_price)
        
        # Moon Dev logging 🌙
        print(f"\n🌙✨ WAVE RIDER DETECTED! ✨🌙")
        print(f"📈 Entry Price: {entry_price:.2f}")
        print(f"🎯 Size: {position_size} contracts (🚀 {leverage}x leverage)")
        print(f"⛑️  SL: {sl_price:.2f} | 🏆 TP: {tp_price:.2f}")

    def update_trailing_stop(self):
        if self