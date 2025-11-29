Here's the fixed and complete backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class DivergenceFade(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    atr_period = 14
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    swing_window = 20

    def init(self):
        # Clean data and calculate indicators with TA-Lib 🛠️
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # MACD components ✨
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, close, 
                                                            fastperiod=self.macd_fast,
                                                            slowperiod=self.macd_slow,
                                                            signalperiod=self.macd_signal)
        
        # RSI momentum indicator 🌊
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        
        # Swing points detection 🔍
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window)
        
        # Volatility measurement 🌪️
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

    def next(self):
        # Skip if not enough data 📉
        if len(self.data.Close) < 50 or len(self.macd_hist) < 3:
            return

        current_close = self.data.Close[-1]
        current_macd_hist = self.macd_hist[-1]
        prev_macd_hist = self.macd_hist[-2]
        current_rsi = self.rsi[-1]
        
        # Get swing levels 🏔️
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        fib_38 = swing_high - 0.382*(swing_high - swing_low)
        fib_50 = swing_high - 0.5*(swing_high - swing_low)
        
        # Risk management calculations 🌙
        stop_loss = swing_low - self.atr[-1]
        risk_amount = self.equity * self.risk_percent
        risk_per_share = current_close - stop_loss
        
        # Entry conditions check ✅
        macd_decreasing = current_macd_hist < prev_macd_hist
        rsi_confirmation = current_rsi < 50
        bullish_divergence = (self.data.Low[-1] < self.data.Low[-2]) and (current_macd_hist > prev_macd_hist)

        if not self.position and macd_decreasing and rsi_confirmation and bullish_divergence:
            if risk_per_share <= 0:
                print("🌙 MOON DEV WARNING: Invalid risk per share, skipping trade 🚫")
                return
            
            position_size = int(round(risk_amount / risk_per_share))
            if position_size > 0:
                # Execute trade with proper position sizing 🌙✨
                self.buy(size=position_size, 
                        sl=stop_loss,
                        tp=fib_50,
                        tag="Moon Dev Entry 🌙")
                print(f"🌙✨ MOON DEV SIGNAL: LONG {position_size} units @ {current_close:.2f}")
                print(f"   SL: {stop_loss:.2f} | TP1: {fib_38:.2f} | TP2: {fib_50:.2f} 🚀")

        # Trailing stop logic after hitting 38.2% 🔄
        if self.position and current_close >= fib_38:
            if self.position.sl < current_close:
                new_sl = max(self.position.sl, current_close - self.atr[-1])
                self.position.sl = new_sl
                print(f"🌙 Trailing SL Updated: {new_sl:.2f} ✨")

# Data preparation phase 📊
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data 🧹
data.columns = data.columns.str.strip().str.lower()
data = data