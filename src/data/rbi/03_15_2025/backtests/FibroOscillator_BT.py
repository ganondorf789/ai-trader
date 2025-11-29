```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

class FibroOscillator(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Define indicators using TA-Lib wrapped in self.I()
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50, name='SMA50')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI14')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING_LOW')
        
        print("🌙 Initializing FibroOscillator Strategy...")
        print("✨ Loading Moon-powered indicators...")
        print("🚀 Ready for lunar market analysis!\n")

    def next(self):
        # Prevent multiple concurrent positions
        if self.position:
            return

        # Wait for indicators to warm up
        if len(self.data.Close) < 50:
            return

        # Get current values
        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        diff = swing_high - swing_low

        # Calculate Fibonacci levels (prioritize deeper retracements first)
        fib_levels = [
            ('61.8%', swing_high - diff * 0.618),
            ('50%', swing_high - diff * 0.5),
            ('38.2%', swing_high - diff * 0.382)
        ]

        # Trend detection
        trend = 'BULL' if close > self.sma50[-1] else 'BEAR'

        # Moon Dev themed analysis messages
        debug_msg = f"🌙 {self.data.index[-1]} | Price: {close:.2f} | Trend: {trend}"
        debug_msg += f"\n   RSI: {self.rsi[-1]:.1f} | Swing High: {swing_high:.2f} | Swing Low: {swing_low:.2f}"
        print(debug_msg)

        # Long entry condition (bullish retracement)
        if trend == 'BULL':
            for level_name, fib_price in fib_levels:
                if low <= fib_price and close > fib_price:
                    if self.rsi[-2] < 30 and self.rsi[-1] > self.rsi[-2]:
                        self.execute_trade(
                            direction='LONG',
                            entry_price=close,
                            fib_level=fib_price,
                            level_name=level_name,
                            stop_loss=fib_price * 0.995,
                            take_profit=swing_high + diff * 1.618
                        )
                        return

        # Short entry condition (bearish retracement)
        if trend == 'BEAR':
            for level_name, fib_price in fib_levels:
                if high >= fib_price and close < fib_price:
                    if self.rsi[-2] > 70 and self.rsi[-1] < self.rsi[-2]:
                        self.execute_trade(
                            direction='SHORT',
                            entry_price=close,
                            fib_level=fib_price,
                            level_name=level_name,
                            stop_loss=fib_price * 1.005,
                            take_profit=swing_low - diff * 1.618
                        )
                        return

    def execute_trade(self, direction, entry_price, fib_level, level_name, stop_loss, take_profit):
        """Moon-powered trade execution with cosmic risk management"""
        risk_amount = self.risk_per_trade * self.equity
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            print("⚡ Risk per unit zero - aborting trade!")
            return

        position_size = int(round(risk_amount / risk_per_unit))
        
        if position_size <= 0:
            print(f"🌑 Zero size calculated: {position_size} - skipping trade")
            return

        # Execute trade with lunar precision
        if direction == 'LONG':
            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            trade_msg = f"🚀🌕 BULLISH IMPACT! Entry: {entry_price:.2f}"
        else:
            self.sell(size=position_size, sl=stop_loss, tp