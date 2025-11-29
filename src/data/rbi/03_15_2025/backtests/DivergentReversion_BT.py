```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class DivergentReversion(Strategy):
    def init(self):
        # Clean data columns
        self.data.columns = self.data.columns.str.strip().str.lower()
        self.data.drop(columns=[col for col in self.data.columns if 'unnamed' in col], errors='ignore', inplace=True)
        
        # Calculate indicators using TA-Lib
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        self.minus_di = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, 14)
        self.plus_di = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Swing lows for divergence detection
        self.swing_low_price = self.I(talib.MIN, self.data.Low, 5)
        self.swing_low_rsi = self.I(talib.MIN, self.rsi, 5)
        
        # Risk management tracking
        self.daily_high_equity = self.equity
        self.weekly_high_equity = self.equity
        self.current_date = None
        self.current_week = None

    def next(self):
        # Update date tracking
        current_dt = self.data.index[-1]
        current_date = current_dt.date()
        current_week = current_dt.isocalendar()[1]
        
        # Reset daily/weekly highs
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_high_equity = self.equity
        if current_week != self.current_week:
            self.current_week = current_week
            self.weekly_high_equity = self.equity
            
        # Update equity highs
        self.daily_high_equity = max(self.daily_high_equity, self.equity)
        self.weekly_high_equity = max(self.weekly_high_equity, self.equity)
        
        # Check loss limits 🌙
        daily_loss = (self.daily_high_equity - self.equity)/self.daily_high_equity
        if daily_loss >= 0.03:
            print("🌙💸 Daily loss limit hit! Trading halted for today")
            return
        weekly_loss = (self.weekly_high_equity - self.equity)/self.weekly_high_equity
        if weekly_loss >= 0.05:
            print("🌙📉 Weekly loss limit breached! No trades this week")
            return
            
        # Check existing positions
        if self.position:
            # Check trailing stop
            current_atr = self.atr[-1]
            trailing_stop = self.data.Close[-1] + 0.5*current_atr
            if self.data.High[-1] >= trailing_stop:
                print(f"🌙✨ Trailing stop triggered at {trailing_stop:.2f}")
                self.position.close()
            return
            
        # Entry conditions
        if len(self.rsi) < 5: return
        
        # Bearish divergence check 🌙
        price_low_current = self.swing_low_price[-1]
        price_low_prev = self.swing_low_price[-2] if len(self.swing_low_price) >=2 else np.nan
        rsi_low_current = self.swing_low_rsi[-1]
        rsi_low_prev = self.swing_low_rsi[-2] if len(self.swing_low_rsi) >=2 else np.nan
        
        valid_divergence = (price_low_current < price_low_prev) and (rsi_low_current > rsi_low_prev)
        adx_ok = self.adx[-1] > 25 and self.minus_di[-1] > self.plus_di[-1]
        rsi_ok = self.rsi[-1] < 20
        
        if valid_divergence and adx_ok and rsi_ok:
            # Position sizing 🌙
            risk_amount = self.equity * 0.01
            atr_value = self.atr[-1]
            position_size = int(round(risk_amount