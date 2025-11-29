```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VoltaicPulse(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    ema200_period = 200
    ema50_period = 50
    atr_period = 5
    atr_ma_period = 20
    
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema200_period, name='EMA200')
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema50_period, name='EMA50')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR5')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period, name='ATR_MA20')
        
        # Track trade parameters
        self.take_profit_price = None
        print("🌙 VoltaicPulse Strategy Activated! ✨")

    def next(self):
        # Skip early bars where indicators are not calculated
        if len(self.data) < self.ema200_period or np.isnan(self.ema200[-1]):
            return

        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # Moon Dev debug prints for key levels
        if len(self.data) % 100 == 0:
            print(f"🌕 Current EMA200: {self.ema200[-1]:.2f}, EMA50: {self.ema50[-1]:.2f}")
            print(f"📡 ATR5: {self.atr[-1]:.2f} vs MA20: {self.atr_ma[-1]:.2f}")

        if not self.position:
            # Entry conditions check
            trend_filter = current_close > self.ema200[-1]
            atr_cross = self.atr[-1] > self.atr_ma[-1] and self.atr[-2] <= self.atr_ma[-2]

            if trend_filter and atr_cross:
                risk_amount = current_close - self.ema50[-1]
                
                if risk_amount <= 0:
                    print("🌑 Invalid risk calculation - EMA50 above price!")
                    return

                # Position sizing calculation
                equity = self.equity
                position_size = (equity * self.risk_per_trade) / risk_amount
                position_size = int(round(position_size))
                
                if position_size <= 0:
                    print("🌑 Invalid position size calculated!")
                    return

                # Execute entry
                self.buy(size=position_size)
                self.take_profit_price = current_close + 2 * risk_amount
                print(f"\n🚀⚡ VOLTAIC PULSE ENTRY! ⚡🚀"
                      f"\nEntry Price: {current_close:.2f}"
                      f"\nPosition Size: {position_size} units"
                      f"\nStop Level (EMA50): {self.ema50[-1]:.2f}"
                      f"\nTake Profit: {self.take_profit_price:.2f}")
        else:
            # Exit conditions check
            if current_high >= self.take_profit_price:
                self.position.close()
                print(f"\n🎯💎 PROFIT PULSE CAPTURED! 💎🎯"
                      f"\nExit Price: {self.take_profit_price:.2f}")
            elif current_close < self.ema50[-1]:
                self.position.close()
                print(f"\n🛑⚡ VOLTAGE DROP! STOPPED OUT! ⚡"
                      f"\nExit Price: {current_close:.2f}"
                      f"\nEMA50 Level: {self.ema50[-1]:.2f}")

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
                   parse_dates=['datetime'], index_col='datetime')

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open