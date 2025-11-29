I'll fix the code while maintaining the original strategy logic. Here are the issues I found and fixed:

1. Fixed incomplete print statement in entry logic
2. Added missing exit logic (SMA crossover)
3. Ensured proper position sizing with whole numbers
4. Added proper stop loss implementation
5. Fixed string formatting in print statements

Here's the complete fixed code with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Map to proper case columns
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# Moon Dev Strategy Implementation 🌙🚀
class VolatileDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiplier = 3  # 3x ATR for stop loss
    
    def init(self):
        # Convert days to 15m periods (96 periods/day)
        self.rsi_period = 3 * 96  # 288
        self.atr_short = 5 * 96   # 480
        self.atr_long = 20 * 96   # 1920
        self.sma_exit = 10 * 96  # 960
        
        # Moon Dev Indicators ✨
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.low_min = self.I(talib.MIN, self.data.Low, timeperiod=self.rsi_period)
        self.atr5 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_short)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_long)
        self.sma10 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_exit)
        
        # Moon Dev State Tracking 🌙
        self.last_low_price = None
        self.last_low_rsi = None

    def next(self):
        # Wait for full indicator window
        if len(self.data) < self.atr_long:
            return
        
        current_low = self.data.Low[-1]
        current_rsi = self.rsi[-1]
        current_atr5 = self.atr5[-1]
        current_atr20 = self.atr20[-1]
        current_close = self.data.Close[-1]
        current_sma10 = self.sma10[-1]
        
        # Moon Dev Divergence Detection 🌗
        bullish_divergence = False
        if current_low == self.low_min[-1]:
            if self.last_low_price is not None:
                if (current_low < self.last_low_price) and (current_rsi > self.last_low_rsi):
                    print(f"🌙✨ BULLISH DIVERGENCE! Price LL: {self.last_low_price:.2f}→{current_low:.2f} | RSI HL: {self.last_low_rsi:.2f}→{current_rsi:.2f}")
                    bullish_divergence = True
                self.last_low_price = current_low
                self.last_low_rsi = current_rsi
            else:
                self.last_low_price = current_low
                self.last_low_rsi = current_rsi
        
        # Moon Dev ATR Volatility Check 🌪️
        atr_condition = current_atr5 > 2 * current_atr20
        if atr_condition:
            print(f"📈 ATR Expansion! {current_atr5:.2f} > 2x{current_atr20:.2f}")
        
        # Moon Dev Entry Logic 🚀
        if not self.position and bullish_divergence and atr_condition:
            # Risk Management Calculation 🔒
            sl_price = current_close - self.atr_multiplier * current_atr5
            risk_per_share = current_close - sl_price
            position_size