Here's the fixed code with all `backtesting.lib` imports and functions removed, replaced with proper Moon Dev themed implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLTA DIVERGENCE STRATEGY 🌙

# Import cosmic libraries
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Stellar data preparation
def prepare_data(path):
    # Load celestial data
    data = pd.read_csv(path)
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Map to proper case
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Convert to pandas datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    return data

# Load cosmic data
data = prepare_data('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

class VoltaDivergence(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    atr_period = 14
    swing_window = 20  # For swing high/low detection 🌗
    rsi_period = 14
    ema_short = 50
    ema_long = 200

    def init(self):
        # 🌌 Calculate cosmic indicators
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.ema50 = self.I(talib.EMA, self.data.Close, self.ema_short)
        self.ema200 = self.I(talib.EMA, self.data.Close, self.ema_long)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)

    def next(self):
        # 🚀 Check if indicators are initialized
        if len(self.rsi) < 2 or len(self.ema50) < 2 or len(self.ema200) < 2:
            return

        # Current celestial readings
        price = self.data.Close[-1]
        prev_rsi = self.rsi[-2]
        curr_rsi = self.rsi[-1]
        ema50_now = self.ema50[-1]
        ema50_prev = self.ema50[-2]
        ema200_now = self.ema200[-1]
        ema200_prev = self.ema200[-2]

        # 🌗 Check EMA crossovers
        golden_cross = ema50_prev < ema200_prev and ema50_now > ema200_now
        death_cross = ema50_prev > ema200_prev and ema50_now < ema200_now

        # 🌓 Divergence detection
        bullish_div = (self.data.Low[-1] < self.data.Low[-2]) and (curr_rsi > prev_rsi)
        bearish_div = (self.data.High[-1] > self.data.High[-2]) and (curr_rsi < prev_rsi)

        # 🪐 Risk management calculations
        atr_value = self.atr[-1]
        equity = self.equity

        # 🌑 Long entry constellation
        if not self.position:
            if golden_cross and bullish_div and curr_rsi > 70 and prev_rsi <= 70:
                # Calculate moon trajectory (stop loss)
                sl_price = min(self.swing_low[-1], price - atr_value)
                risk_amount = self.risk_pct * equity
                risk_distance = price - sl_price
                
                if risk_distance <= 0:
                    return  # Avoid black holes
                
                position_size = int(round(risk_amount / risk_distance))
                tp_price = price + risk_distance  # 1:1 reward
                
                self.buy(size=position_size, sl=sl_price, tp=tp_price)
                print(f"🌙🚀 MOON SHOT! Long {position_size} units at {price:.2f} | SL: {sl_price:.2f} | TP: {tp