```python
# 🌙 Moon Dev's DynamicCrossfire Backtest Implementation
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

class DynamicCrossfire(Strategy):
    # 🌒 Strategy Configuration
    risk_per_trade = 0.01  # 1% of equity per trade
    stop_loss_pct = 0.02   # 2% initial stop loss
    adx_threshold = 25     # ADX trend strength filter
    rsi_exit_level = 70    # RSI overbought exit threshold
    
    def init(self):
        # 🌗 Indicator Calculation using TA-Lib
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA200')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14, name='ADX14')
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI14')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='SwingLow20')
        
    def next(self):
        # 🌑 Skip initial warmup period
        if len(self.data) < 200:
            return
            
        # 🌓 Get current indicator values
        ema50_now = self.ema50[-1]
        ema50_prev = self.ema50[-2]
        ema200_now = self.ema200[-1]
        ema200_prev = self.ema200[-2]
        adx_now = self.adx[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2] if len(self.rsi) >= 2 else rsi_now
        
        # 🚀 Long Entry Logic
        if not self.position:
            # Golden Cross & Strong Trend confirmation
            if (ema50_now > ema200_now and ema50_prev <= ema200_prev) and adx_now > self.adx_threshold:
                entry_price = self.data.Close[-1]
                swing_low = self.swing_low[-1]
                price_stop = entry_price * (1 - self.stop_loss_pct)
                
                # 🌙 Dynamic Stop Calculation
                stop_price = max(swing_low, price_stop)
                risk_per_unit = entry_price - stop_price
                
                if risk_per_unit > 0:
                    position_size = int(round((self.equity * self.risk_per_trade) / risk_per_unit))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price)
                        print(f"🌕 MOON DEV ALERT: Long Entry at {entry_price:.2f}")
                        print(f"   ░ Size: {position_size}, Stop: {stop_price:.2f}")
                        print(f"   ░ EMAs: 50({ema50_now:.2f}) > 200({ema200_now:.2f})")
                        print(f"   ░ ADX: {adx_now:.2f}, RSI: {rsi_now:.2f}\n")
        
        # 🛑 Exit Management
        else:
            # RSI Exit Signal
            if rsi_now < self.rsi_exit_level and rsi_prev >= self.rsi_exit_level:
                self.position.close()
                print(f"🌑 MOON DEV EXIT: RSI Reversal {rsi_prev:.2f}→{rsi_now:.2f}")
            
            # Death Cross Protection
            if ema50_now < ema200_now and ema50_prev >= ema200_prev:
                self.position.close()
                print(f"💀 DEATH CROSS DETECTED: EMAs 50({ema50_now:.2f}) < 200({ema200_now:.2f})")

# 🌍 Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🧹 Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, in