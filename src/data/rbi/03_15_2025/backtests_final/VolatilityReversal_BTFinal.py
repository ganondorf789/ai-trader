Here's the complete fixed code with all necessary corrections and Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's Volatility Reversal Backtest 🌙
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# 🚀 Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['DateTime'] = pd.to_datetime(data['datetime'])
data = data.set_index('DateTime')

class VolatilityReversal(Strategy):
    # 🌌 Strategy Parameters
    rsi_period = 14
    atr_period = 14
    swing_window = 20
    volume_spike_multiplier = 1.5
    risk_per_trade = 0.01  # 1% of equity per trade
    tp_multiplier = 2
    sl_multiplier = 1
    
    def init(self):
        # ✨ Indicator Calculation
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 5)
        
    def next(self):
        # 🌙 Moon Dev's Core Logic
        if len(self.data) < self.swing_window + 2:
            return
        
        current_close = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]
        current_volume = self.data.Volume[-1]
        
        # 🎯 Entry Conditions
        if not self.position:
            # Bullish Reversal Detection 🌈
            bull_div = (self.data.Low[-1] < self.swing_low[-2] and 
                       current_rsi > self.rsi[-2])
            volume_spike = current_volume > self.volume_ma[-1] * self.volume_spike_multiplier
            atr_filter = current_atr > np.mean(self.atr[-self.atr_period:])
            
            if bull_div and volume_spike and atr_filter:
                # 🧮 Risk Management Calculations
                sl_price = self.swing_low[-2]
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = current_close - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_price = current_close + self.tp_multiplier * current_atr
                    
                    # 🚀 Execute Long Entry
                    self.buy(size=position_size, 
                            sl=sl_price,
                            tp=tp_price)
                    print(f"🌙✨ BULLISH REVERSAL! Long {position_size} @ {current_close:.2f}")
                    print(f"   SL: {sl_price:.2f} | TP: {tp_price:.2f} | RSI: {current_rsi:.2f}")

            # Bearish Reversal Detection 🌧️
            bear_div = (self.data.High[-1] > self.swing_high[-2] and 
                       current_rsi < self.rsi[-2])
            
            if bear_div and volume_spike and atr_filter:
                # 🧮 Risk Management Calculations
                sl_price = self.swing_high[-2]
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = sl_price - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_price = current_close - self.tp_multiplier * current_atr
                    
                    # 🚀 Execute Short Entry
                    self.sell(size=position_size,
                             sl=sl_price,
                             tp=tp_price)
                    print(f"🌙🌧️ BEARISH REVERSAL! Short {position_size} @ {