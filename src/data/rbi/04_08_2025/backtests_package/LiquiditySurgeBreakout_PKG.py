Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# 🌙 MOON DEV BACKTESTING ENGINE 🌙
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# 🚀 DATA PREPARATION 
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquiditySurgeBreakout(Strategy):
    # 🌌 STRATEGY CONSTELLATIONS
    risk_percent = 0.01  # 1% risk per trade
    swing_window = 20
    volume_ma_period = 20
    rr_ratio = 1.5
    
    def init(self):
        # 🌠 CELESTIAL INDICATORS (PURE TALIB IMPLEMENTATION)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
        
        # MACD Constellation (STARLIGHT EDITION)
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, self.data.Close,
                                                           fastperiod=12, slowperiod=26,
                                                           signalperiod=9, name='MACD')
        
    def next(self):
        # 🌑 MOON PHASE CHECK - Ensure enough data
        if len(self.data) < self.swing_window + 26:
            return
        
        # 🌊 CURRENT COSMIC ENERGY
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        
        # 🌌 INDICATOR ALIGNMENTS
        current_sw_high = self.swing_high[-1]
        current_sw_low = self.swing_low[-1]
        volume_ma = self.volume_ma[-1] or 1  # Prevent division by zero
        
        # 🌠 MACD MYSTIQUE (NO CROSSOVER IMPORTS USED)
        macd_hist_current = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2] if len(self.macd_hist) > 1 else 0
        
        # 🌟 SIGNAL CONFIRMATIONS (PURE NUMPY/PANDAS LOGIC)
        volume_surge = volume >= 3 * volume_ma
        macd_bullish = (macd_hist_prev < 0) and (macd_hist_current > 0)  # Manual crossover detection
        breakout = price > current_sw_high
        
        # 🚀 LONG ENTRY PROTOCOL
        if breakout and volume_surge and macd_bullish:
            # 💎 RISK MANAGEMENT CRYSTAL
            sl_price = current_sw_low
            risk_per_share = price - sl_price
            
            if risk_per_share > 0:
                position_size = (self.equity * self.risk_percent) / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    tp_price = price + (price - sl_price) * self.rr_ratio
                    self.buy(size=position_size, 
                            sl=sl_price,
                            tp=tp_price)
                    print(f"🌕 MOONSHOT ALERT! 🌕 | Entry: {price:.2f} | Size: {position_size}")
                    print(f"   🛡️ Shield at {sl_price:.2f} | 🌠 Target {tp_price:.2f}")
        
        # 🪐 SHORT LOGIC (MIRRORED FOR COMPLETENESS)
        # ... Similar bearish condition checks ...

    def notify_trade(self, trade):
        if trade.is_closed:
            profit = trade.pl_pct * 100
            if profit > 0:
                print(f"🎯 PROPHECY FULFILLED