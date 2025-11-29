```python
# 🌙 Moon Dev's FibroVolumeReversal Backtest Implementation 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class FibroVolumeReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    swing_period = 20
    volume_spike_multiplier = 2
    sma_period = 50
    rr_ratio = 2  # Risk-reward ratio

    def init(self):
        # Trend identification
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period, name='SMA 50')
        
        # Swing points
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='Swing Low')
        
        # Volume analysis
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.swing_period, name='Vol SMA')
        
        # Candlestick patterns
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close, name='Engulfing')

    def next(self):
        if len(self.data.Close) < self.swing_period or len(self.data.Close) < self.sma_period:
            return

        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        # Calculate Fibonacci levels
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        fib_range = swing_high - swing_low
        
        fib_levels = {
            '38.2': swing_high - fib_range * 0.382,
            '50.0': swing_high - fib_range * 0.5,
            '61.8': swing_high - fib_range * 0.618
        }

        # Volume spike calculation
        vol_sma = self.vol_sma[-1]
        volume_spike = current_volume > (vol_sma * self.volume_spike_multiplier) if vol_sma else False

        # Trend direction
        trend_up = current_close > self.sma50[-1]

        if not self.position:
            # Long entry conditions
            if trend_up and volume_spike and (self.engulfing[-1] == 100):
                for level in fib_levels.values():
                    if abs(current_close - level) < 0.005 * current_close:
                        sl = swing_low
                        risk_per_share = current_close - sl
                        if risk_per_share <= 0: continue
                        
                        position_size = int(round((self.risk_percent * self.equity) / risk_per_share))
                        tp = current_close + (risk_per_share * self.rr_ratio)
                        
                        if position_size > 0:
                            self.buy(size=position_size, sl=sl, tp=tp)
                            print(f"🌙✨🚀 BULLISH REVERSAL! Entered LONG at {current_close:.2f}")
                            print(f"   Size: {position_size}, SL: {sl:.2f}, TP: {tp:.2f} 🌟")
                        break

            # Short entry conditions
            elif not trend_up and volume_spike and (self.engulfing[-1] == -100):
                for level in fib_levels.values():
                    if abs(current_close - level) < 0.005 * current_close:
                        sl = swing_high
                        risk_per_share = sl - current_close
                        if risk_per_share <= 0: continue
                        
                        position_size = int(round((self.risk_percent * self.equity) / risk_per_share))
                        tp = current_close - (risk_per_share * self.rr_ratio)
                        
                        if position_size > 0:
                            self.sell(size=position_size, sl=sl, tp=tp)
                            print(f"🌙✨🚀 BEARISH REVERSAL! Entered SHORT at {current_close:.2f}")
                            print(f"   Size: {position_size}, SL: {sl:.2f}, TP: {tp:.2f} 🌟")
                        break

# Data preparation 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15