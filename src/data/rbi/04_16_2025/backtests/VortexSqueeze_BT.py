```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VORTEXSQUEEZE STRATEGY 🚀

import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VortexSqueeze(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_dev = 2
    vi_period = 14
    sar_acceleration = 0.02
    sar_max = 0.2
    volume_ma_period = 20
    risk_pct = 0.01
    volume_spike_multiplier = 1.5
    vi_convergence_threshold = 0.05

    def init(self):
        # Clean data and prepare indicators 🌙
        # Bollinger Bands using TA-Lib
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev,
            name=['UPPER_BB', 'MIDDLE_BB', 'LOWER_BB']
        )
        
        # Vortex Indicator using pandas_ta
        vi_plus, vi_minus = ta.vortex(
            high=self.data.High, low=self.data.Low, close=self.data.Close,
            length=self.vi_period
        )
        self.vi_plus = self.I(vi_plus, name='VI_PLUS')
        self.vi_minus = self.I(vi_minus, name='VI_MINUS')
        
        # Volume MA using TA-Lib
        self.volume_ma = self.I(
            talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period,
            name='VOLUME_MA20'
        )
        
        # Parabolic SAR using TA-Lib
        self.sar = self.I(
            talib.SAR, self.data.High, self.data.Low,
            acceleration=self.sar_acceleration, maximum=self.sar_max,
            name='SAR'
        )
        
        # Bollinger Bandwidth Squeeze Detection ✨
        bb_bandwidth = self.upper_bb - self.lower_bb
        self.bb_bandwidth_ma = self.I(
            talib.SMA, bb_bandwidth, timeperiod=self.bb_period,
            name='BB_BANDWIDTH_MA'
        )

    def next(self):
        # Moon Dev's Cosmic Trade Execution Logic 🌌
        if self.position:
            # Parabolic SAR Exit Conditions
            if self.position.is_long and self.data.Close[-1] < self.sar[-1]:
                self.position.close()
                print(f"🌙 MOON DEV LONG EXIT ✨ | Price: {self.data.Close[-1]:.2f}, SAR: {self.sar[-1]:.2f}")
            elif self.position.is_short and self.data.Close[-1] > self.sar[-1]:
                self.position.close()
                print(f"🌙 MOON DEV SHORT EXIT ✨ | Price: {self.data.Close[-1]:.2f}, SAR: {self.sar[-1]:.2f}")
        else:
            # Calculate current market conditions
            current_bb_width = self.upper_bb[-1] - self.lower_bb[-1]
            bb_squeeze = current_bb_width < self.bb_bandwidth_ma[-1]
            vi_diff = abs(self.vi_plus[-1] - self.vi_minus[-1])
            vi_converged = vi_diff < self.vi_convergence_threshold
            volume_spike = self.data.Volume[-1] > (self.volume_ma[-1] * self.volume_spike_multiplier)

            # Long Entry Constellation 🌠
            if (self.data.Close[-1] > self.upper_bb[-1] and
                bb_squeeze and
                vi_converged and
                volume_spike):
                
                stop_loss = self.lower_bb[-1]
                risk_per_share = self.data.Close[-1] - stop_loss
                if risk_per_share <= 0: return
                
                position_size = int(round((self.equity * self.risk_pct) / risk_per_share))
                position_size = min(position_size, int(self.equity // self.data.Close[-1]))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙 MOON DEV LONG ENTRY 🚀 | Size: {position_size}, Entry: {self.data.Close[-1