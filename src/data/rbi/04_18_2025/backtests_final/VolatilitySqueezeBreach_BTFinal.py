I've analyzed the code and found a few technical issues that need fixing while preserving the strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

# Data preparation - Lunar data acquisition initiated 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data - Cosmic dust removal in progress ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySqueezeBreach(Strategy):
    def init(self):
        # Calculate indicators using TA-Lib - Lunar technical analysis engaged 🌕
        close = self.data.Close.values
        high = self.data.High.values
        low = self.data.Low.values
        volume = self.data.Volume.values
        
        # Bollinger Bands (20,2) - Cosmic pressure bands forming 🌌
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.upper_band = self.I(lambda: pd.Series(upper), 'Upper BB')
        self.lower_band = self.I(lambda: pd.Series(lower), 'Lower BB')
        
        # Bandwidth calculation - Gravitational squeeze detected 🌑
        bandwidth = (upper - lower) / middle
        self.bandwidth = self.I(lambda: pd.Series(bandwidth), 'Bandwidth')
        self.min_bandwidth = self.I(lambda: pd.Series(talib.MIN(bandwidth, 960)), 'Min Bandwidth')
        
        # Volume indicators - Cosmic energy flow measured 🌠
        self.volume_avg = self.I(lambda: pd.Series(talib.SMA(volume, 960)), 'Volume Avg')
        self.swing_low = self.I(lambda: pd.Series(talib.MIN(low, 5)), 'Swing Low')
        
        print("🌙 MOON DEV: Lunar indicators activated! Trading systems at full power! 🚀✨")

    def next(self):
        price = self.data.Close[-1]
        
        # Entry conditions - Launch sequence initiated 🛰️
        if not self.position:
            if (np.isclose(self.bandwidth[-1], self.min_bandwidth[-1], rtol=0.001) and \
               self.data.High[-1] > self.upper_band[-1] and \
               self.data.Volume[-1] > self.volume_avg[-1]):
                
                # Risk management calculations - Cosmic shield engaged 🛡️
                risk_percent = 0.01  # 1% risk
                stop_price = self.swing_low[-1] * 0.995
                risk_per_share = price - stop_price
                
                if risk_per_share > 0:
                    size = int(round((self.equity * risk_percent) / risk_per_share))
                    if size > 0:
                        self.buy(size=size)
                        print(f"🌕 MOON DEV LIFTOFF: Long {size} units @ {price} - Cosmic thrusters engaged! 🚀🌌")

        # Exit conditions - Re-entry protocol activated 🪂
        elif self.position:
            exit_trade = False
            if self.data.Low[-1] <= self.lower_band[-1]:
                print(f"🌑 MOON DEV LANDING: Lower band touched @ {price} - Gravitational pull too strong! 🌙")
                exit_trade = True
            elif self.data.Volume[-1] < self.volume_avg[-1]:
                print(f"🌒 MOON DEV ECLIPSE: Volume dropped @ {price} - Cosmic energy fading! 🌘")
                exit_trade = True
            
            if exit_trade:
                self.position.close()
                print(f"🌗 MOON DEV ORBIT: Position closed @ {price} - Safe in mothership! 💫🛸")

# Execute backtest - Lunar mission commencing 🌖
bt = Backtest(data, VolatilitySqueezeBreach, cash=1_000_000)
stats = bt.run()
print("🌕 MOON DEV MISSION COMPLETE: Back