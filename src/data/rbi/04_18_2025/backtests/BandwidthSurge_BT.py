To address the problem, we create a backtesting strategy for the BandwidthSurge approach using Bollinger Bands contraction and volume confirmation. Here's the structured solution:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names and handle data
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

class BandwidthSurge(Strategy):
    def init(self):
        # Bollinger Bands components
        self.upper, self.middle, self.lower = self.I(
            lambda close: talib.BBANDS(close, 20, 2, 2, 0),
            self.data.Close,
            name=['upper', 'middle', 'lower']
        )
        
        # Bandwidth calculation
        self.bandwidth = self.I(
            lambda u, l, m: (u - l) / m,
            self.upper, self.lower, self.middle,
            name='bandwidth'
        )
        
        # Minimum bandwidth over 20 periods
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, 20)
        
        # Volume EMAs
        self.vol_ema5 = self.I(talib.EMA, self.data.Volume, 5)
        self.vol_ema20 = self.I(talib.EMA, self.data.Volume, 20)
        
        # ATR for risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.pending_atr = None

    def next(self):
        # Moon Dev debug prints
        if len(self.data) % 100 == 0:
            print(f"🌙 MOON DEV: Processing bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} 📊")
        
        if not self.position:
            if len(self.data) < 34:  # 20 BB + 14 ATR periods
                return
                
            # Bandwidth condition check
            bandwidth_match = np.isclose(self.bandwidth[-1], self.min_bandwidth[-1], atol=1e-6)
            
            # Volume crossover check
            vol_cross = (self.vol_ema5[-1] > self.vol_ema20[-1]) and \
                        (self.vol_ema5[-2] <= self.vol_ema20[-2])
            
            if bandwidth_match and vol_cross:
                risk_amount = self.equity * 0.01  # 1% risk
                atr_value = self.atr[-1]
                position_size = int(round(risk_amount / atr_value))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.pending_atr = atr_value
                    print(f"🚀 MOON BLASTOFF: Entering {position_size} units at {self.data.Open[-1]:.2f} 🌕")

        else:
            if not hasattr(self, '_entry_atr'):
                self._entry_atr = self.pending_atr
                self._entry_price = self.position.entry_price
                self._entry_bar = len(self.data) - 1
                del self.pending_atr
                print(f"🌑 MOON GRAVITY: Position active | Entry: {self._entry_price:.2f} ⚡")
            
            # Calculate exit levels
            stop_loss = self._entry_price - self._entry_atr
            take_profit = self._entry_price + 1.5 * self._entry_atr
            
            # Check exits
            if self.data.Low[-1] <= stop_loss:
                self.position.close()
                print(f"🌪️ MOON VORTEX: Stop loss triggered @ {stop_loss:.2f} 🛑")
            elif self.data.High[-1] >= take_profit:
                self.position.close()
                print(f"🎯 MOON BULLSEYE: Take profit hit @ {take_profit:.2f