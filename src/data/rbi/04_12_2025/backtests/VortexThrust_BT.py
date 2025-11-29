```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preprocessing
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class VortexThrustStrategy(Strategy):
    adx_period = 14
    vi_period = 14
    volume_ma_period = 20
    vp_lookback = 20
    risk_percent = 0.01
    sar_acceleration = 0.02
    sar_maximum = 0.2
    max_trade_bars = 480  # 5 days in 15m intervals

    def init(self):
        # Calculate indicators with self.I()
        self.vi_plus = self.I(talib.VORTEX, self.data.High, self.data.Low, self.data.Close, self.vi_period, name='+VI', index=0)
        self.vi_minus = self.I(talib.VORTEX, self.data.High, self.data.Low, self.data.Close, self.vi_period, name='-VI', index=1)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period, name='ADX')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period, name='Volume MA')
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, acceleration=self.sar_acceleration, maximum=self.sar_maximum, name='SAR')

    def next(self):
        # Moon Dev debug header 🌙
        if len(self.data) < max(self.adx_period, self.vi_period) + 2:
            return

        # Calculate volume profile levels
        current_idx = len(self.data) - 1
        start_idx = max(0, current_idx - self.vp_lookback)
        end_idx = current_idx  # Lookback excluding current bar
        volume_window = self.data.Volume[start_idx:end_idx]

        if len(volume_window) < 1:
            return

        max_volume_idx = start_idx + np.argmax(volume_window)
        volume_support = self.data.Low[max_volume_idx]
        volume_resistance = self.data.High[max_volume_idx]

        print(f"🌙 Moon Dev Levels: Support={volume_support:.2f}, Resistance={volume_resistance:.2f}")

        # Entry logic
        entry_price = self.data.Close[-1]
        adx_value = self.adx[-1]
        volume_value = self.data.Volume[-1]
        volume_ma_value = self.volume_ma[-1]

        # Long entry conditions
        if (crossover(self.vi_plus, self.vi_minus) and
            adx_value > 25 and
            entry_price > volume_support and
            volume_value > volume_ma_value and
            not self.position.is_long):
            
            risk = entry_price - volume_support
            if risk <= 0:
                print("🚨 Invalid long risk - moon abort!")
                return
                
            size = int(round((self.equity * self.risk_percent) / risk))
            self.buy(size=size, sl=volume_support, tag="🌕 BULLISH THRUST")
            print(f"🚀 MOON ENTRY LONG: {size} units @ {entry_price:.2f}")

        # Short entry conditions
        elif (crossover(self.vi_minus, self.vi_plus) and
              adx_value > 25 and
              entry_price < volume_resistance and
              volume_value > volume_ma_value and
              not self.position.is_short):
            
            risk = volume_resistance - entry_price
            if risk <= 0:
                print("🚨 Invalid short risk - moon abort!")
                return
                
            size = int(round((self.equity