```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VoltaicContraction(Strategy):
    risk_pct = 0.02
    atr_period = 14
    adx_period = 14
    bb_width_threshold = 0.2
    exit_bars = 192  # 48 hours in 15m intervals
    
    def init(self):
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, self.atr_period)
        
        # Trend strength
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        
        # VWAP with 2σ bands
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume, anchor='D')
        self.vwap_std = self.I(talib.STDDEV, self.data.Close, 14)
        self.upper_band = self.I(lambda vwap, std: vwap + 2*std, self.vwap, self.vwap_std)
        self.lower_band = self.I(lambda vwap, std: vwap - 2*std, self.vwap, self.vwap_std)
        
        # Bollinger Bands (4h equivalent)
        self.bb_upper = self.I(talib.MAX, self.data.High, 16)
        self.bb_lower = self.I(talib.MIN, self.data.Low, 16)
        self.bb_mid = self.I(talib.SMA, self.data.Close, 16)
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_mid)
        
        self.entry_bar = 0

    def next(self):
        price = self.data.Close[-1]
        
        if self.position:
            # Exit conditions
            bb_width = self.bb_width[-1]
            time_exit = len(self.data) - self.entry_bar >= self.exit_bars
            sl_hit = (self.position.is_long and price <= self.stop_loss) or \
                    (self.position.is_short and price >= self.stop_loss)
            
            if bb_width < self.bb_width_threshold or time_exit or sl_hit:
                moon_emoji = '🌙✨' if bb_width < self.bb_width_threshold else '🕒⏳' if time_exit else '🔴💔'
                print(f"{moon_emoji} Closing position | BB Width: {bb_width:.4f}")
                self.position.close()
        else:
            # Entry conditions
            atr_valid = self.atr[-1] > self.atr_ma[-1]
            adx_valid = self.adx[-1] < 25
            vwap_deviation = price > self.upper_band[-1] or price < self.lower_band[-1]
            
            if atr_valid and adx_valid and vwap_deviation:
                risk_amount = self.equity * self.risk_pct
                atr_value = self.atr[-1]
                stop_distance = 1.5 * atr_value
                
                if price < self.vwap[-1]:  # Long setup
                    self.stop_loss = price - stop_distance
                    position_size = int(round(risk_amount / stop_distance))
                    self.buy(size=position_size)
                    self.entry_bar = len(self.data)
                    print(f"🚀🌙 LONG | Size: {position_size} | SL: {self.stop_loss:.2f}")
                else:  # Short