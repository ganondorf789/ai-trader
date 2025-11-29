```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as pd_ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolClusterBreakout(Strategy):
    def init(self):
        # Precompute indicators
        # Volatility-adjusted VWAP Channel
        vwap = pd_ta.vwap(high=self.data.High, low=self.data.Low, close=self.data.Close, 
                         volume=self.data.Volume, anchor='D')
        std_dev = pd_ta.stddev(self.data.Close, length=20)
        upper_band = vwap + 2 * std_dev
        lower_band = vwap - 2 * std_dev
        
        self.vwap_ind = self.I(vwap, name='VWAP')
        self.upper_band_ind = self.I(upper_band, name='UpperBand')
        self.lower_band_ind = self.I(lower_band, name='LowerBand')

        # Liquidation Volume Confirmation
        volume_ma = pd_ta.sma(self.data.Volume, length=20)
        self.volume_surge = self.I((self.data.Volume > 2*volume_ma).astype(int), name='VolumeSurge')

        # Fractal Volatility Measure
        self.atr = self.I(talib.ATR(self.data.High, self.data.Low, self.data.Close, 14), name='ATR')
        self.swing_high = self.I(talib.MAX(self.data.High, 20), name='SwingHigh')
        self.swing_low = self.I(talib.MIN(self.data.Low, 20), name='SwingLow')

        # Median ATR for volatility filter
        self.median_atr = self.atr.series.median()

    def next(self):
        # Skip trades during high volatility or existing positions
        if self.atr[-1] > 3*self.median_atr:
            print(f"🌙⚠️ Volatility alert! ATR {self.atr[-1]:.2f} > 3x median")
            return
            
        if self.position:
            return

        # Entry logic
        price_close = self.data.Close[-1]
        upper_band = self.upper_band_ind[-1]
        lower_band = self.lower_band_ind[-1]
        volume_surge = self.volume_surge[-1]

        if price_close > upper_band and volume_surge:
            # Long entry with Moon-themed risk management 🌙
            sl_price = self.swing_low[-1] - 1.5*self.atr[-1]
            risk_per_unit = price_close - sl_price
            position_size = int(round((self.equity*0.01)/risk_per_unit))
            
            if position_size > 0:
                self.buy(size=position_size, 
                        sl=sl_price,
                        tp=price_close + 2*risk_per_unit)
                print(f"🌙🚀 LONG SIGNAL! Size: {position_size} | Entry: {price_close:.2f}")
                print(f"   MOON STOP: {sl_price:.2f} | STARGAZER TP: {price_close + 2*risk_per_unit:.2f}")

        elif price_close < lower_band and volume_surge:
            # Short entry with cosmic precision 🌌
            sl_price = self.swing_high[-1] + 1.5*self.atr[-1]
            risk_per_unit = sl_price - price_close
            position_size = int(round((self.equity*0.01)/risk_per_unit))
            
            if position_size > 0:
                self.sell(size=position_size,
                         sl=sl_price,
                         tp=price_close - 2*risk_per_unit)
                print(f"🌙📉 SHORT SIGNAL! Size: {position_size} | Entry: {price_close:.2f}")
                print(f"   BLACK HOLE STOP: {sl_price:.2f} | GRAVITY TP