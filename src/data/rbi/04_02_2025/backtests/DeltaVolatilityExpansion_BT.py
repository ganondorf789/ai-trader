```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR DELTA VOLATILITY EXPANSION STRATEGY 🚀

# 1. NECESSARY IMPORTS
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

# 2. DATA HANDLING 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean columns like a Moon Dev pro 🌟
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping with Moon Dev precision 🎯
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'DateTime'
}, inplace=True)
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

# 3. STRATEGY CLASS WITH MOON DEV FLAIR 🌙✨
class DeltaVolatilityExpansion(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    atr_period = 20
    
    def init(self):
        # 🌙 CORE INDICATORS USING TALIB
        self.ema20 = self.I(talib.EMA, self.data.Close, self.atr_period, name='EMA20')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR20')
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, self.atr_period, name='VolumeSMA20')
        
        # 🚀 Precompute upper/lower bands once during init
        upper_band = np.array(self.ema20) + 2 * np.array(self.atr20)
        lower_band = np.array(self.ema20) - 2 * np.array(self.atr20)
        self.upper_band = self.I(lambda: upper_band, name='UpperBand')
        self.lower_band = self.I(lambda: lower_band, name='LowerBand')

    def next(self):
        # 🌙 MOON DEV SAFETY CHECK - Ensure enough data ✨
        if len(self.data) < self.atr_period + 1:
            return
            
        # 🚀 CURRENT VALUES
        price = self.data.Close[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        volume_spike = self.data.Volume[-1] > 2 * self.volume_sma20[-1]

        # 🌙 DELTA IMBALANCE DETECTION (Volume Spike Proxy) 🌊
        if volume_spike:
            print(f"🌙 MOON DEV ALERT! Volume Spike Detected: {self.data.Volume[-1]:.2f} vs SMA20 {self.volume_sma20[-1]:.2f}")

        # 🚀 ENTRY LOGIC: Liquidation Cluster + Breakout
        if not self.position:
            if volume_spike and price > upper:
                self._enter_long()
            elif volume_spike and price < lower:
                self._enter_short()

    def _enter_long(self):
        # 🌙 RISK CALCULATIONS WITH MOON PRECISION 🌕
        entry_price = self.data.Close[-1]
        atr_value = self.atr20[-1]
        stop_loss = entry_price - atr_value
        take_profit = entry_price + 1.5 * atr_value
        
        # 🚀 POSITION SIZING CALCULATION
        risk_amount = self.equity * self.risk_per_trade
        risk_per_share = entry_price - stop_loss
        position_size = int(round(risk_amount / risk_per_share))
        
        if position_size > 0:
            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            print(f"🌙 MOON DEV LONG ENTRY 🚀 | Price: {entry_price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

    def _enter_short(self):
        # 🌑 SHORT ENTRY WITH MOON DEV RISK MANAGEMENT 🌙
        entry_price = self