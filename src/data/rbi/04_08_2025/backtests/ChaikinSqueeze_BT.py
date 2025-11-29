```python
# backtesting_chaikin_squeeze.py
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class ChaikinSqueeze(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade
    atr_period = 14
    bb_period = 20
    adx_period = 14
    swing_period = 5  # For swing low detection
    
    def init(self):
        # Calculate required indicators
        self.chaikin_osc = self.I(self._chaikin_oscillator)
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                    timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)
        self.bbw = self.I(self._calculate_bbw)
        self.bbw_ma = self.I(talib.SMA, self.bbw, timeperiod=self.bb_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def _chaikin_oscillator(self):
        ad = talib.AD(self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        ema3 = talib.EMA(ad, timeperiod=3)
        ema10 = talib.EMA(ad, timeperiod=10)
        return ema3 - ema10

    def _calculate_bbw(self):
        return (self.upper - self.lower) / self.middle

    def next(self):
        current_idx = len(self.data)-1
        if current_idx < max(self.bb_period, self.adx_period, 20)+1:
            return

        # Moon Dev Debug Prints 🌙
        print(f"\n🌕 Bar {current_idx} | Price: {self.data.Close[-1]:.2f}")
        print(f"✨ BBW: {self.bbw[-1]:.4f} vs MA: {self.bbw_ma[-1]:.4f}")
        print(f"📈 ADX: {self.adx[-1]:.2f}, Volume: {self.data.Volume[-1]:.2f} vs MA: {self.volume_ma[-1]:.2f}")

        # Check position exists
        if self.position:
            self._manage_position()
        else:
            self._check_entries()

    def _check_entries(self):
        # Prerequisite conditions
        bbw_condition = self.bbw[-1] < self.bbw_ma[-1]
        adx_condition = self.adx[-1] < 25
        in_squeeze = bbw_condition and adx_condition

        # Bullish divergence check
        price_lows = self.data.Low[-self.swing_period:]
        osc_lows = self.chaikin_osc[-self.swing_period:]
        lower_low = (price_lows[-2] < price_lows[-1]) 
        higher_osc_low = (osc_lows[-2] > osc_lows[-1])
        divergence = lower_low and higher_osc_low

        # Price inside Bollinger Bands
        price_in_bands = (self.data.Close[-1] < self.upper[-1]) and (self.data.Close[-1] > self.lower[-1])

        if in_squeeze and divergence and price_in_bands:
            print("🚀 MOON DEV LONG SIGNAL! 🌟")
            self._enter_long()

    def _enter_long(self):
        # Risk management calculations
        equity