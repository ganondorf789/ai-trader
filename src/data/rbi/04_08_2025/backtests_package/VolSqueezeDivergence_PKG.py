Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev Backtest Script for VolSqueezeDivergence Strategy
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolSqueezeDivergence(Strategy):
    def init(self):
        # 🌙 Indicator Parameters
        self.bb_period = 20
        self.rsi_period = 14
        self.atr_period = 20
        self.keltner_multiplier = 2.5
        self.percentile_window = 100
        self.swing_period = 5

        # 🛠️ Clean Data (already preprocessed in backtest)
        
        # 📈 Technical Indicators
        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                            timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, 
                                                            matype=0, name=['BB_UP', 'BB_MID', 'BB_LOW'])
        
        # Bollinger Bandwidth Calculation
        bb_bandwidth = (self.bb_upper - self.bb_lower) / self.bb_mid
        self.bb_bandwidth = self.I(lambda: bb_bandwidth, name='BB_WIDTH')
        self.bb_percentile = self.I(lambda x: x.rolling(self.percentile_window).quantile(0.1), 
                                   self.bb_bandwidth, name='BB_PERC_10')

        # Volume-Weighted RSI
        rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')
        vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_SMA')
        self.vw_rsi = self.I(lambda: rsi * (self.data.Volume / vol_sma), name='VW_RSI')

        # Keltner Channels
        ema = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA')
        atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                    timeperiod=self.atr_period, name='ATR')
        self.keltner_upper = self.I(lambda: ema + self.keltner_multiplier * atr, name='KELT_UP')
        self.keltner_lower = self.I(lambda: ema - self.keltner_multiplier * atr, name='KELT_LOW')

        # Volume Confirmation
        self.vol_confirm = self.I(lambda: self.data.Volume > vol_sma, name='VOL_CONFIRM')

        # Swing Lows
        self.swing_low_price = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING_LOW')
        self.swing_low_rsi = self.I(talib.MIN, self.vw_rsi, timeperiod=self.swing_period, name='RSI_SWING_LOW')

        print("✨ Moon Dev Indicators Initialized! ✨")
        print("🌙 All indicators powered by TA-Lib with no backtesting.lib dependencies! 🌙")

    def next(self):
        # 🌙 Skip initial bars without indicator data
        if len(self.data) < 100 or len(self.vw_rsi) < 2:
            return

        # 🚀 Entry Conditions
        if not self.position:
            # 1. Bollinger Bandwidth contraction
            bb_contraction = self.bb_bandwidth[-1] < self.bb_percentile[-1]
            
            # 2. Bullish RSI divergence
            price_lower_low = self.data.Low[-1] < self.data.Low[-2]
            rsi_higher_low = self.vw_rsi[-1] > self.vw_rsi[-2]
            divergence = price_lower_low and rsi_higher_low
            
            # 3. Breakout confirmation
            breakout = self.data.Close[-1] > self.bb_upper[-1]
            volume_ok = self.vol_confirm[-1]

            if bb_contraction and divergence and breakout and volume_ok:
                # 🎯 Risk Management Calculations
                risk_pct = 0.01  # 1% of equity
                equity = self._broker.equity
                risk_amount = equity * risk_pct
                
                # Stop Loss Calculation
                swing_l