```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class DynamicDivergence(Strategy):
    adx_period = 14
    vwap_period = 20
    atr_period = 14
    risk_percent = 0.01
    swing_window = 5

    def init(self):
        # Calculate indicators using self.I()
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Calculate VWAP using pandas_ta
        def calc_vwap(close, volume):
            return ta.vwma(close, volume, length=self.vwap_period)
        self.vwap = self.I(calc_vwap, self.data.Close, self.data.Volume)
        
        # Swing detection
        self.price_swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.price_swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        self.obv_swing_high = self.I(talib.MAX, self.obv, self.swing_window)
        self.obv_swing_low = self.I(talib.MIN, self.obv, self.swing_window)

        self.price_lows = []
        self.obv_lows = []
        self.price_highs = []
        self.obv_highs = []

    def next(self):
        # Moon Dev debug prints 🌙✨
        print(f"\n🌙 New candle: {self.data.index[-1]} | Close: {self.data.Close[-1]:.2f}")
        
        if len(self.data) < max(self.adx_period, self.vwap_period, self.swing_window)+1:
            return

        current_adx = self.adx[-1]
        if current_adx < 25:
            print(f"    ✨ ADX {current_adx:.1f} <25 - Market not trending")
            return

        # Detect swing points
        current_price_low = self.data.Low[-1]
        if current_price_low == self.price_swing_low[-1]:
            self.price_lows.append((len(self.data)-1, current_price_low))
            print(f"    🌙 Price Swing Low at {current_price_low:.2f}")
            
        current_obv_low = self.obv[-1]
        if current_obv_low == self.obv_swing_low[-1]:
            self.obv_lows.append((len(self.data)-1, current_obv_low))
            print(f"    🌙 OBV Swing Low at {current_obv_low:.2f}")

        # Check for divergences
        long_signal = self.check_bullish_divergence()
        short_signal = self.check_bearish_divergence()

        # Entry conditions
        vwap_distance = abs(self.data.Close[-1] - self.vwap[-1])/self.vwap[-1]
        print(f"    ✨ VWAP Distance: {vwap_distance*100:.2f}%")
        
        if not self.position:
            if long_signal and vwap_distance < 0.005:
                self.enter_long()
            elif short_signal and vwap_distance < 0.005:
                self.enter_short()

        # Update trailing stops
        for trade in self.trades:
            if trade.is_long:
                new_sl = self.trailing_stop_long(trade)
                trade.sl = new_sl
            elif trade.is_short:
                new_sl = self.trailing_stop_short(trade)
                trade.sl = new_sl

    def check_bullish_divergence(self):
        if len(self.price_lows) < 2 or len(self.obv_lows) < 2:
            return False
            
        # Get last two swing lows
        price_low1 = self.price_lows[-2][1]
        price_low2 = self.price_lows[-1][1]
        obv_low1 = self.obv_lows[-2][1]
        obv_low2 = self.obv_lows[-1][1]
        
        return (price_low