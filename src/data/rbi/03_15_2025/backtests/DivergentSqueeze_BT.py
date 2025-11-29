```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, crossunder

class DivergentSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    swing_period = 5
    bb_period = 20
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    def init(self):
        # Precompute indicators using TA-Lib
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low
        
        # MACD components
        macd, macd_signal, _ = talib.MACD(self.close, 
                                         fastperiod=self.macd_fast,
                                         slowperiod=self.macd_slow,
                                         signalperiod=self.macd_signal)
        self.macd_line = self.I(lambda: macd, name='MACD')
        self.macd_signal = self.I(lambda: macd_signal, name='MACD_Signal')
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.close, 
            timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Volume analysis
        self.volume_slope = self.I(talib.LINEARREG_SLOPE, self.data.Volume, 
                                  timeperiod=5, name='Volume_Slope')
        
        # Swing points
        self.swing_high = self.I(talib.MAX, self.high, 
                                timeperiod=self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.low, 
                               timeperiod=self.swing_period, name='Swing_Low')
        
        # Bollinger Band Width analysis
        bb_width = self.bb_upper - self.bb_lower
        self.bb_width_slope = self.I(talib.LINEARREG_SLOPE, bb_width, 
                                    timeperiod=20, name='BB_Width_Slope')
        
    def next(self):
        # Skip early bars where indicators are not fully formed
        if len(self.data) < 30:
            return
        
        # Current market conditions
        current_close = self.close[-1]
        current_high = self.high[-1]
        current_low = self.low[-1]
        prev_low = self.low[-2]
        prev_macd = self.macd_line[-2]
        
        # Check for bullish divergence conditions
        bullish_divergence = (
            current_low < prev_low and 
            self.macd_line[-1] > prev_macd
        )
        
        # Volume contraction check
        volume_contraction = self.volume_slope[-1] < 0
        
        # Bollinger Band narrowing check
        bb_narrowing = self.bb_width_slope[-1] < 0
        
        # Confirmation candle (close above previous high)
        confirmation_bullish = current_close > self.high[-2]
        
        # Long entry logic
        if (bullish_divergence and volume_contraction and 
            bb_narrowing and confirmation_bullish and not self.position):
            
            # Calculate stop loss and take profit
            stop_loss_price = self.swing_low[-1] * 0.995
            take_profit_price = self.bb_middle[-1]
            
            # Position sizing calculation
            risk_amount = self.equity * self.risk_per_trade
            risk_per_share = self.data.Open[-1] - stop_loss_price
            position_size = int(round(risk_amount / abs(risk_per_share)))
            
            if position_size > 0:
                self.buy(
                    size=position_size,
                    sl=stop_loss_price,
                    tp=take_profit_price,
                    tag="🌙 Bullish Divergence Entry"
                )
                print(f"🚀 Moon Launch: LONG @ {self.data.Open[-1]:.2f} | "
                      f"SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f}")
        
        # Check for bearish divergence conditions
        bearish_divergence = (
            self.high[-1] > self.high[-2] and 
            self.macd_line[-1]