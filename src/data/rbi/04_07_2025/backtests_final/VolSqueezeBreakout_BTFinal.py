I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

class VolSqueezeBreakout(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_dev = 2
    cmf_period = 20
    volume_ma_period = 20
    atr_period = 14
    risk_per_trade = 0.01  # 1% risk per trade
    max_positions = 3
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib
        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                   timeperiod=self.bb_period, nbdevup=self.bb_dev, 
                                                   nbdevdn=self.bb_dev, matype=0)
        # Bollinger Band Width
        self.bb_width = self.I(lambda u,m,l: (u - l)/m, self.upper, self.middle, self.lower)
        
        # Chaikin Money Flow
        self.cmf = self.I(talib.ADOSC, self.data.High, self.data.Low, self.data.Close, self.data.Volume,
                        fastperiod=3, slowperiod=self.cmf_period)
        
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
        self.current_volume = self.I(lambda: self.data.Volume)
        
        # ATR for risk calculation
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # Track position counts
        self.active_positions = 0

    def next(self):
        current_price = self.data.Close[-1]
        
        # Moon Dev debug prints 🌙
        print(f"\n🌕 Moon Dev Status Update 🌕")
        print(f"🕰 Time: {self.data.index[-1]}")
        print(f"💰 Price: {current_price:.2f}")
        print(f"📈 BB Width: {self.bb_width[-1]:.4f}")
        print(f"🌊 CMF: {self.cmf[-1]:.2f}")
        print(f"📊 Volume: {self.current_volume[-1]:.2f} vs MA: {self.volume_ma[-1]:.2f}")
        
        # Exit conditions
        for trade in self.trades:
            if trade.is_long:
                # Partial profit target
                if current_price >= trade.entry_price + 1.5*self.atr[-1]:
                    if trade.size > 1:
                        print(f"🚀 Taking partial profits! Selling 50% at {current_price:.2f}")
                        self.sell(size=int(round(trade.size/2)))
                
                # Trailing stop
                trail_price = current_price - 2*self.atr[-1]
                if trail_price > trade.sl:
                    print(f"🎢 Updating trailing stop to {trail_price:.2f}")
                    trade.sl = max(trail_price, trade.sl)
            
            if not trade.is_active:
                self.active_positions -= 1

        # Entry conditions check
        if self.active_positions >= self.max_positions:
            print("🌙 Maximum positions reached - no new trades")
            return

        # Calculate BB width percentile
        bb_lookback = 20
        if len(self.bb_width) > bb_lookback:
            recent_width = self.bb_width[-bb_lookback:]
            width_percentile = (self.bb_width[-1] - np.min(recent_width)) / (np.max(recent_width) - np.min(recent_width))
            print(f"📉 BB Width Percentile: {width_percentile:.2%}")
        else:
            return

        # CMF trend check (last 5 periods)
        cmf_slope = np.polyfit(range(5), self.cmf[-5:], 1)[0] if len(self.cmf) >=5 else 0

        # Volume surge check
        volume_ok = self.current_volume[-1] >= 1.5 * self.