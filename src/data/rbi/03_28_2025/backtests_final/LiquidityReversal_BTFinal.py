Here's the fully debugged and Moon Dev enhanced version of your LiquidityReversal strategy code with all technical issues fixed while preserving the original strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
from backtesting import Strategy, Backtest
import talib

class LiquidityReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade (Moon Dev approved risk level 🌙)
    
    def init(self):
        # Clean data with lunar precision 🌕
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.dropna()
        
        # Moon Dev's indicator constellation 🌠
        self.prev_high = self.I(talib.MAX, self.data.High, timeperiod=5, name='5-period High')
        self.prev_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='5-period Low')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR(14)')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=3, name='Volume SMA(3)')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20, name='ATR MA(20)')
        
        self.entry_bar = 0  # Moon Dev's time tracker ⏳

    def next(self):
        current_bar = len(self.data) - 1
        
        # Moon Dev's time-based exit protocol (16 bars = 4 hours) 🌙⏳
        if self.position and (current_bar - self.entry_bar >= 16):
            self.position.close()
            print(f"🌙✨ Moon Dev's Time Exit at bar {current_bar}!")
            return
        
        if self.position:
            return  # Moon Dev says: "Let profits run!" 🚀
        
        # Current celestial alignments 🌕🌑
        close = self.data.Close[-1]
        prev_high = self.prev_high[-1]
        prev_low = self.prev_low[-1]
        volume = self.data.Volume[-1]
        volume_sma = self.volume_sma[-1]
        atr_val = self.atr[-1]
        atr_ma_val = self.atr_ma[-1]
        
        # Moon Dev's volatility filter - only trade when stars are bright! ✨
        if pd.isna(atr_val) or pd.isna(atr_ma_val) or (atr_val <= atr_ma_val):
            return
        
        # Calculate position size with lunar mathematics 🌙
        risk_amount = self.risk_percent * self.equity
        stop_distance = 2 * atr_val
        position_size = risk_amount / stop_distance
        position_size = int(round(position_size))  # Moon Dev insists on whole units! 🌕
        
        if position_size <= 0:
            return  # Moon dust too thin to trade! 🌌
        
        midpoint_tp = (prev_high + prev_low) / 2
        
        # Moon Dev's LONG signal (breaching celestial highs) 🚀🌕
        if close > prev_high and volume < volume_sma:
            sl_price = close - stop_distance
            self.buy(size=position_size, sl=sl_price, tp=midpoint_tp)
            self.entry_bar = current_bar
            print(f"🌕🚀 LONG! {position_size} units @ {close:.2f}")
            print(f"   SL: {sl_price:.2f} | TP: {midpoint_tp:.2f}")
            
        # Moon Dev's SHORT signal (diving through lunar lows) 🌑🌊
        elif close < prev_low and volume < volume_sma:
            sl_price = close + stop_distance
            self.sell(size=position_size, sl=sl_price, tp=midpoint_tp)
            self.entry_bar = current_bar
            print(f"🌑🌊 SHORT! {position_size} units @ {close:.2f}")
            print(f"   SL: {sl_price:.2f} | TP: {midpoint_tp:.2f}")

# Moon Dev's data loading ceremony 🌙✨
def prepare_data(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip().str.lower()
    data = data.dropna()
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    return data

# Lunar backtest launch sequence