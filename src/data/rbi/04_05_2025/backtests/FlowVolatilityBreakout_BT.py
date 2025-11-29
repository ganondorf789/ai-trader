```python
# 🌙 Moon Dev's FlowVolatilityBreakout Backtest Script
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas as pd
import numpy as np

class FlowVolatilityBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Core Indicators
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        self.ema200 = self.I(talib.EMA, self.data.Close, 200)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr22 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 22)
        
        # 🌊 Netflow Indicators
        self.netflow_ma20 = self.I(talib.SMA, self.data.Netflow, 20)
        self.netflow_std = self.I(talib.STDDEV, self.data.Netflow, 20)
        
        # 🎯 Swing Points
        self.swing_high = self.I(talib.MAX, self.data.High, 5)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        
        # 📊 Volume Analysis
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        # 🚦 Trade Tracking
        self.entry_price = None
        self.trailing_stop = None

    def next(self):
        current_close = self.data.Close[-1]
        current_time = self.data.index[-1].time()
        
        # 🌙 Moon Dev Debug Prints
        print(f"🌙 Current Balance: ${self.equity:,.2f} | Price: {current_close:.2f} | ADX: {self.adx[-1]:.1f}")
        
        # 🚫 Skip entries during high liquidity
        if not self._is_low_liquidity(current_time):
            return
            
        # 🛑 Exit Logic First
        if self.position:
            self._handle_exits(current_close)
            return
            
        # ✅ Entry Conditions Check
        if self._valid_long_conditions():
            self._enter_long()
        elif self._valid_short_conditions():
            self._enter_short()

    def _is_low_liquidity(self, time):
        # 🕒 Moon Dev's Liquidity Hours (12AM-4AM & 12PM-2PM UTC)
        low_vol = self.data.Volume[-1] < self.volume_ma[-1]
        return (0 <= time.hour < 4) or (12 <= time.hour < 14) and low_vol

    def _valid_long_conditions(self):
        bull_div = (self.data.Low[-3:].idxmin() == len(self.data)-1 and 
                   self.data.Netflow[-3:].idxmin() < len(self.data)-1)
        return (self.adx[-1] > 25 and
                self.data.Close[-1] > self.ema200[-1] and
                bull_div and
                self.data.Close[-1] > self.swing_high[-1] and
                (self.data.Netflow[-1] - self.netflow_ma20[-1]) > 2*self.netflow_std[-1])

    def _enter_long(self):
        risk_amount = self.equity * self.risk_per_trade
        stop_loss = self.data.Close[-1] - 1.5*self.atr14[-1]
        position_size = int(round(risk_amount / (self.data.Close[-1] - stop_loss)))
        
        if position_size > 0:
            self.buy(size=position_size)
            self.entry_price = self.data.Close[-1]
            self.trailing_stop = stop_loss
            print(f"🚀 LONG ENTRY: {self.data.index[-1]} | Size: {position_size} | SL: {stop_loss:.2f}")

    def _handle_exits(self, current_close):
        if self.position.is_long:
            # 🌗 Chande Volatility Stop
            self.trailing_stop = max(self.trailing_stop, 
                                    talib.MAX(self.data.High, 22)[-1] - 3*self.atr22[-1])
            
            if current_close <= self.trailing_stop:
                self.position.close()
                print(f"🛑 EXIT LONG: Chande Stop | PNL: {self