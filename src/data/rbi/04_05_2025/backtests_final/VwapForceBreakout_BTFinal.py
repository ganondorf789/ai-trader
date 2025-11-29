I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Strategy, Backtest

class VwapForceBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    vwap_period = 20
    force_index_period = 13
    
    def init(self):
        # Clean and prepare data with lunar precision 🌙
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)
        
        # Calculate indicators using self.I()
        # VWAP using pandas_ta
        self.vwap = self.I(pta.vwap, 
                          high=self.data.High, 
                          low=self.data.Low, 
                          close=self.data.Close, 
                          volume=self.data.Volume, 
                          length=self.vwap_period,
                          name='VWAP')
        
        # Elder's Force Index with SMA
        self.force_index = self.I(pta.fi,
                                 self.data.Close,
                                 self.data.Volume,
                                 length=self.force_index_period,
                                 name='ForceIndex')
        
        self.force_index_sma = self.I(talib.SMA,
                                     self.force_index,
                                     timeperiod=self.force_index_period,
                                     name='ForceIndex_SMA')
        
        # Volume filter
        self.volume_sma = self.I(talib.SMA,
                                self.data.Volume,
                                timeperiod=20,
                                name='Volume_SMA')
        
        # Trend filter (50-period SMA)
        self.sma50 = self.I(talib.SMA,
                           self.data.Close,
                           timeperiod=50,
                           name='SMA50')
        
        print("🌙 Moon Dev Indicators Initialized: VWAP, Force Index, Volume SMA & Trend Filter Active!")

    def next(self):
        current_close = self.data.Close[-1]
        vwap_value = self.vwap[-1]
        force_value = self.force_index[-1]
        
        # Moon Dev Trade Filters
        volume_filter = self.data.Volume[-1] > self.volume_sma[-1]
        trend_filter_long = current_close > self.sma50[-1]
        trend_filter_short = current_close < self.sma50[-1]
        
        # Entry Logic with Moon Dev Precision 🌙
        if not self.position:
            # Long Entry Constellation 🌟
            if (self.data.Close[-2] < self.vwap[-2] and self.data.Close[-1] > self.vwap[-1]) \
                and (force_value > 0.5 or (self.force_index[-2] < self.force_index_sma[-2] and self.force_index[-1] > self.force_index_sma[-1])) \
                and volume_filter \
                and trend_filter_long:
                
                risk_amount = self.equity * self.risk_per_trade
                stop_loss = self.vwap[-1]
                risk_per_unit = abs(current_close - stop_loss)
                
                if risk_per_unit == 0:
                    print("🌑 Moon Dev Warning: Zero risk detected, skipping trade")
                    return
                
                position_size = int(round(risk_amount / risk_per_unit))
                take_profit = current_close + 2 * (current_close - stop_loss)
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=stop_loss,
                            tp=take_profit,
                            tag="Moon Dev Long Entry 🚀")
                    print(f"🚀 LUNAR LIFTOFF: Long {position_size} units at {current_close} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

            # Short Entry Meteor Shower 🌠    
            elif (self.data.Close[-2] > self.vwap[-2] and self.data.Close[-1] < self.vwap[-1]) \
                and (force_value < -0.5 or (self.force_index[-2] > self.force_index_sma[-2] and self.force_index[-1] < self.force_index_sma[-1])) \
                and volume_filter \
                and trend_filter_short:
                
                risk_amount = self.equity * self.risk_per_trade
                stop_loss = self.vwap[-1]
                risk_per_unit = abs(stop_loss - current_close)
                
                if risk_per_unit == 0:
                    print("🌑 Moon Dev Warning: Zero risk detected, skipping trade")
                    return