Here's the complete backtesting implementation for the VolatilityDeviance strategy:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🚀
import pandas as pd
import numpy as np
import talib
import backtrader as bt
from backtesting import Backtest, Strategy

class VolatilityDeviance(Strategy):
    params = (
        ('risk_per_trade', 0.02),    # 2% risk per trade
        ('stop_loss_pct', 0.05),     # 5% stop loss
        ('leverage', 2),             # 2x leverage
        ('rsi_period', 14),          # RSI lookback
        ('di_sma_period', 20),       # DI SMA period
        ('di_atr_period', 20),       # DI ATR period
        ('di_ma_period', 20),        # DI Bollinger MA period
        ('di_std_dev', 2),           # DI Bollinger std dev
    )

    def init(self):
        # 🌙 INDICATOR CALCULATIONS ✨
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Deviance Index (DI) Calculation
        sma = talib.SMA(close, self.p.di_sma_period)
        atr = talib.ATR(high, low, close, self.p.di_atr_period)
        di = (close - sma) / atr
        self.di = self.I(lambda: di, name='DI')
        
        # DI Bollinger Bands
        di_ma = talib.SMA(di, self.p.di_ma_period)
        di_std = talib.STDDEV(di, self.p.di_ma_period)
        self.upper = self.I(lambda: di_ma + self.p.di_std_dev*di_std, name='UPPER')
        self.lower = self.I(lambda: di_ma - self.p.di_std_dev*di_std, name='LOWER')
        
        # RSI Indicator
        self.rsi = self.I(talib.RSI, close, self.p.rsi_period, name='RSI')
        
        self.entry_time = None  # Track entry time for time exits

    def next(self):
        # 🌙 CORE STRATEGY LOGIC 🚀
        if not self.position:
            # Entry Conditions
            if len(self.di) > 1 and len(self.rsi) > 1:
                di_cross = self.di[-2] > self.lower[-2] and self.di[-1] < self.lower[-1]
                rsi_condition = self.rsi[-1] < 30
                
                if di_cross and rsi_condition:
                    # 🚨 RISK MANAGEMENT CALCULATIONS 🛡️
                    equity = self.equity
                    entry_price = self.data.Close[-1]
                    stop_loss = entry_price * (1 - self.p.stop_loss_pct)
                    risk_per_unit = (entry_price - stop_loss) * self.p.leverage
                    position_size = (equity * self.p.risk_per_trade) / risk_per_unit
                    position_size = int(round(position_size))
                    
                    if position_size > 0:
                        print(f"🌙✨🚀 MOON DEV ENTRY SIGNAL! Buying {position_size} @ {entry_price:.2f}")
                        self.buy(size=position_size)
                        self.entry_time = self.data.index[-1]
                        
                        # Set stop loss
                        self.stop_loss = self.sell(size=position_size, 
                                                 exectype=bt.Order.Stop,
                                                 stop=stop_loss)
        else:
            # Exit Conditions
            exit_trade = False
            current_time = self.data.index[-1]
            
            # RSI Exit
            if self.rsi[-1] > 70:
                print(f"🌙✨💎 RSI EXIT @ {self.data.Close[-1]:.2f}")
                exit_trade = True
                
            # DI Reversal Exit
            if len(self.di) > 1:
                di_reversal = self.di[-2] < self.upper[-2] and self.di[-1] > self.upper[-1]
                if di_reversal:
                    print(f"🌙✨💎 DI REVERSAL EXIT @ {self.data.Close[-1]:.2f}")
                    exit_trade = True
                    
            # Time-based Exit (3 months)
            if (current_time - self.entry_time).days >= 90:
                print(f"🌙✨⏰