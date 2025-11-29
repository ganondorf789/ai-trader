Here's the fixed code with all `backtesting.lib` imports and functions removed, and proper Moon Dev themed debug prints added:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest

class VolatilitySurge(Strategy):
    # Strategy parameters
    atr_threshold = 100       # Minimum ATR value for volatility filter
    risk_percent = 0.01       # Risk 1% of equity per trade
    atr_period = 14           # ATR period
    bb_period = 20            # Bollinger Bands period
    volume_sma_period = 20    # Volume SMA period
    
    def init(self):
        # Calculate indicators using TA-Lib wrapped in self.I()
        self.atr = self.I(talib.ATR, 
                         self.data.High, self.data.Low, self.data.Close, 
                         self.atr_period, name='ATR')
        
        self.prev_day_high = self.I(talib.MAX, self.data.High, 96, 
                                   name='Previous Day High')
        self.prev_day_low = self.I(talib.MIN, self.data.Low, 96, 
                                  name='Previous Day Low')
        
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 
                                self.volume_sma_period, name='Volume SMA')
        
        # Bollinger Bands (upper, middle, lower)
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            talib.BBANDS, self.data.Close, self.bb_period, 
            nbdevup=2, nbdevdn=2, matype=0, name='BBANDS'
        )

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        equity = self.equity

        if not self.position:
            # Long Entry Conditions 🌕
            if (current_close > self.prev_day_high[-2] and
                current_volume > self.volume_sma[-1] and
                self.atr[-1] > self.atr_threshold):
                
                atr_value = self.atr[-1]
                risk_amount = equity * self.risk_percent
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - atr_value
                risk_per_share = entry_price - stop_loss
                
                position_size = risk_amount / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        sl=stop_loss,
                        tp=entry_price + 2*atr_value
                    )
                    print(f"🌕 MOON DEV LONG SIGNAL! 🚀\n"
                          f"Entry: {entry_price:.2f} | Size: {position_size}\n"
                          f"SL: {stop_loss:.2f} | TP: {entry_price + 2*atr_value:.2f}\n"
                          f"Equity: {equity:.2f} 🌙\n{'-'*30}")

            # Short Entry Conditions 🌑
            elif (current_close < self.prev_day_low[-2] and
                  current_volume > self.volume_sma[-1] and
                  self.atr[-1] > self.atr_threshold):
                
                atr_value = self.atr[-1]
                risk_amount = equity * self.risk_percent
                entry_price = self.data.Close[-1]
                stop_loss = entry_price + atr_value
                risk_per_share = stop_loss - entry_price
                
                position_size = risk_amount / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.sell(
                        size=position_size,
                        sl=stop_loss,
                        tp=entry_price - 2*atr_value
                    )
                    print(f"🌑 MOON DEV SHORT SIGNAL! 🌧️\n"
                          f"Entry: {entry_price:.2f} | Size: {position_size}\n"
                          f"SL: {stop_loss:.2f} | TP: {entry_price - 2*atr_value:.2f}\n"
                          f"Equity: {equity:.2f} 🌙\n{'-'*30}")

    def notify_trade(self, trade):
        if trade.is_closed:
            emoji = "🚀💰 MOON PROFIT!" if trade.pnl > 0 else "🌧️💸 MOON DIP..."
            print(f"🌗 {emoji}\