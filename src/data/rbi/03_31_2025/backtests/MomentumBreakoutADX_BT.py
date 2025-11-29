```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy
from backtesting import Backtest

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class MomentumBreakoutADX(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 2
    trailing_atr_multiplier = 1.5
    
    def init(self):
        # Calculate indicators using TA-Lib through self.I() ✨
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=5)
        self.recent_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        
        self.trailing_stop = None
        
    def next(self):
        current_idx = len(self.data) - 1
        
        # Skip early bars before all indicators are calculated
        if current_idx < 200:
            return
            
        # Moon Dev trend check 🌙
        ema50_current = self.ema50[-1]
        ema200_current = self.ema200[-1]
        adx_current = self.adx[-1]
        atr_current = self.atr[-1]
        recent_high = self.recent_high[-1]
        
        # Entry Logic 🚀
        if not self.position:
            # Golden cross condition
            ema_cross = ema50_current > ema200_current and self.ema50[-2] <= self.ema200[-2]
            
            # Trend strength confirmation
            adx_strong = adx_current > 25
            
            # Breakout confirmation
            price_breakout = self.data.Close[-1] > recent_high
            
            if all([ema_cross, adx_strong, price_breakout]):
                # Risk management calculations 🌙
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - (self.atr_multiplier * atr_current)
                risk_per_share = entry_price - stop_loss
                
                # Position sizing calculation
                risk_amount = self.risk_per_trade * self.equity
                position_size = risk_amount / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    # Moon Dev entry signal 🌕
                    print(f"🚀 MOON DEV ENTRY SIGNAL 🚀")
                    print(f"Entry Price: {entry_price:.2f}")
                    print(f"Position Size: {position_size} units")
                    print(f"Stop Loss: {stop_loss:.2f} (ATR: {atr_current:.2f})")
                    
                    self.buy(size=position_size, sl=stop_loss)
                    self.trailing_stop = None
        
        # Exit Logic 💰
        elif self.position.is_long:
            entry_price = self.position.entry_price
            current_high = self.data.High[-1]
            
            # Update trailing stop
            new_trailing_stop = current_high - (self.trailing_atr_multiplier * atr_current)
            self.trailing_stop = max(self.trailing_stop, new_trailing_stop) if self.trailing_stop else new_trailing_stop
            
            # RSI exit condition
            if self.rsi[-1] < 70 and self.rsi[-2] >= 70:
                print(f"🌙 MOON DEV RSI EXIT ({self.rsi[-1]:.2f} < 70)")
                self.position.close()
            
            # Trailing stop check
            elif self.data.Close[-1