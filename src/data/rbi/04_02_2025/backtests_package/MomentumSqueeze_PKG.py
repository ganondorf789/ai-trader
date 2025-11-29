Here's the fixed and completed code with proper Moon Dev theming and indicator package verification:

```python
# 🌙 Moon Dev's Momentum Squeeze Backtest 🌙
from backtesting import Backtest, Strategy
from talib import CMO, BBANDS, SMA, ATR
import pandas as pd
import numpy as np

class MomentumSqueezeStrategy(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🚀 Moon Dev Indicator Setup
        self.cmo = self.I(CMO, self.data.Close, timeperiod=14, name='CMO_14')
        
        # Bollinger Bands (20,2) - Pure TA-Lib implementation
        def bb_upper(close):
            upper, _, _ = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_Upper')
        
        def bb_middle(close):
            _, middle, _ = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return middle
        self.bb_middle = self.I(bb_middle, self.data.Close, name='BB_Middle')
        
        # Bollinger Band Width (Upper-Lower)/Middle
        def bb_width(close):
            upper, middle, lower = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close, name='BB_Width')
        
        # BB Width 20 SMA
        self.bb_width_ma = self.I(SMA, self.bb_width, timeperiod=20, name='BB_Width_MA')
        
        # ATR for exits
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_14')
        
        print("✨ Moon Dev Indicators Initialized with Pure TA-Lib Power! ✨")

    def next(self):
        # 🌙 Moon Dev Entry Logic
        if len(self.cmo) < 2 or len(self.bb_width_ma) < 1:
            return
            
        # Entry Conditions (No backtesting.lib used!)
        cmo_rising = self.cmo[-1] > self.cmo[-2]  # Direct array comparison
        bb_squeeze = self.bb_width[-1] < self.bb_width_ma[-1]  # Direct comparison
        price_above_middle = self.data.Close[-1] > self.bb_middle[-1]
        
        if cmo_rising and bb_squeeze and price_above_middle and not self.position:
            # 🎯 Risk Calculation
            atr_value = self.atr[-1]
            if atr_value <= 0:
                return
                
            risk_amount = self.risk_per_trade * self.equity
            position_size = int(round(risk_amount / atr_value))
            
            if position_size > 0:
                # 🚀 Execute Long Entry
                entry_bar = len(self.data)
                self.buy(size=position_size, 
                         tag={'atr': atr_value, 'entry_bar': entry_bar})
                print(f"🚀 MOON DEV LONG ENTRY: {position_size} shares @ {self.data.Close[-1]:.2f} | ATR: {atr_value:.2f}")
        
        # 🌑 Moon Dev Exit Management
        for trade in self.trades:
            if trade.is_open:
                entry_price = trade.entry_price
                atr_value = trade.tag['atr']
                entry_bar = trade.tag['entry_bar']
                
                tp_level = entry_price + 2 * atr_value
                sl_level = entry_price - 1 * atr_value
                current_bar = len(self.data) - 1
                
                # Take Profit Check
                if self.data.High[-1] >= tp_level:
                    trade.close()
                    print(f"🌕 MOON DEV TP HIT: {self.data.Close[-1]:.2f} | Profit: {trade.pl:.2f}")
                
                # Stop Loss Check
                elif self.data.Low[-1] <= sl_level:
                    trade.close()
                    print(f"🌑 MOON DEV SL HIT: {self.data.Close[-1]:.2f} | Loss: {trade.pl:.2f}")
                
                # Time Exit (5 bars)
                elif (current_bar - entry_bar) >= 5:
                    trade.close()
                    print(f"