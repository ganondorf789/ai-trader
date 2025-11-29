Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class SqueezeReversion(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    bb_period = 20
    atr_period = 14
    lookback_period = 1344  # 14 days in 15m intervals (14*96=1344)
    
    def init(self):
        # 🌙 MOON DEV INDICATOR SETUP ✨
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER')
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_MIDDLE')
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER')
        
        # ✨ Calculate BB Width dynamically using pure pandas operations
        self.bb_width = self.I(lambda: ((self.bb_upper - self.bb_lower) / self.bb_middle), name='BB_WIDTH')
        
        # 📉 Funding Rate analysis with pure talib
        self.funding_rate = self.data.df['fundingrate']
        self.funding_low = self.I(talib.MIN, self.funding_rate, timeperiod=self.lookback_period, name='FUNDING_LOW')
        
        # 📈 ATR for risk management (pure talib)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        
        self.entry_signal = False  # 🌙 Signal tracking
        
    def next(self):
        # 🚀 MOON DEV CORE LOGIC
        if len(self.data) < self.lookback_period:
            return
            
        # Current market state (using array indexing instead of crossover)
        current_bb_width = ((self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1])
        bb_squeeze = current_bb_width < self.bb_width[-self.lookback_period:].quantile(0.05)
        funding_drop = self.funding_rate[-1] < self.funding_low[-1]
        
        # 🌋 Entry Condition Check
        if bb_squeeze and funding_drop and not self.position:
            self.entry_signal = True
            print(f"🌋 MOON ERUPTION: Squeeze + Funding Drop @ {self.data.Close[-1]}")

        # 🚀 Execute Entry on next candle
        if self.entry_signal and not self.position:
            risk_amount = self.portfolio.equity * self.risk_per_trade
            atr_value = self.atr[-1]
            
            if atr_value == 0:
                return  # Avoid division by zero
            
            position_size = int(round(risk_amount / atr_value))
            if position_size > 0:
                self.buy(size=position_size, tag={'entry_price': self.data.Close[-1], 'atr': atr_value})
                print(f"🚀 LIFTOFF: {position_size} units @ {self.data.Close[-1]} | ATR: {atr_value:.2f}")
            self.entry_signal = False

        # 🔄 Exit Conditions
        for trade in self.trades:
            if trade.is_long:
                entry_price = trade.tag['entry_price']
                atr_entry = trade.tag['atr']
                
                # 🎯 Profit Target (2x ATR)
                if self.data.High[-1] >= entry_price + 2 * atr_entry:
                    trade.close()
                    print(f"🎯 MOON SHOT: +2ATR Target Reached (+{2*atr_entry:.2f})")
                
                # ⚠️ Funding Rate Reversion
                elif self.funding_rate[-1] >= 0:
                    trade.close()
                    print(f"⚡ FUNDING REVERTED: {self.funding_rate[-1]:.6f} (Exit @ {self.data.Close[-1]})")
                
                #