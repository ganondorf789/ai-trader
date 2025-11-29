Here's the fully debugged and Moon Dev themed version of your SqueezeReversion strategy:

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
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda: talib.BBANDS(self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0),
            name=['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
        )
        
        # ✨ Calculate BB Width dynamically
        self.bb_width = self.I(
            lambda: ((self.bb_upper - self.bb_lower) / self.bb_middle),
            name='BB_WIDTH'
        )
        
        # 📉 Funding Rate analysis
        self.funding_rate = self.I(
            lambda: self.data.df['fundingrate'],
            name='FUNDING_RATE'
        )
        self.funding_low = self.I(
            lambda: talib.MIN(self.funding_rate, timeperiod=self.lookback_period),
            name='FUNDING_LOW'
        )
        
        # 📈 ATR for risk management
        self.atr = self.I(
            lambda: talib.ATR(self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period),
            name='ATR'
        )
        
        self.entry_signal = False  # 🌙 Signal tracking
        
    def next(self):
        # 🚀 MOON DEV CORE LOGIC
        if len(self.data) < self.lookback_period:
            return
            
        # Current market state
        current_bb_width = ((self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1])
        bb_squeeze = current_bb_width < pd.Series(self.bb_width[-self.lookback_period:]).quantile(0.05)
        funding_drop = self.funding_rate[-1] < self.funding_low[-1]
        
        # 🌋 Entry Condition Check
        if bb_squeeze and funding_drop and not self.position:
            self.entry_signal = True
            print(f"🌋 MOON ERUPTION SIGNAL | Price: {self.data.Close[-1]:.2f} | BB Width: {current_bb_width:.4f} | Funding: {self.funding_rate[-1]:.6f}")

        # 🚀 Execute Entry on next candle
        if self.entry_signal and not self.position:
            risk_amount = self.equity * self.risk_per_trade
            atr_value = self.atr[-1]
            
            if atr_value == 0:
                return  # Avoid division by zero
            
            position_size = int(round(risk_amount / atr_value))
            if position_size > 0:
                self.buy(
                    size=position_size,
                    sl=self.data.Close[-1] - atr_value,  # 🌙 1x ATR stop loss
                    tag={
                        'entry_price': self.data.Close[-1],
                        'atr': atr_value,
                        'target': self.data.Close[-1] + 2 * atr_value
                    }
                )
                print(f"🚀 LIFTOFF | Size: {position_size} units | Entry: {self.data.Close[-1]:.2f} | ATR: {atr_value:.2f} | Target: {self.data.Close[-1] + 2*atr_value:.2f}")
            self.entry_signal = False

        # 🔄 Exit Conditions
        for trade in self.trades:
            if trade.is_long:
                # 🎯 Profit Target (2x ATR)
                if self.data.Close[-1] >= trade.tag['target']:
                    trade.close()
                    print(f"🎯 MOON SHOT | Profit: {trade.pl_pct:.2f}% | Price: {self.data.Close[-1]:.2f}")
                
                # ⚠️ Funding Rate Reversion
                elif self.funding_rate[-1] >= 0:
                    trade.close()
                    print(f"⚡ FUNDING REVERT