```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class MomentumCrossfire(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_exposure = 0.05    # 5% maximum portfolio exposure
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I()
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA 50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA 200')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ADX 14')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        
        print("🌙✨ Momentum Crossfire Strategy Initialized! Ready to launch rockets! 🚀")

    def next(self):
        current_idx = len(self.data.Close) - 1
        if current_idx < 1:
            return

        # Get indicator values
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        prev_ema50 = self.ema50[-2]
        prev_ema200 = self.ema200[-2]
        adx = self.adx[-1]
        prev_adx = self.adx[-2]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        # Moon Dev debug prints
        print(f"🌙 EMA50: {ema50:.2f} | EMA200: {ema200:.2f} | ADX: {adx:.2f}")

        # Entry logic
        if not self.position:
            # Long entry conditions
            if (ema50 > ema200 and prev_ema50 <= prev_ema200 and
                adx > 25 and adx > prev_adx and
                self.data.Close[-1] > swing_high):
                
                self.enter_long()
            
            # Short entry conditions
            elif (ema50 < ema200 and prev_ema50 >= prev_ema200 and
                  adx > 25 and adx > prev_adx and
                  self.data.Close[-1] < swing_low):
                
                self.enter_short()

        # Exit logic
        else:
            # Emergency exit if ADX drops below 20
            if self.adx[-1] < 20:
                self.position.close()
                print(f"🌪️🌙 EMERGENCY EXIT! ADX collapsed to {self.adx[-1]:.2f}")

    def enter_long(self):
        entry_price = self.data.Close[-1]
        stop_loss = self.swing_low[-1]
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            print("🌙💥 Aborted Long: Risk per share <= 0")
            return
        
        # Position sizing calculations
        risk_amount = self.risk_per_trade * self.equity
        position_size = int(round(risk_amount / risk_per_share))
        
        # Check maximum exposure
        max_size = int((self.max_exposure * self.equity) // entry_price)
        position_size = min(position_size, max_size)
        
        if position_size > 0:
            tp_price = entry_price + 2 * risk_per_share
            self.buy(size=position_size, sl=stop_loss, tp=tp_price)
            print(f"🚀🌙 BLASTOFF LONG! Size: {position_size} | Entry: {entry_price:.2f}")
            print(f"   🛡️ SL: {stop_loss:.2f} | 🎯 TP: {tp_price:.2f}")

    def enter_short(self):
        entry_price = self.data.Close[-1]
        stop_loss = self.swing_high[-1]
        risk_per_share = stop_loss - entry_price
        
        if risk_per_share <= 0:
            print("🌙💥 Ab