I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed improvements:

```python
# moon_dev_vol_compression_strategy.py
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class VolCompressionDivergence(Strategy):
    def init(self):
        # 🌙✨ Initialize indicators with Moon Dev precision
        # Bollinger Bands (20,2)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                             timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Bollinger Band Width calculations
        bb_width = (self.bb_upper - self.bb_lower)/self.bb_middle
        self.bb_width_percentile = self.I(lambda x: x.rolling(14).rank(pct=True).values, bb_width)
        
        # Put/Call Ratio 5-day MA (480 periods for 15m data)
        self.put_call_ratio = self.data.df['put_call_ratio']
        self.put_call_ma = self.I(talib.SMA, self.put_call_ratio, timeperiod=480)
        
        # ATR for risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=5)
        
        # Prior day's high/low (96 periods = 24hrs in 15m)
        self.prior_day_high = self.I(talib.MAX, self.data.High, timeperiod=96)
        self.prior_day_low = self.I(talib.MIN, self.data.Low, timeperiod=96)
        
        print("🌙✨ MOON DEV STRATEGY INITIALIZED! Ready for cosmic profits! 🚀")

    def next(self):
        # 🌙 Cosmic debug prints every 100 bars
        if len(self.data) % 100 == 0:
            print(f"🌙 MOON DEV DEBUG: Bar {len(self.data)} | Close: {self.data.Close[-1]:.2f} | BB%: {self.bb_width_percentile[-1]*100:.1f}%")

        # Entry logic
        if not self.position:
            # Check volatility compression
            if self.bb_width_percentile[-1] < 0.1:
                # Check PCR surge
                pcr = self.put_call_ratio
                if len(pcr) >= 5:
                    current_pcr = pcr[-1]
                    pcr_5_ago = pcr[-5]
                    
                    # Bearish surge check
                    if current_pcr >= pcr_5_ago * 1.2:
                        if self.data.Low[-1] < self.prior_day_low[-1]:
                            self.enter_short()
                            
                    # Bullish surge check    
                    elif current_pcr <= pcr_5_ago * 0.8:
                        if self.data.High[-1] > self.prior_day_high[-1]:
                            self.enter_long()

        # Exit logic
        else:
            self.manage_exits()

    def enter_long(self):
        # 🌕 Cosmic long entry with proper risk management
        atr = self.atr[-1]
        if atr == 0: 
            print("🌑 MOON DEV WARNING: Zero ATR detected, skipping trade")
            return
        
        risk_amount = 0.02 * self.equity
        position_size = int(round(risk_amount / atr))
        entry_price = self.data.Close[-1]
        
        if position_size > 0:
            self.buy(size=position_size, 
                    sl=entry_price - atr,
                    tp=entry_price + 2*atr,
                    tag={'entry_price': entry_price, 'atr': atr, 'entry_bar': len(self.data)})
            print(f"🚀🌕 MOON DEV LONG! Size: {position_size} | Entry: {entry_price:.2f} | TP: {entry_price+2*atr:.2f} | SL: {entry_price-atr:.2f}")

    def enter_short(self):
        # 🌑 Galactic short entry with proper risk management
        atr = self.atr[-1]
        if atr == 0: 
            print("🌑 MOON DEV WARNING: Zero ATR detected, skipping trade")
            return
        
        risk_amount = 0.02 * self.equity
        position_size = int(round(risk_amount / atr))
        entry_price = self.data