Here's the complete backtesting implementation for the LiquidityBreakout strategy with Moon Dev themed debugging:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION - LIQUIDITY BREAKOUT STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

class LiquidityBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Clean data columns first 🌙
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        
        # 1. BOLLINGER BAND WIDTH CALCULATION ✨
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close, name='BB_Width')
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, 20, name='BB_Width_SMA')
        
        # 2. LIQUIDITY CLUSTER PROXIES (SWING HIGHS/LOWS) 🌊
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        
        # 3. VOLUME CONFIRMATION INDICATORS 📈
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='Volume_SMA')
        
        # 4. ATR FOR EXIT TARGETS 🎯
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED! Ready for liftoff! 🚀")

    def next(self):
        # Skip if position already open 🌙
        if self.position:
            return
            
        # Current market conditions 🌙
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        bbw = self.bb_width[-1]
        bbw_sma = self.bb_width_sma[-1]
        swing_h = self.swing_high[-1]
        swing_l = self.swing_low[-1]
        vol_sma = self.volume_sma[-1]
        atr = self.atr[-1]
        
        # 1. BOLLINGER BAND CONTRACTION CHECK 🌌
        if bbw < bbw_sma:
            # 2. LIQUIDITY CLUSTER BREAKOUT CHECK 🌊
            # LONG ENTRY: Break above swing high with volume surge 🌙
            if price > swing_h and volume > 1.5 * vol_sma:
                sl = swing_l  # Stop at swing low
                risk_amount = self.equity * self.risk_percent
                stop_distance = price - sl
                
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size > 0:
                        tp = price + 1.5 * atr  # Profit target 🎯
                        self.buy(size=position_size, sl=sl, tp=tp)
                        print(f"🌙🚀 BULLISH BREAKOUT! Buying {position_size} @ {price}")
                        print(f"   SL: {sl} | TP: {tp:.2f} | ATR: {atr:.2f} ✨")
            
            # SHORT ENTRY: Break below swing low with volume surge 🌙
            elif price < swing_l and volume > 1.5 * vol_sma:
                sl = swing_h  # Stop at swing high
                risk_amount = self.equity * self.risk_percent
                stop_distance = sl - price
                
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size > 0:
                        tp = price - 1.5 * atr  # Profit target 🎯
                        self.sell(size=position_size, sl=sl, tp=tp)
                        print(f"🌙💥 BEARISH BREAKDOWN! Selling {position_size} @ {price}")
                        print(f"   SL: {sl} | TP: {tp