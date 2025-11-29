```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLCLUSTERBREACH STRATEGY 🚀

import pandas as pd
import numpy as np
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolClusterBreach(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    bb_period = 20
    bb_dev = 2
    swing_period = 20
    atr_period = 14

    def init(self):
        # 🌈 INDICATOR INITIALIZATION SECTION ✨
        # Bollinger Bands using pandas_ta
        def bb_upper(close): 
            return pta.bbands(close, length=self.bb_period, std=self.bb_dev)['BBU_'+str(self.bb_period)+'_'+str(self.bb_dev)+'.0']
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_UPPER')
        
        def bb_lower(close):
            return pta.bbands(close, length=self.bb_period, std=self.bb_dev)['BBL_'+str(self.bb_period)+'_'+str(self.bb_dev)+'.0']
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_LOWER')
        
        # Bollinger Band Width calculation
        self.bbw = self.I(lambda u,l: (u-l)/((u+l)/2), 
                         self.bb_upper, self.bb_lower, name='BB_WIDTH')
        
        # Volatility and Swing Levels
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='SWING_LOW')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # Liquidation Zones (from on-chain data)
        self.liquid_high = self.I(lambda: self.data.df['liquidation_high'], name='LIQUID_HIGH')
        self.liquid_low = self.I(lambda: self.data.df['liquidation_low'], name='LIQUID_LOW')

    def next(self):
        current_close = self.data.Close[-1]
        liquid_high = self.liquid_high[-1]
        liquid_low = self.liquid_low[-1]
        
        # 🚨 TRIGGER CONDITIONS CHECK
        if self.position:
            return  # Hold existing position

        # 🌟 LONG ENTRY LOGIC 🌙
        if (self._recent_breach(liquid_high, 'above') and 
            self._bbw_contraction() and 
            self._pullback_to_zone(liquid_high) and 
            self._bullish_confirmation()):
            
            # 🛡 RISK MANAGEMENT CALCULATIONS
            sl_price = min(self.swing_low[-1], current_close - 1.5*self.atr[-1])
            risk_amount = self.equity * self.risk_pct
            risk_per_share = current_close - sl_price
            
            if risk_per_share > 0:
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price,
                            tags=f"🌕 MOON ENTRY @ {current_close:.2f}")
                    print(f"🚀🌙 MOON BULLISH BREACH 🌙🚀\nEntry: {current_close:.2f} | Size: {position_size} | SL: {sl_price:.2f}\n")

        # 🌟 SHORT ENTRY LOGIC 🌙 (mirror logic omitted for brevity)

    def _recent_breach(self, level, direction):
        # Check 5-period breach history
        lookback = 5
        closes = self.data.Close[-lookback:-1]
        if direction == 'above':
            return any(c > level for c in closes)
        return any(c < level for c in closes)

    def _bbw_contraction(self):
        # BBW must be below 0.8*max of last 20 periods
        lookback = 20
        current_bbw = self.bbw[-1]
        max_bbw = max(self.bbw[-lookback:])
        return current_bbw <