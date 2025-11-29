```python
# 🌙 MOON DEV BACKTESTING SUITE - SQUEEZEFADE STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class SqueezeFade(Strategy):
    # Strategy parameters
    squeeze_threshold = 0.03  # 3% bandwidth contraction
    risk_percentage = 0.01    # 1% risk per trade
    atr_multiplier = 1.5      # Stop loss multiplier
    rr_ratio = 3              # Risk-reward ratio
    
    def init(self):
        # 🌌 CORE INDICATORS USING TALIB
        # Bollinger Bands
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_middle = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Volatility and liquidity markers
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("✨ Strategy initialized with Moon Dev precision! 🌕")

    def next(self):
        # Skip early bars without indicator data
        if len(self.data) < 20:
            return

        # 🌗 CALCULATE CURRENT CONDITIONS
        current_bandwidth = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]
        squeeze_active = current_bandwidth < self.squeeze_threshold
        volume_surge = self.data.Volume[-1] > 2 * self.volume_sma[-1]
        
        # Liquidity cluster detection
        near_swing_high = self.data.Close[-1] >= self.swing_high[-1]
        near_swing_low = self.data.Close[-1] <= self.swing_low[-1]
        
        # 🌑 ENTRY LOGIC
        if not self.position:
            # Short entry: breakout above upper band into liquidity cluster
            if squeeze_active and volume_surge and near_swing_high and self.data.Close[-1] > self.bb_upper[-1]:
                self.enter_short()
            
            # Long entry: breakdown below lower band into liquidity cluster
            elif squeeze_active and volume_surge and near_swing_low and self.data.Close[-1] < self.bb_lower[-1]:
                self.enter_long()
        
        # 🛑 EARLY EXIT LOGIC
        else:
            if self.position.is_short and self.data.Close[-1] < self.bb_middle[-1]:
                self.position.close()
                print(f"🌑 Closing short early! Price reclaimed squeeze zone at {self.data.Close[-1]} ✨")
            elif self.position.is_long and self.data.Close[-1] > self.bb_middle[-1]:
                self.position.close()
                print(f"🌕 Closing long early! Price reclaimed squeeze zone at {self.data.Close[-1]} 🚀")

    def enter_short(self):
        entry_price = self.data.Close[-1]
        atr_value = self.atr[-1]
        
        # RISK MANAGEMENT CALCULATIONS 🌙
        stop_loss = entry_price + self.atr_multiplier * atr_value
        take_profit = entry_price - self.rr_ratio * self.atr_multiplier * atr_value
        risk_amount = self.equity * self.risk_percentage
        position_size = int(round(risk_amount / (self.atr_multiplier * atr_value)))
        
        if position_size > 0:
            self.sell(size=position_size, sl=stop_loss, tp=take_profit)
            print(f"🌙✨ SHORTING GALACTIC RALLY! Entry: {entry_price}, Size: {position_size} BTC")
            print(f