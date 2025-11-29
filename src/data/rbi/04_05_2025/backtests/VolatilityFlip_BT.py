Here's the implementation of the VolatilityFlip strategy for backtesting.py with Moon Dev themed debug prints and proper risk management:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION: VOLATILITYFLIP STRATEGY

# 1. NECESSARY IMPORTS
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

# 2. STRATEGY CLASS WITH INDICATORS
class VolatilityFlip(Strategy):
    # Strategy parameters
    risk_pct = 0.01  # 1% risk per trade
    chaikin_period = 10
    sma_period = 20
    bb_period = 20
    bb_dev = 2
    swing_period = 20
    
    def init(self):
        # 🌙 Chaikin Volatility Calculation (4H resampled)
        def calculate_chaikin(df):
            high_low_range = df.High - df.Low
            return talib.EMA(high_low_range, timeperiod=self.chaikin_period)
        
        # Resample to 4H timeframe for volatility calculation
        self.chaikin = resample_apply('4H', calculate_chaikin, self.data)
        self.chaikin_sma = self.I(talib.SMA, self.chaikin, self.sma_period)
        
        # 🚀 Bollinger Bands for exit signals
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, 
            nbdevdn=self.bb_dev
        )
        
        # 🌊 Swing High/Low for stop-loss
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # 📈 ATR for position sizing
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         timeperiod=14)

    def next(self):
        # 🌙✨ MOON DEV DEBUG CHECKS
        if len(self.data) < max(self.sma_period, self.bb_period, 14):
            return
            
        current_chaikin = self.chaikin[-1]
        chaikin_sma = self.chaikin_sma[-1]
        funding_rate = self.data.Funding_Rate[-1]
        price = self.data.Close[-1]
        
        # 🌙 ENTRY LOGIC: Volatility expansion + negative funding
        if (not self.position and
            current_chaikin > chaikin_sma and 
            self.chaikin[-2] <= self.chaikin_sma[-2] and
            funding_rate < 0):
            
            # 🛑 Risk Management Calculations
            sl_long = self.swing_low[-1]
            sl_short = self.swing_high[-1]
            
            # Determine trend direction using Bollinger Bands
            if price > self.bb_middle[-1]:
                direction = 'LONG'
                risk_amount = price - sl_long
                stop_price = sl_long
            else:
                direction = 'SHORT'
                risk_amount = sl_short - price
                stop_price = sl_short
                
            if risk_amount <= 0:
                print("🌙❌ Invalid risk amount, skipping trade")
                return
                
            # Calculate position size (volatility-adjusted)
            position_size = (self.risk_pct * self.equity) / risk_amount
            position_size = int(round(position_size))
            
            if direction == 'LONG':
                print(f"🌙🚀 ENTERING LONG | Size: {position_size} | Price: {price:.2f}")
                self.buy(size=position_size, sl=stop_price)
            else:
                print(f"🌙🚀 ENTERING SHORT | Size: {position_size} | Price: {price:.2f}")
                self.sell(size=position_size, sl=stop_price)

        # 🌙 EXIT LOGIC: Bollinger Band extremes
        for trade in self.trades:
            if trade.is_long and price >= self.bb_upper[-1]:
                print(f"🌙💎 EXITING LONG at BB Upper: {self.bb_upper[-1]:.2f}")
                trade.close()
            elif trade.is_short and price <= self.bb_lower[-1]:
                print(f"🌙💎 EXITING SHORT at BB