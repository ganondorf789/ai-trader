Here's the fixed code with all backtesting.lib imports removed and replaced with proper alternatives, along with Moon Dev themed debug prints:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION: VOLATILITYFLIP STRATEGY
# 🌌✨ Now with 100% backtesting.lib-free implementation!

# 1. NECESSARY IMPORTS
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

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
        # Custom resampling implementation to replace backtesting.lib.resample_apply
        def resample_ohlc(series, timeframe):
            """Moon Dev custom resampling function"""
            df = pd.DataFrame({
                'Open': series.Open,
                'High': series.High,
                'Low': series.Low,
                'Close': series.Close
            })
            resampled = df.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
            return resampled
        
        # Resample data to 4H timeframe
        resampled_data = resample_ohlc(self.data.df, '4H')
        high_low_range = resampled_data.High - resampled_data.Low
        chaikin_raw = talib.EMA(high_low_range, timeperiod=self.chaikin_period)
        
        # Reindex to original timeframe
        self.chaikin = self.I(lambda: np.repeat(chaikin_raw.values, 4)[:len(self.data)])
        self.chaikin_sma = self.I(talib.SMA, self.chaikin, timeperiod=self.sma_period)
        
        # 🚀 Bollinger Bands for exit signals
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, 
            nbdevdn=self.bb_dev
        )
        
        # 🌊 Swing High/Low for stop-loss
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # 📈 ATR for position sizing
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         timeperiod=14)

    def next(self):
        # 🌙✨ MOON DEV DEBUG CHECKS
        if len(self.data) < max(self.sma_period, self.bb_period, 14):
            print("🌙⏳ Not enough data points yet, waiting...")
            return
            
        current_chaikin = self.chaikin[-1]
        chaikin_sma = self.chaikin_sma[-1]
        funding_rate = self.data.Funding_Rate[-1]
        price = self.data.Close[-1]
        
        # 🌙 ENTRY LOGIC: Volatility expansion + negative funding
        if (not self.position and
            current_chaikin > chaikin_sma and 
            self.chaikin[-2] <= self.chaikin_sma[-2] and  # Bullish crossover check
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