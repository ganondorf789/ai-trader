```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class LiquiditySqueeze(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    
    def init(self):
        # Clean data and prepare columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I()
        close = self.data.Close
        
        # Bollinger Bands (20,2)
        self.bb_upper, self.bb_middle, self.bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.bb_upper = self.I(self.bb_upper, name='UPPER_BB')
        self.bb_lower = self.I(self.bb_lower, name='LOWER_BB')
        
        # Volatility metrics
        bandwidth = (self.bb_upper - self.bb_lower)/self.bb_middle*100
        self.bandwidth = self.I(bandwidth, name='BANDWIDTH')
        self.bandwidth_ma = self.I(talib.SMA, bandwidth, timeperiod=20, name='BANDWIDTH_MA')
        
        # Liquidity clusters (simulated with swing highs and volume)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING_LOW')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=50, name='VOLUME_MA')
        
        # Momentum filter
        self.rsi = self.I(talib.RSI, close, timeperiod=14, name='RSI')
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED! READY FOR LIFTOFF! 🚀")

    def next(self):
        # Skip weekends/holidays (low liquidity filter)
        if pd.Timestamp(self.data.index[-1]).weekday() >= 5:
            return
        
        # Wait for all indicators to warm up
        if len(self.data) < 50:
            return
        
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        equity = self.equity
        
        # 1. BOLLINGER SQUEEZE CONDITION
        squeeze_active = (self.bandwidth[-1] < self.bandwidth_ma[-1] * 0.5)
        
        # 2. LIQUIDITY CLUSTER PROXIMITY
        near_swing_high = self.data.High[-1] >= self.swing_high[-1] * 0.995
        near_swing_low = self.data.Low[-1] <= self.swing_low[-1] * 1.005
        volume_spike = current_volume > self.volume_ma[-1] * 1.5
        
        # LONG ENTRY: Breakout above upper BB with confirmation
        if (not self.position and
            squeeze_active and
            current_close > self.bb_upper[-1] and
            near_swing_high and
            volume_spike and
            self.rsi[-1] < 70):
            
            stop_price = self.bb_lower[-1]
            risk_distance = current_close - stop_price
            position_size = int(round((equity * self.risk_per_trade) / risk_distance))
            
            if position_size > 0:
                tp_price = current_close + 2*risk_distance  # 2:1 reward
                self.buy(size=position_size, 
                        sl=stop_price,
                        tp=tp_price)
                print(f"🌙🚀 BULLISH BREAKOUT! Long {position_size} units at {current_close:.2f}")
                print(f"   MOON SL: {stop_price:.2f} | GALACTIC TP: {tp_price:.2f} ✨")

        # SHORT ENTRY: Breakdown below lower BB with confirmation    
        elif (not self.position and
              squeeze_active and
              current_close < self.bb_lower[-1] and
              near_swing_low and
              volume_spike and
              self.rsi[-1] > 30):