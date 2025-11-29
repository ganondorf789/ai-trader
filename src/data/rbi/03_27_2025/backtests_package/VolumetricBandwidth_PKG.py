I'll fix the code by removing all `backtesting.lib` imports and replacing any usage of `crossover` or `crossunder` with manual comparisons. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VolumetricBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_daily_loss = 0.03  # 3% maximum daily loss (not implemented in this version)
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators with self.I()
        close = self.data.Close
        volume = self.data.Volume
        
        # Volume-Weighted RSI (14-period)
        self.vw_rsi = self.I(ta.vw_rsi, close, volume, length=14)
        
        # Bollinger Bands (20, 2)
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Bollinger Bandwidth
        self.bandwidth = self.I(lambda u, m, l: (u - l) / m, self.upper, self.middle, self.lower)
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, timeperiod=20)
        
        # Volume MA (50-period)
        self.volume_ma = self.I(talib.SMA, volume, timeperiod=50)
        
        print("🌙✨ Moon Dev Indicators Initialized Successfully! ✨🌙")
        
    def next(self):
        if len(self.data) < 50:  # Ensure enough data for calculations
            return
            
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_volume = self.data.Volume[-1]
        
        # Common conditions
        bandwidth_contraction = self.bandwidth[-1] < self.bandwidth_ma[-1]
        volume_trigger = current_volume > self.volume_ma[-1]
        
        # Long entry conditions
        if (not self.position and
            bandwidth_contraction and
            current_close <= self.lower[-1] and
            volume_trigger and
            (current_low < self.data.Low[-2] and self.vw_rsi[-1] > self.vw_rsi[-2])):
            
            bb_width = self.upper[-1] - self.lower[-1]
            stop_loss_price = current_close - 1.5 * bb_width
            risk_amount = self.equity * self.risk_per_trade
            risk_per_share = abs(current_close - stop_loss_price)
            
            if risk_per_share == 0:
                print("🌙⚠️ Warning: Zero risk detected - skipping trade")
                return  # Avoid division by zero
            
            position_size = int(round(risk_amount / risk_per_share))
            if position_size > 0:
                self.buy(size=position_size, tag={
                    'stop_loss': stop_loss_price,
                    'entry_type': 'long'
                })
                print(f"🌙✨ BULLISH ENTRY: {current_close:.2f} | Size: {position_size} 🚀")
                print(f"🌙 SL Price: {stop_loss_price:.2f} | Risk: {self.risk_per_trade*100:.1f}%")
        
        # Short entry conditions
        if (not self.position and
            bandwidth_contraction and
            current_close >= self.upper[-1] and
            volume_trigger and
            (current_high > self.data.High[-2] and self.vw_rsi[-1] < self.vw_rsi[-2])):
            
            bb_width = self.upper[-1] - self.lower[-1]
            stop_loss_price = current_close + 1.5 * bb_width
            risk_amount = self.equity * self.risk_per_trade
            risk_per_share = abs(current_close - stop_loss_price)
            
            if risk_per_share == 0:
                print("🌙⚠️ Warning: Zero risk detected - skipping trade")
                return  # Avoid division by zero
            
            position_size = int(round(risk_amount / risk_per_share))
            if position_size