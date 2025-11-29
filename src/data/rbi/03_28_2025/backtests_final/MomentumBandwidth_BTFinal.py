I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
# 🌙 Moon Dev's MomentumBandwidth Backtest Implementation
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class MomentumBandwidth(Strategy):
    risk_per_trade = 0.01  # Risk 1% of equity per trade
    cmo_threshold_long = 50
    cmo_threshold_short = -50
    cmo_exit_long = 20
    cmo_exit_short = -20
    bandwidth_lookback = 20
    atr_multiplier = 1.5
    
    def init(self):
        # Preprocess data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I()
        # Chande Momentum Oscillator (CMO)
        self.cmo = self.I(lambda close: ta.cmo(close, length=14), self.data.Close)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.data.Close, 
                                           timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_band = self.I(lambda x: upper, name='UPPER')
        self.middle_band = self.I(lambda x: middle, name='MIDDLE')
        self.lower_band = self.I(lambda x: lower, name='LOWER')
        
        # Bollinger Bandwidth
        bandwidth = (upper - lower) / middle
        self.bandwidth = self.I(lambda x: bandwidth, name='BANDWIDTH')
        
        # 20-day Low in Bandwidth
        self.bandwidth_low = self.I(talib.MIN, self.bandwidth, timeperiod=20, name='BANDWIDTH_LOW')
        
        # Bandwidth Moving Average
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, timeperiod=20, name='BANDWIDTH_MA')
        
        # ATR for stop loss
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=14, name='ATR')
        
        print("🌙 Lunar Indicators Activated! Ready for launch 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        current_cmo = self.cmo[-1]
        prev_cmo = self.cmo[-2] if len(self.cmo) > 1 else None
        
        # Moon Dev Debug Prints
        if len(self.data) % 100 == 0:
            print(f"🌕 Moon Phase Update | Candle: {len(self.data)} | CMO: {current_cmo:.1f} | Bandwidth: {self.bandwidth[-1]:.4f}")

        # Trade Entry/Exit Logic
        if not self.position:
            # Long Entry Condition
            if (prev_cmo is not None and 
                prev_cmo < self.cmo_threshold_long and 
                current_cmo >= self.cmo_threshold_long and
                self.bandwidth[-1] <= self.bandwidth_low[-1]):
                
                atr_value = self.atr[-1]
                stop_loss = current_close - (self.atr_multiplier * atr_value)
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / (current_close - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss, 
                            tag="🌙 LONG: CMO Breakout + Vol Squeeze")
                    print(f"🚀 LIFT OFF! Long Entry @ {current_close:.2f} | Size: {position_size} | SL: {stop_loss:.2f}")

            # Short Entry Condition
            elif (prev_cmo is not None and
                  prev_cmo > self.cmo_threshold_short and 
                  current_cmo <= self.cmo_threshold_short and
                  self.bandwidth[-1] <= self.bandwidth_low[-1]):
                
                atr_value = self.atr[-1]
                stop_loss = current_close + (self.atr_multiplier * atr_value)
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / (stop_loss - current_close