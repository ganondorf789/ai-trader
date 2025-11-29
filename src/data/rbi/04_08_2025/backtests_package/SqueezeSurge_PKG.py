Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class SqueezeSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I()
        # Bollinger Bands (20, 2)
        self.upper_band = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.middle_band = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # Bandwidth and squeeze detection
        self.bandwidth = self.I(lambda u, l: u - l, self.upper_band, self.lower_band)
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, timeperiod=20)
        
        # Volume indicators
        self.volume_avg = self.I(talib.SMA, self.data.Volume, 10)
        
        # RSI for exits
        self.rsi = self.I(talib.RSI, self.data.Close, 5)
        
        print("🌙✨ Moon Dev Indicators Activated! Ready for cosmic profits! 🚀")

    def next(self):
        i = len(self.data) - 1
        
        # Wait for all indicators to be ready
        if i < 20 or i < 10 or i < 5:
            return
        
        # Calculate current conditions
        squeeze = self.bandwidth[i] <= self.min_bandwidth[i]
        volume_surge = self.data.Volume[i] > 1.5 * self.volume_avg[i-1]
        current_close = self.data.Close[i]
        
        # Moon Dev Debug Prints 🌙
        print(f"🌙 Bar {i} | Close: {current_close:.2f} | Bandwidth: {self.bandwidth[i]:.2f} | Min BW: {self.min_bandwidth[i]:.2f} | RSI: {self.rsi[i]:.2f}")

        # Entry Logic
        if not self.position:
            if squeeze and volume_surge:
                # Long Entry
                if current_close > self.upper_band[i]:
                    risk_amount = self.equity * self.risk_percent
                    stop_price = self.middle_band[i]
                    risk_per_share = current_close - stop_price
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount / risk_per_share))
                        if position_size > 0:
                            self.buy(size=position_size, sl=stop_price)
                            print(f"🌙🚀 COSMIC LONG SURGE! Entry: {current_close:.2f} | Size: {position_size} | SL: {stop_price:.2f} | Equity: {self.equity:.2f}")

                # Short Entry        
                elif current_close < self.lower_band[i]:
                    risk_amount = self.equity * self.risk_percent
                    stop_price = self.middle_band[i]
                    risk_per_share = stop_price - current_close
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount / risk_per_share))
                        if position_size > 0:
                            self.sell(size=position_size, sl=stop_price)
                            print(f"🌙🌌 GALACTIC SHORT SQUEEZE! Entry: {current_close:.2f} | Size: {position_size} | SL: {stop_price:.2f} | Equity: {self.equity:.2f}")

        # Exit Logic
        else:
            current_rsi = self.rsi[i]
            prev_rsi = self.rsi[i-1]
            
            # Long exit condition (replaced crossunder with direct comparison)
            if self.position.is_long and prev_rsi >= 70 and current_rsi < 70: