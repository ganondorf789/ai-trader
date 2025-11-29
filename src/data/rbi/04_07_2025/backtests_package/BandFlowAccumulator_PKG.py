Here's the fixed code with all backtesting.lib references removed and replaced with proper implementations:

```python
# 🌙 Moon Dev's BandFlow Accumulator Backtest 🌙
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy

class BandFlowAccumulator(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Indicator Calculation Phase
        print("🌙 Initializing Moon Dev's Quantum Indicators...")
        
        # Bollinger Bands Components
        self.middle = self.I(talib.SMA, self.data.Close, timeperiod=20, name='Middle BB')
        self.std = self.I(talib.STDDEV, self.data.Close, timeperiod=20, name='STDDEV 20')
        self.upper = self.I(lambda c, m, s: m + 2*s, self.data.Close, self.middle, self.std, name='Upper BB')
        self.lower = self.I(lambda c, m, s: m - 2*s, self.data.Close, self.middle, self.std, name='Lower BB')
        
        # Chaikin Money Flow
        self.cmf = self.I(ta.cmf, self.data.High, self.data.Low, self.data.Close, self.data.Volume, length=21, name='CMF 21')
        
        # Trend Filter (200-period SMA)
        self.ma_200 = self.I(talib.SMA, self.data.Close, timeperiod=200, name='SMA 200')
        
        # Swing Low Detection
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        
        # Volatility Measurement
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR 14')
        
        print("✨ Cosmic indicators aligned! Ready for launch 🚀")

    def next(self):
        # 🌙 Debugging Pulse
        print(f"\n🌙 Candle: {self.data.index[-1]} | Close: {self.data.Close[-1]:.2f} | Lower BB: {self.lower[-1]:.2f} | CMF: {self.cmf[-1]:.4f}")
        
        if not self.position:
            # 🚀 Entry Logic
            bb_touch = self.data.Close[-1] <= self.lower[-1]
            cmf_cross = (self.cmf[-2] < 0) and (self.cmf[-1] > 0)  # Bullish crossover replacement
            trend_up = self.data.Close[-1] > self.ma_200[-1]
            
            if bb_touch and cmf_cross and trend_up:
                entry_price = self.data.Close[-1]
                stop_loss = self.swing_low[-1]
                risk_per_share = entry_price - stop_loss
                atr_value = self.atr[-1]
                
                # 🛡️ Risk Validation
                if risk_per_share <= 0:
                    print(f"🚫 Aborted: Stop loss {stop_loss:.2f} above entry {entry_price:.2f}")
                    return
                    
                if risk_per_share > 2 * atr_value:
                    print(f"🚫 Aborted: Risk {risk_per_share:.2f} > 2xATR ({2*atr_value:.2f})")
                    return
                
                # 💰 Position Sizing
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size < 1:
                    print(f"🚫 Aborted: Micro position {position_size} shares")
                    return
                
                # 🚀 Execute Trade
                print(f"\n🔥🌕 COSMIC BUY SIGNAL! 🔥")
                print(f"Entry: {entry_price:.2f} | SL: {stop_loss:.2f}")
                print(f"TP: {self.middle[-1]:.2f} | Size: {position_size} shares")
                self.buy(size=position_size, sl=stop_loss, tp=self.middle[-1])
        
        else:
            # 🌧️ Exit Conditions
            if (self.cmf[-2] > 0) and (self.cmf[-1] < 0):  # Bearish crossover replacement
                print(f"\n🌧️ CMF DARKENING! Closing position at {self.data.Close[-1]:.2f}")
                self.position.close()

# 🛠️ Data Preparation Ritual
data_path =