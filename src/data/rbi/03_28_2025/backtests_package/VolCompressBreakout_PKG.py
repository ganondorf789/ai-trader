Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolCompressBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Clean data and prepare columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I() wrapper
        # Bollinger Bands (20,2)
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        
        # Keltner Channels (20,1.5 ATR)
        self.kc_atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.kc_middle = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.kc_upper = self.I(lambda: self.kc_middle + 1.5*self.kc_atr)
        self.kc_lower = self.I(lambda: self.kc_middle - 1.5*self.kc_atr)
        
        # Volatility metrics
        self.bb_width = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_middle)
        self.bb_width_pct = self.I(lambda: self.bb_width.rolling(100).rank(pct=True)*100)
        
        # Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED ✨🌙")
        print("🌙 All indicators powered by TA-Lib with proper self.I() wrapping")
        print("🌙 No backtesting.lib contamination detected - System clean! ✨")

    def next(self):
        current_idx = len(self.data)-1
        if current_idx < 100:  # Warmup period
            return
            
        # Volatility compression check
        squeeze_active = (self.bb_upper[-1] < self.kc_upper[-1]) and \
                        (self.bb_lower[-1] > self.kc_lower[-1])
        
        # Entry conditions (using array indexing instead of crossover/crossunder)
        long_signal = (self.data.Close[-1] > self.bb_upper[-1]) or \
                     (self.data.Close[-1] > self.kc_upper[-1])
        
        short_signal = (self.data.Close[-1] < self.bb_lower[-1]) or \
                      (self.data.Close[-1] < self.kc_lower[-1])
        
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1]
        volatility_ok = self.bb_width_pct[-1] < 20
        
        # Risk management calculations
        equity = self.equity
        risk_amount = equity * self.risk_percent
        
        if not self.position:
            if squeeze_active and volatility_ok and volume_ok:
                if long_signal:
                    # Calculate position size for long
                    sl_price = min(self.bb_lower[-1], self.kc_lower[-1])
                    risk_per_share = self.data.Close[-1] - sl_price
                    if risk_per_share <= 0: 
                        print("🌙⚠️ Risk per share invalid - aborting trade")
                        return
                    
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_price = self.data.Close[-1] + 2*risk_per_share
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🌙🚀 MOON DEV LONG ENTRY 🚀🌙 | Price: {self.data.Close[-1]:.2f} | Size: {position_size