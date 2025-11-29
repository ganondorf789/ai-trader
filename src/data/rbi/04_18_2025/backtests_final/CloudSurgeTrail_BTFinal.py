Here's the complete fixed code with Moon Dev themed improvements and proper position sizing:

```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# Clean and prepare data with Moon Dev sparkles ✨
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
print("🌙 Moon Dev: Data loaded and cleaned successfully!")

class CloudSurgeTrail(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 22
    chandelier_multiplier = 3
    
    def init(self):
        # Calculate Ichimoku components with Moon precision 🌙
        self.ichimoku = self.I(ta.ichimoku, 
                             high=self.data.High, 
                             low=self.data.Low, 
                             close=self.data.Close,
                             tenkan=9, kijun=26, senkou=52)
        
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Momentum filter
        self.stoch_k, self.stoch_d = self.I(talib.STOCHRSI, 
                                          self.data.Close, 
                                          timeperiod=14,
                                          fastk_period=3,
                                          fastd_period=3)
        
        # Volatility measure
        self.atr = self.I(talib.ATR,
                         self.data.High,
                         self.data.Low,
                         self.data.Close,
                         timeperiod=self.atr_period)
        
        # Risk management variables
        self.consecutive_losses = 0
        self.daily_equity = self.equity
        self.last_day = None
        print("✨ Moon Dev: Indicators initialized successfully!")

    def enter_long(self, sl):
        """Moon Dev enhanced long entry with proper position sizing"""
        risk_amount = self.equity * self.risk_per_trade
        size = round(risk_amount / (self.data.Close[-1] - sl), 2)  # Proper rounding
        self.buy(sl=sl, size=min(size, 0.99))  # Cap at 99% of equity
        print(f"🚀 Moon Dev Long Entry: {self.data.index[-1]} | Size: {size:.2f}")

    def enter_short(self, sl):
        """Moon Dev enhanced short entry with proper position sizing"""
        risk_amount = self.equity * self.risk_per_trade
        size = round(risk_amount / (sl - self.data.Close[-1]), 2)  # Proper rounding
        self.sell(sl=sl, size=min(size, 0.99))  # Cap at 99% of equity
        print(f"🌑 Moon Dev Short Entry: {self.data.index[-1]} | Size: {size:.2f}")

    def next(self):
        # Moon Dev risk checks 🌙
        current_day = self.data.index[-1].date()
        if current_day != self.last_day:
            self.daily_equity = self.equity
            self.last_day = current_day
            
        if (self.daily_equity - self.equity) / self.daily_equity > 0.05:
            print("🌧️ Moon Dev Alert: Daily loss limit triggered!")
            return
            
        if self.consecutive_losses >= 3:
            print("🌧️ Moon Dev Alert: 3 consecutive losses!")
            return
            
        price = self.data.Close[-1]
        ichi = self.ichimoku
        
        # Cloud components ✨
        senkou_a = ichi['ISA_9'][-1]
        senkou_b = ichi['ISB_26'][-1]
        cloud_bullish = senkou_a > senkou_b
        cloud_bearish = senkou_a < senkou_b
        
        # Entry conditions 🚀
        if not self.position:
            # Long setup 🌙
            long_cond = (
                price > max(senkou_a, senkou_b) and
                self.data