```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VolcanicSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate core indicators with self.I()
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=15, name='EMA_15')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=15, name='ATR_15')
        self.upper_band = self.I(lambda ema, atr: ema + 1.5 * atr, self.ema, self.atr, name='UPPER_BAND')
        self.lower_band = self.I(lambda ema, atr: ema - 1.5 * atr, self.ema, self.atr, name='LOWER_BAND')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_MA20')
        
        print("🌋✨ Moon Dev Indicators Activated! VolcanicSqueeze strategy initialized 🚀")

    def next(self):
        # Wait until we have sufficient historical data
        if len(self.data) < 20:
            return

        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        vol_ma = self.volume_ma[-1]

        # Calculate dynamic conditions
        volume_surge = current_volume >= 1.5 * vol_ma
        squeeze = (self.upper_band[-1] - self.lower_band[-1]) < (self.upper_band[-2] - self.lower_band[-2])

        # Moon Dev themed debug prints
        print(f"🌙 Current Close: {current_close:.2f} | Upper: {upper:.2f} | Lower: {lower:.2f}")
        print(f"📈 Volume: {current_volume:.2f} vs MA20: {vol_ma:.2f} | Surge: {'✅' if volume_surge else '❌'}")
        print(f"🔀 Squeeze Condition: {'✅' if squeeze else '❌'}")

        # Entry logic
        if not self.position:
            # Long entry - volcanic breakout upwards 🌋🚀
            if current_close > upper and volume_surge and squeeze:
                risk_amount = self.equity * self.risk_per_trade
                sl_price = lower
                risk_per_unit = current_close - sl_price
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price)
                        print(f"\n🚀🌋 MOON DEV LONG ERUPTION! Size: {position_size}")
                        print(f"Entry: {current_close:.2f} | SL: {sl_price:.2f} | Potential R/R: 1:2 ✨\n")

            # Short entry - volcanic breakdown downwards 🌋🌪️    
            elif current_close < lower and volume_surge and squeeze:
                risk_amount = self.equity * self.risk_per_trade
                sl_price = upper
                risk_per_unit = sl_price - current_close
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl_price)
                        print(f"\n🌪️🌋 MOON DEV SHORT ERUPTION! Size: {position_size}")
                        print(f"Entry: {current_close:.2f} | SL: {sl_price:.2f} | Potential R/R: 1:2 ✨\n")

        # Exit logic - return to channel calm 🌙💫
        else:
            if self.position.is_long and current_close < upper:
                self.position.close()
                print(f"\n🌙💫 MOON DEV LONG EXIT: Price returned to channel at {current_close:.2f}")
                
            elif self.position.is_short and current_close > lower:
                self.position.close()
                print(f"\n🌙💫 MOON DEV SHORT EXIT: Price returned to channel at {current_close:.2f}")

# Data preparation with Moon Dev safeguards 🌙🛡️
data = pd.read_csv('/Users/md/D