```python
# 🌙 Moon Dev Backtest Implementation for VolatilityLiquidation Strategy

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VolatilityLiquidation(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    atr_period = 96  # 24h in 15m intervals (4*24=96)
    trailing_atr_mult = 0.75  # Trailing stop multiplier
    liquidation_volume_mult = 2.0  # Volume surge threshold

    def init(self):
        # 🌙 Calculate indicators using TA-Lib through self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.pivot = self.I(self.calculate_pivot)  # Custom pivot calculation
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # ✨ Threshold calculations
        self.upper_threshold = self.pivot + 1.5 * self.atr
        self.lower_threshold = self.pivot - 1.5 * self.atr

    def calculate_pivot(self):
        # 🌀 Moon Dev Pivot Calculation Magic
        highs = self.I(talib.MAX, self.data.High, 96)
        lows = self.I(talib.MIN, self.data.Low, 96)
        closes = self.I(talib.SMA, self.data.Close, 96)  # Approximation for daily close
        return (highs + lows + closes) / 3

    def next(self):
        # 🌙 Current market conditions
        price = self.data.Close[-1]
        atr_value = self.atr[-1]
        volume = self.data.Volume[-1]
        volume_sma = self.volume_sma[-1]

        # 🚨 Moon Dev Debug Console
        print(f"🌙 [BAR {len(self.data)}] Price: {price:.2f} | Pivot: {self.pivot[-1]:.2f} | ATR: {atr_value:.2f} | Vol: {volume:.2f} vs SMA: {volume_sma:.2f}")

        if not self.position:
            # 🌗 Long Entry Condition
            if price <= self.lower_threshold[-1]:
                if volume > self.liquidation_volume_mult * volume_sma:
                    risk_amount = self.risk_pct * self.equity
                    stop_loss = 2 * atr_value
                    position_size = int(round(risk_amount / stop_loss))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=price - stop_loss)
                        print(f"🚀🌕 MOON BLAST LONG! Size: {position_size} | Entry: {price:.2f} | SL: {price-stop_loss:.2f}")

            # 🌑 Short Entry Condition
            elif price >= self.upper_threshold[-1]:
                if volume > self.liquidation_volume_mult * volume_sma:
                    risk_amount = self.risk_pct * self.equity
                    stop_loss = 2 * atr_value
                    position_size = int(round(risk_amount / stop_loss))
                    
                    if position_size > 0:
                        self.sell(size=position_size, sl=price + stop_loss)
                        print(f"🌑💫 BLACK HOLE SHORT! Size: {position_size} | Entry: {price:.2f} | SL: {price+stop_loss:.2f}")
        else:
            # 🌈 Exit Conditions
            if self.position.is_long:
                # Pivot retracement or trailing stop
                if price >= self.pivot[-1]:
                    self.position.close()
                    print(f"🎯💰 PIVOT HIT! Long closed at {price:.2f}")
                else:
                    new_sl = price - (self.trailing_atr_mult * atr_value)
                    self.position.sl = max(new_sl, self.position.sl)
                    
            elif self.position.is_short:
                if price <= self.pivot[-1]:
                    self.position.close()
                    print(f"🎯💰 PIVOT HIT! Short closed at {price:.2f}")
                else:
                    new_sl = price + (self.trailing_atr_mult * atr_value)
                    self.position.sl = min(new_sl, self.position.sl)

# 🌙 Data Preparation Ritual
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC