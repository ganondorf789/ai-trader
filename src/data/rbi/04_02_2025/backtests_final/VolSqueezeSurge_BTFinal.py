I'll fix the code while maintaining the original strategy logic and adding Moon Dev themed debug prints. Here's the corrected version:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolSqueezeSurge(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    time_exit_bars = 480  # 5 days in 15m intervals (5*96=480 bars)
    
    def init(self):
        # Bollinger Band Width calculation
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close)
        self.bb_width_min = self.I(talib.MIN, self.bb_width, 10)

        # Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)

        # Keltner Channel components
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        self.ema = self.I(talib.EMA, typical_price, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)

        print("🌙 Moon Dev Indicators Initialized! ✨")
        print("⚡ All indicators powered by pure cosmic energy (TA-Lib)")
        print("🔭 No backtesting.lib contamination detected in our lunar observatory")

    def next(self):
        if not self.position:
            # Entry logic
            bb_squeeze = self.bb_width[-1] == self.bb_width_min[-1]
            volume_surge = self.data.Volume[-1] > 1.5 * self.volume_sma[-1]
            
            if bb_squeeze and volume_surge:
                # Risk management calculations
                stop_price = min(self.swing_low[-1], self.ema[-1] - 2*self.atr[-1])
                risk_amount = self.risk_pct * self.equity
                entry_price = self.data.Close[-1]
                
                if entry_price <= stop_price:
                    print("🌑 Moon Dev Alert: Invalid risk parameters!")
                    print("⚠️ Cosmic alignment disrupted - aborting trade!")
                    return
                
                position_size = int(round(risk_amount / (entry_price - stop_price)))
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_price)
                    self.squeeze_level = self.bb_width[-1]
                    self.entry_bar = len(self.data)
                    print(f"🚀 Lunar Boost: Long {position_size} units @ {entry_price} | SL {stop_price} 🌕")
                    print(f"🌌 Position secured with moon-grade risk management")
        else:
            # Exit logic
            current_upper_keltner = self.ema[-1] + 2*self.atr[-1]
            
            if self.data.High[-1] >= current_upper_keltner:
                self.position.close()
                print(f"🎯 Cosmic Target Hit: {self.data.Close[-1]} 🌠")
                print("💰 Profit harvested by lunar gravity assist")
            elif self.bb_width[-1] > 2*self.squeeze_level:
                self.position.close()
                print(f"🌪️ Volatility Storm: Exit @ {self.data.Close[-1]} 🌧️")
                print("☄️ Emergency thrusters activated - exiting turbulence")
            elif len(self.data) - self.entry_bar >= self.time_exit_bars:
                self.position.close()
                print(f"⌛ Time Warp Exit: {self.data.Close[-1]} ⏳")
                print("⏱️ Temporal trade limit reached - returning to present")

# Data preparation
print("\n🌠 Preparing cosmic data streams...")
data = pd.read_csv('BTC-USD-15m.csv', parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index('datetime', inplace=True)

print("🌕 Data successfully aligned with lunar cycles!")
print("📡 Beginning backtest transmission...\n")

bt = Back