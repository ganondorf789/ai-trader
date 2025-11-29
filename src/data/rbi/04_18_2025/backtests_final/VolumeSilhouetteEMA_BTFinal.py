I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's VolumeSilhouetteEMA Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolumeSilhouetteEMA(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # ✨ Calculate indicators using TA-Lib
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50 🌙')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA200 🌌')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI ✨')
        self.volume_slope = self.I(talib.LINEARREG_SLOPE, self.data.Volume, timeperiod=5, name='Volume Slope 📉')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low 🔻')

    def next(self):
        # 🚀 Ensure enough data for calculations
        if len(self.data) < 200 or len(self.volume_slope) < 5:
            return

        # 🌙 Entry Logic
        if not self.position:
            # Golden cross conditions (replaced crossover with manual check)
            ema_crossover = (self.ema50[-2] < self.ema200[-2] and self.ema50[-1] > self.ema200[-1])
            close_above_emas = (self.data.Close[-1] > self.ema50[-1]) and (self.data.Close[-1] > self.ema200[-1])
            volume_confirmed = self.volume_slope[-1] < 0  # Declining volume
            
            if ema_crossover and close_above_emas and volume_confirmed:
                # 🎯 Risk Management Calculations
                entry_price = self.data.Close[-1]
                swing_low_price = self.swing_low[-1]
                stop_loss_price = max(swing_low_price, entry_price * 0.98)
                risk_per_share = entry_price - stop_loss_price
                
                if risk_per_share > 0:  # Avoid zero/negative risk
                    position_size = int(round((self.risk_percent * self.equity) / risk_per_share))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss_price, tag='GoldenCross 🌙✨')
                        print(f"🌙✨ MOON DEV ENTRY! ✨🌙 | Price: {entry_price:.2f} | Size: {position_size} | Stop: {stop_loss_price:.2f}")

        # 🛑 Exit Logic
        else:
            # RSI Exit (replaced crossover with manual check)
            if (self.rsi[-2] < 70 and self.rsi[-1] > 70):
                self.position.close(tag='RSI Overbought ✨')
                print(f"✨🌟 RSI EXIT! | RSI: {self.rsi[-1]:.2f} | Price: {self.data.Close[-1]:.2f}")

            # EMA Death Cross Exit (replaced crossover with manual check)
            elif (self.ema50[-2] > self.ema200[-2] and self.ema50[-1] < self.ema200[-1]):
                self.position.close(tag='DeathCross ⚠️')
                print(f"⚠️⚡ DEATH CROSS EXIT! | Price: {self.data.Close[-1]:.2f}")

# 📂 Data Preparation
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

# 🚀 Run Backtest
bt = Backtest(data, VolumeSilhouetteEMA, commission=.002, exclusive_orders=True)
stats = bt.run()
print("🌙✨ MOON DEV BACKTEST COMP