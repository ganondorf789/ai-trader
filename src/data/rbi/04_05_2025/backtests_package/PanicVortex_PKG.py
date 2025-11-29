Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# 🌙 Moon Dev's PanicVortex Backtest Script �
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta
import numpy as np

class PanicVortex(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌙
    stop_loss_pct = 0.03   # 3% initial stop loss ✨
    trailing_stop_pct = 0.05  # 5% trailing stop 🌕
    time_limit = 5 * 96  # 5 days in 15m bars (96 bars/day) ⏳

    def init(self):
        # Clean and prepare VIX data 🌪️
        self.vix = self.data.df['vix']  # Assumes cleaned 'vix' column exists
        
        # VIX Bollinger Bands (20,2) 🌈
        self.vix_upper = self.I(talib.BBANDS, self.vix, 20, 2, 2, name='VIX_UPPER', which=0)
        self.vix_sma = self.I(talib.SMA, self.vix, 20, name='VIX_SMA')
        
        # Vortex Indicator (14) 🌪️➿
        self.vi_plus = self.I(self._calculate_vortex, which='plus', name='VI+')
        self.vi_minus = self.I(self._calculate_vortex, which='minus', name='VI-')
        
        # Swing Low for stop loss 📉
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='SWING_LOW')

    def _calculate_vortex(self, which):
        # Vectorized Vortex calculation using pandas_ta 🚀
        vortex = pandas_ta.vortex(
            high=self.data.High.s,
            low=self.data.Low.s,
            close=self.data.Close.s,
            length=14
        )
        return vortex[f'VORTIC_14_{"1" if which == "minus" else "0"}'].values

    def next(self):
        price = self.data.Close[-1]
        
        # Entry Logic 🌙✨
        if not self.position:
            # VIX Spike + Bullish Vortex Crossover 🌪️⬆️
            if (self.vix[-1] > self.vix_upper[-1] and 
                (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1])):
                
                # Risk-based Position Sizing ⚖️
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_price = min(price * (1 - self.stop_loss_pct), self.swing_low[-1])
                risk_per_unit = price - stop_price
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        sl = stop_price
                        self.buy(size=position_size, sl=sl, 
                                tag=f"🌙 ENTRY {price:.2f}")
                        print(f"🚀 MOON DEV BUY {position_size} @ {price:.2f} | SL: {sl:.2f} 🌕")

        # Exit Logic 🛑
        else:
            # VIX Mean Reversion Exit 📉
            if self.vix[-1] < self.vix_sma[-1]:
                self.position.close()
                print(f"🌙 VIX SMA EXIT @ {price:.2f} 🌌")

            # Time-based Exit ⏰
            elif len(self.data) - self.position.entry_bar >= self.time_limit:
                self.position.close()
                print(f"⏳ TIME EXIT @ {price:.2f} 🌠")

            # Trailing Stop Update 🛑
            elif self.data.High[-1] > self.position.high_price:
                new_sl = self.data.High[-1] * (1 - self.trailing_stop_pct)
                self.position.sl = max(new_sl, self.position.sl)

# Data Preparation 🌙📊
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data =