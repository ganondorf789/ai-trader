```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR HARMONICSTOCHASTIC STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class HarmonicStochastic(Strategy):
    def init(self):
        # 🌟 INDICATOR INITIALIZATION
        # Harmonic Oscillator (RSI)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        
        # Stochastic Convergence
        h, l, c = self.data.High, self.data.Low, self.data.Close
        slowk, slowd = talib.STOCH(h, l, c,
                                   fastk_period=14,
                                   slowk_period=3,
                                   slowk_matype=0,
                                   slowd_period=3,
                                   slowd_matype=0)
        self.stoch_k = self.I(lambda: slowk, name='Stoch_%K')
        self.stoch_d = self.I(lambda: slowd, name='Stoch_%D')
        
        # Volatility Measurement
        self.atr = self.I(talib.ATR, h, l, c, timeperiod=14, name='ATR')
        
        print("🌙 INITIALIZED MOON DEV TRADING BOT WITH COSMIC INDICATORS ✨")

    def next(self):
        # 🌌 CURRENT MARKET CONDITIONS
        current_close = self.data.Close[-1]
        atr_value = self.atr[-1]
        
        # 🚨 ENTRY LOGIC
        if not self.position:
            # LONG SIGNAL: RSI crosses above 70 AND Stoch crossover
            if crossover(self.rsi, 70) and crossover(self.stoch_k, self.stoch_d):
                sl = current_close - 0.5*atr_value
                tp = current_close + 1.5*atr_value
                risk_amount = self.equity * 0.01
                risk_per_share = current_close - sl
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl, tp=tp)
                        print(f"🌙✨ BULLISH BREAKOUT! LONG {position_size} @ {current_close:.2f}")
                        print(f"   🛡️ SL: {sl:.2f} | 🎯 TP: {tp:.2f} | 📏 ATR: {atr_value:.2f}")

            # SHORT SIGNAL: RSI crosses below 30 AND Stoch crossunder
            elif crossunder(self.rsi, 30) and crossunder(self.stoch_k, self.stoch_d):
                sl = current_close + 0.5*atr_value
                tp = current_close - 1.5*atr_value
                risk_amount = self.equity * 0.01
                risk_per_share = sl - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl, tp=tp)
                        print(f"🌙✨ BEARISH BREAKDOWN! SHORT {position_size} @ {current_close:.2f}")
                        print(f"   🛡️ SL: {sl:.2f} | 🎯 TP: {tp:.2f} | 📏 ATR: {atr_value:.2f}")

        # 💫 EXIT LOGIC
        else:
            if self.position.is_long and crossunder(self.rsi, 70):
                self.position.close()
                print(f"🌙💎 CLOSING LONG POSITION | RSI REVERSAL @ {current_close:.2f}")
                
            elif self.position.is_short and crossover(self.rsi, 30):
                self.position.close()
                print(f"🌙💎 CLOSING SHORT POSITION | RSI REVERSAL @ {current_close:.2f}")

# 🌍 DATA PREPARATION
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': '