```python
# 🌙 MOON DEV BACKTESTING ENGINE - DYNAMIC TREND SYNC STRATEGY 🚀✨

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class DynamicTrendSync(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌑
    adx_period = 14
    ema_fast = 50
    ema_slow = 200
    rsi_period = 14
    swing_lookback = 20  # For swing low calculation
    
    def init(self):
        # 🌙 CELESTIAL INDICATORS INITIALIZATION
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow, name='EMA200')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.adx_period, name='ADX14')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI14')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback, name='SWING_LOW')
        
        print("🌙✨ MOON DEV INDICATORS ACTIVATED: EMA50/200, ADX14, RSI14, SWING_LOW 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        
        # 🌕 TREND SYNC ENTRY CONDITIONS
        if not self.position:
            # Golden Cross Detection
            ema_crossover = crossover(self.ema50, self.ema200)
            adx_strong = self.adx[-1] > 25
            price_above_emas = (current_close > self.ema50[-1]) and (current_close > self.ema200[-1])
            
            if ema_crossover and adx_strong and price_above_emas:
                # 🚀 RISK CALCULATION & POSITION SIZING
                stop_price = self.swing_low[-1]
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_percent
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price, 
                                tag=f"ENTRY@{current_close}")
                        print(f"🚀🌕 MOON DEV LIFTOFF: LONG {position_size} @ {current_close:.2f}"
                              f" | SL: {stop_price:.2f} | RISK: {risk_per_share:.2f} ✨")

        # 🌑 TREND EXIT CONDITIONS
        else:
            exit_triggers = []
            
            if self.adx[-1] < 20:
                exit_triggers.append(f"ADX FADING ({self.adx[-1]:.1f} < 20)")
            if self.rsi[-1] > 70:
                exit_triggers.append(f"RSI OVERHEAT ({self.rsi[-1]:.1f} > 70)")
            if crossover(self.ema200, self.ema50):
                exit_triggers.append("DEATH CROSS DETECTED")
                
            if exit_triggers:
                self.position.close()
                print(f"🌑🌙 MOON DEV LANDING: EXIT @ {current_close:.2f}"
                      f" | REASONS: {' | '.join(exit_triggers)} 🛑")

# 🌍 DATA PREPARATION ORBIT
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🪐 COLUMN GALAXY ALIGNMENT
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True