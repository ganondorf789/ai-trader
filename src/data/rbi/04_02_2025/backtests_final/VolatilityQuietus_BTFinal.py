Here's the fully debugged and Moon Dev optimized backtest code:

```python
# 🌙 MOON DEV BACKTESTING ENGINE 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolatilityQuietus(Strategy):
    ema_period = 200
    atr_period = 14
    atr_sma_period = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌟 INDICATOR CALCULATION USING TA-LIB 🌟
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period, name='EMA 200')
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                           timeperiod=self.atr_period, name='ATR 14')
        self.atr_sma20 = self.I(talib.SMA, self.atr14, timeperiod=self.atr_sma_period, name='ATR SMA 20')
        
        self.entry_atr = None
        self.highest_high = None

    def next(self):
        # ✨ MOON DEV DATA CHECK ✨
        if len(self.data) < self.ema_period + self.atr_sma_period:
            return

        # 🌙 CORE STRATEGY LOGIC 🌙
        if not self.position:
            # 🚀 ENTRY CONDITIONS 🚀
            prev_close = self.data.Close[-2]
            prev_ema = self.ema200[-2]
            prev_atr = self.atr14[-2]
            prev_atr_sma = self.atr_sma20[-2]

            if (prev_close > prev_ema) and (prev_atr < prev_atr_sma):
                # 💰 RISK MANAGEMENT CALCULATIONS 💰
                risk_amount = self.equity * self.risk_pct
                entry_price = self.data.Open[-1]
                stop_loss_price = entry_price * 0.99
                risk_per_share = entry_price - stop_loss_price
                
                if risk_per_share > 0:
                    position_size = risk_amount / risk_per_share
                    position_size = int(round(position_size))  # 🌕 CRITICAL: Round to whole units
                    
                    if position_size > 0:
                        # 🌕 MOON DEV ENTRY SIGNAL 🌕
                        self.buy(size=position_size)
                        self.entry_atr = self.atr14[-2]
                        self.highest_high = self.data.High[-1]
                        print(f"\n🌙✨ MOON DEV LONG ENTRY ✨🌙"
                              f"\nEntry Price: {entry_price:.2f}"
                              f"\nSize: {position_size} units"
                              f"\nRisk: {self.risk_pct*100}% Equity"
                              f"\nATR Quietus Level: {self.entry_atr:.2f}")
        else:
            # 🔥 EXIT CONDITIONS 🔥
            current_atr = self.atr14[-1]
            self.highest_high = max(self.highest_high, self.data.High[-1])
            
            # 🚨 EMERGENCY ATR EXIT 🚨
            if current_atr >= 2 * self.entry_atr:
                self.position.close()
                print(f"\n🚨🌑 VOLATILITY SPIKE EXIT 🌑🚨"
                      f"\nCurrent ATR: {current_atr:.2f}"
                      f"\n2x Entry ATR: {2*self.entry_atr:.2f}")
                return
            
            # 🔻 TRAILING STOP UPDATE 🔻
            trailing_stop = self.highest_high * 0.99
            if self.data.Low[-1] < trailing_stop:
                self.position.close()
                print(f"\n🔻🌕 TRAILING STOP TRIGGERED 🌕🔻"
                      f"\nExit Price: {self.data.Close[-1]:.2f}"
                      f"\nPeak Price: {self.highest_high:.2f}")

# 🛠️ DATA PREPARATION 🛠️
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🧹 DATA CLEANING 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# 📊 COLUMN MAPPING 📊