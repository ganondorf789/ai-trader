```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR SQUEEZE SURGE STRATEGY 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class SqueezeSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌟 INDICATOR CALCULATION USING TA-LIB AND SELF.I() WRAPPER
        # Bollinger Bands (20,2)
        self.bb_upper = self.I(lambda x: talib.BBANDS(x, 20, 2, 2)[0], self.data.Close, name='BB_UPPER')
        self.bb_lower = self.I(lambda x: talib.BBANDS(x, 20, 2, 2)[2], self.data.Close, name='BB_LOWER')
        
        # Bollinger Band Width
        self.bb_width = self.I(lambda u, l: (u - l)/self.data.Close,
                              self.bb_upper, self.bb_lower, name='BB_WIDTH')
        
        # Volume MA (50)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 50, name='VOLUME_MA')
        
        # ATR (14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        # RSI (14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI')
        
        self.trailing_high = None  # For trailing stop calculation

    def next(self):
        # 🌌 MOON DEV SIGNAL VALIDATION
        if not self.position:
            # 🚀 ENTRY CONDITIONS
            squeeze_condition = self.bb_width[-1] < 0.02
            breakout_condition = self.data.Close[-1] > self.bb_upper[-1]
            volume_surge = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]
            
            if squeeze_condition and breakout_condition and volume_surge:
                # 💰 RISK-BASED POSITION SIZING
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1]
                position_size = risk_amount / (3 * atr_value)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.trailing_high = self.data.High[-1]
                    print(f"🌙✨ MOON DEV ALERT: Long Entry at {self.data.Close[-1]}! Squeeze Surge Detected! 🚀📈")
        else:
            # 🛑 EXIT CONDITIONS
            # Update trailing high
            self.trailing_high = max(self.trailing_high, self.data.High[-1])
            
            # Trailing stop calculation (3xATR)
            stop_price = self.trailing_high - 3 * self.atr[-1]
            rsi_exit = crossover(self.rsi, 70)[-1]
            
            if self.data.Close[-1] < stop_price or rsi_exit:
                self.position.close()
                if self.data.Close[-1] < stop_price:
                    print(f"🌙🛑 TRAILING STOP TRIGGERED! Exiting at {self.data.Close[-1]} 🌊")
                else:
                    print(f"🌙💹 RSI OVERHEATED! Taking Profit at {self.data.Close[-1]} 🌟")

# 🧹 DATA CLEANING AND PREPARATION
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['DateTime'] = pd.to_datetime(data['datetime'])
data = data.set_index('DateTime').sort_index()

# 🚀 BACKTEST EXECUTION
bt = Backtest(data, SqueezeSurge, cash=1_000_000, margin=1.0)
stats = bt.run()
print("🌕🌖🌗🌘