Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# 🌙 Moon Dev's Volatility Pulse Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preprocessing moon magic ✨
def load_data(path):
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    # Cleanse column names
    data.columns = data.columns.str.strip().str.lower()
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper case mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    return data

class VolatilityPulse(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # 🌗 Calculate indicators using TA-Lib
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=14, name='ATR_14')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20, name='ATR_MA_20')
        self.vix_sma = self.I(talib.SMA, self.data['vix'], timeperiod=10, name='VIX_SMA_10')
        
        print("✨ Indicators Initialized:")
        print(f"   ATR(14): {self.atr[-1]:.2f}")
        print(f"   ATR MA(20): {self.atr_ma[-1]:.2f}")
        print(f"   VIX SMA(10): {self.vix_sma[-1]:.2f}")
    
    def next(self):
        print(f"\n🌙 Processing {self.data.index[-1]}")
        
        # Skip early bars
        if len(self.data) < 20: return
        
        # 🕯️ Indicator values
        current_atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        vix = self.data['vix'][-1]
        vix_sma = self.vix_sma[-1]
        
        # 🌪️ Entry conditions (replaced crossover with manual check)
        atr_cross = (self.atr[-2] < self.atr_ma[-2]) and (self.atr[-1] > self.atr_ma[-1])
        vix_filter = vix < vix_sma
        
        # ⏳ Time filter (00:00-00:30 and 23:30-23:59)
        time = self.data.index[-1].time()
        forbidden = (pd.Timestamp('00:00') <= time <= pd.Timestamp('00:30')) or \
                    (pd.Timestamp('23:30') <= time <= pd.Timestamp('23:59:59'))
        
        print(f"✨ ATR CROSS: {atr_cross} | VIX FILTER: {vix_filter}")
        print(f"⏰ Time Check: {time} | Forbidden: {forbidden}")
        
        # 🚀 Long entry logic
        if not self.position and atr_cross and vix_filter and not forbidden:
            # 🧮 Risk management calculations
            risk_amount = self.equity * self.risk_pct
            stop_distance = 2 * current_atr
            position_size = int(round(risk_amount / stop_distance))
            
            if position_size > 0:
                print(f"🚀 BUY SIGNAL! Size: {position_size}")
                self.buy(size=position_size)
                # Initialize trailing stop
                self.highest_high = self.data.High[-1]
                self.position.sl = self.highest_high - 2 * current_atr
                print(f"🔐 Initial SL: {self.position.sl:.2f}")
        
        # 🔄 Update trailing stop for open positions
        if self.position:
            # Update highest high
            if self.data.High[-1] > self.highest_high:
                self.highest_high = self.data.High[-1]
                print(f"🌕 New High! {self.highest_high:.2f}")
            
            # Update stop loss
            new_sl = self.highest_high - 2 * self.atr[-1]
            if new_sl > self.position.sl:
                self.position.sl = new_sl
                print