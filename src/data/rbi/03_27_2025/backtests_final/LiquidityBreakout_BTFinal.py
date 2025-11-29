Here's the fully debugged and optimized code with Moon Dev enhancements while preserving the original strategy logic:

```python
import pandas as pd
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, crossunder

# Moon Dev Data Preparation 🌙✨
def load_data(data_path):
    data = pd.read_csv(data_path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    # Convert datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class LiquidityBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    trailing_atr_multiplier = 2.0  # 2x ATR for trailing stop
    
    def init(self):
        # Moon Dev Indicators ✨ (Pure TA-Lib implementation)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=20)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # Trade tracking variables
        self.breakout_level = None
        self.initial_sl = None
        self.trailing_stop_peak = None

    def next(self):
        # Moon Dev Trading Logic 🌙🚀
        if not self.position:
            # Long entry condition
            if (self.data.Close[-1] > self.swing_high[-1]) and (self.atr[-1] < self.atr_sma[-1]):
                self.enter_long()
                
            # Short entry condition
            elif (self.data.Close[-1] < self.swing_low[-1]) and (self.atr[-1] < self.atr_sma[-1]):
                self.enter_short()
        else:
            self.manage_exits()

    def enter_long(self):
        risk_amount = self.risk_percent * self.equity
        stop_loss_price = self.swing_low[-1]
        entry_price = self.data.Close[-1]
        risk_per_share = entry_price - stop_loss_price
        
        if risk_per_share <= 0:
            print("🌙⚠️ MOON DEV WARNING: Invalid risk calculation for long entry")
            return
            
        position_size = int(round(risk_amount / risk_per_share))
        if position_size > 0:
            self.buy(size=position_size)
            self.breakout_level = self.swing_high[-1]
            self.initial_sl = stop_loss_price
            self.trailing_stop_peak = self.data.High[-1]
            print(f"🌙✨ MOON DEV LONG ENTRY: {entry_price:.2f} | Size: {position_size} | SL: {stop_loss_price:.2f} 🚀")

    def enter_short(self):
        risk_amount = self.risk_percent * self.equity
        stop_loss_price = self.swing_high[-1]
        entry_price = self.data.Close[-1]
        risk_per_share = stop_loss_price - entry_price
        
        if risk_per_share <= 0:
            print("🌙⚠️ MOON DEV WARNING: Invalid risk calculation for short entry")
            return
            
        position_size = int(round(risk_amount / risk_per_share))
        if position_size > 0:
            self.sell(size=position_size)
            self.breakout_level = self.swing_low[-1]
            self.initial_sl = stop_loss_price
            self.trailing_stop_peak = self.data.Low[-1]
            print(f"🌙✨ MOON DEV SHORT ENTRY: {entry_price:.2f} | Size: {position_size} | SL: {stop_loss_price:.2f} 🚀")

    def manage_exits(self):
        current_atr = self.atr[-1]
        current_atr_sma = self.atr_sma[-1]
        
        # ATR expansion exit
        if current_atr > current_atr_sma:
            self.position.close()
            print(f"