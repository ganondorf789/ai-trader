Here's the debugged backtest code with Moon Dev themed improvements 🌙✨:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
from datetime import time

# Moon Dev Data Loading 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    print("🌙✨ MOON DEV DATA LOADED SUCCESSFULLY ✨🌙")
except Exception as e:
    print(f"🚨 MOON DEV ERROR: Data loading failed - {str(e)}")
    raise

# Moon Dev Data Cleaning ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'liquidation': 'Liquidation'
})

class LiquidationContraction(Strategy):
    def init(self):
        # Moon Dev Liquidation Indicator 🌙
        self.liquidation_ma = self.I(talib.SMA, self.data.Liquidation, timeperiod=2880, name='LIQ_30D_MA')
        
        # Moon Dev Bollinger Band Width Calculation ✨
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close, name='BB_WIDTH')
        
        # Moon Dev BB Width Range Indicators �
        self.bb_width_max = self.I(talib.MAX, self.bb_width, timeperiod=1344, name='BB_WIDTH_MAX')
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=1344, name='BB_WIDTH_MIN')
        
        # Moon Dev ATR for Risk Management 🌙
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        
    def next(self):
        # Moon Dev session filter 🌙
        current_time = self.data.index[-1].time()
        if not (8 <= current_time.hour < 16):
            return
        
        # Moon Dev position limit 🚀
        if len(self.trades) >= 3:
            return
        
        # Moon Dev entry logic ✨
        liq_condition = self.data.Liquidation[-1] > 2 * self.liquidation_ma[-1]
        bb_range = self.bb_width_max[-1] - self.bb_width_min[-1]
        bb_condition = self.bb_width[-1] < (self.bb_width_min[-1] + 0.5 * bb_range)
        
        if liq_condition and bb_condition and not self.position:
            # Moon Dev risk calculation 🌙
            risk_amount = self.equity * 0.01
            atr_value = self.atr[-1]
            
            if atr_value > 0:
                position_size = int(round(risk_amount / (1.5 * atr_value)))
                if position_size > 0:
                    tp_price = self.data.Close[-1] + 1.5 * atr_value
                    self.buy(
                        size=position_size,
                        tag={
                            'take_profit': tp_price,
                            'entry_bar': len(self.data)
                        }
                    )
                    print(f"🌙✨ MOON DEV ENTRY ✨🌙 Size: {position_size} @ {self.data.Close[-1]:.2f} | TP: {tp_price:.2f} 🚀")

        # Moon Dev exit logic 🌙
        for trade in self.trades:
            if trade.is_long and trade.is_open:
                tp_price = trade.tag['take_profit']
                entry_bar = trade.tag['entry_bar']
                
                # Take profit check
                if self.data.High[-1] >= tp_price:
                    trade.close()
                    print(f"🚀🌙 TP HIT! Profit: {trade.pl:.2f} ✨")
                
                # Time exit check
                elif (len(self.data) - entry_bar) >= 5:
                    trade.close()
                    print(f"⏰🌙 TIME EXIT! Bars held: {len(self.data) - entry_bar}