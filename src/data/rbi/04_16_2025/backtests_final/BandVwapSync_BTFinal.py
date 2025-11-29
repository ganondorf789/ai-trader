Here's the debugged code with all technical issues fixed while preserving the original strategy logic:

```python
# 🌙 Moon Dev's BandVwapSync Backtest Implementation 🚀
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

class BandVwapSync(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌙
    
    def init(self):
        # 🌗 Bollinger Bands (20,2) with TA-Lib
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                              timeperiod=20, nbdevup=2, nbdevdn=2, 
                                                              matype=0)
        
        # 🌀 Volume-Weighted Average Price (20-period)
        self.vwap_20 = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, 
                            self.data.Volume, length=20)
        
        # 📈 5-period MA of VWAP
        self.vwap_ma_5 = self.I(talib.SMA, self.vwap_20, timeperiod=5)
        
    def next(self):
        # Wait for sufficient data 🌊
        if len(self.data.Close) < 20:
            return
            
        # 🛸 Current indicator values
        price_low = self.data.Low[-1]
        bb_lower = self.bb_lower[-1]
        vwap = self.vwap_20[-1]
        prev_vwap = self.vwap_20[-2] if len(self.vwap_20) > 1 else vwap
        
        # 🌙 Entry Logic: Band touch + VWAP momentum
        if not self.position:
            if price_low <= bb_lower and vwap > prev_vwap:
                # 🛡️ Risk Management Calculations
                entry_price = self.data.Close[-1]
                bb_upper = self.bb_upper[-1]
                bandwidth = bb_upper - bb_lower
                stop_loss = bb_lower - (bandwidth * 0.5)
                
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🌙 MOON DEV ALERT: LONG @ {entry_price:.2f} ✨ | Size: {position_size} | SL: {stop_loss:.2f} 🚀")
        
        # 💼 Exit Logic
        else:
            # 📉 VWAP trend reversal exit
            if len(self.vwap_ma_5) > 1 and len(self.vwap_20) > 1:
                if self.vwap_ma_5[-2] < self.vwap_20[-2] and self.vwap_ma_5[-1] > self.vwap_20[-1]:
                    self.position.close()
                    print("🌙 VWAP Trend Reversal - Closing Position! 🏁")
                
            # 🎯 Upper band profit taking
            if self.data.High[-1] >= self.bb_upper[-1]:
                self.position.close()
                print("🚀 Upper Band Reached - Profit Harvest! 🌕")
                
            # 🛡️ Trailing stop to breakeven
            if self.data.High[-1] >= self.bb_middle[-1] and self.position.sl < self.position.entry_price:
                self.position.sl = self.position.entry_price
                print("🌙 Trail Stop Activated - Breakeven Secured! 🛡️")

# 🧹 Data Cleaning Ritual
data = pd.read_csv('BTC-USD-15m.csv')
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

# 🌙 Launch Backtest
bt = Backtest(data, BandVwapSync, commission=.002, exclusive_orders=True)
stats = bt.run()
print("🌙✨ Backtest Complete! Moon Dev's Cosmic Performance Report:")
print(stats)
bt.plot