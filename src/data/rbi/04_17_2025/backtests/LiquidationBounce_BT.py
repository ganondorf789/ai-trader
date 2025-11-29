```python
# 🌙 MOON DEV BACKTESTING SCRIPT 🚀

# 1. NECESSARY IMPORTS
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 2. DATA HANDLING 🌍
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# 3. STRATEGY CLASS WITH INDICATORS 📈
class LiquidationBounce(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    time_exit_bars = 16  # 4 hours in 15m intervals
    
    def init(self):
        # 🌗 BOLLINGER BANDS (20,2)
        self.mid_bb = self.I(talib.SMA, self.data.Close, 20)
        self.stddev = self.I(talib.STDDEV, self.data.Close, 20)
        self.upper_bb = self.I(lambda: self.mid_bb + 2*self.stddev)
        self.lower_bb = self.I(lambda: self.mid_bb - 2*self.stddev)
        
        # 🌓 KELTNER MIDLINE (EMA 20)
        self.keltner = self.I(talib.EMA, self.data.Close, 20)
        
        # 🌀 LIQUIDATION CLUSTER DETECTION 
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        self.entry_bar = None  # Track entry time

    def next(self):
        price = self.data.Close[-1]
        equity = self.equity
        
        # 🌌 MOON DEV ENTRY CONDITIONS 🌠
        if not self.position:
            # Price below Bollinger Band
            bb_break = price < self.lower_bb[-1]
            
            # Liquidation cluster detection (price at swing low)
            liquidation_signal = self.data.Low[-1] <= self.swing_low[-1]
            
            # Order book imbalance (volume spike)
            volume_spike = self.data.Volume[-1] > self.volume_sma[-1] * 1.5
            
            if bb_break and liquidation_signal and volume_spike:
                # 🛑 RISK MANAGEMENT CALCULATIONS
                sl_price = self.mid_bb[-1]  # Middle Bollinger as stop loss
                risk_per_unit = abs(sl_price - price)
                
                if risk_per_unit > 0:
                    risk_amount = equity * self.risk_pct
                    position_size = int(round(risk_amount / risk_per_unit))
                    
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl_price)
                        self.entry_bar = len(self.data)
                        print(f"🌕 MOON DEV SHORT ACTIVATED! 🌑\nEntry: {price:.2f} | Size: {position_size} | SL: {sl_price:.2f}")

        # 🚀 EXIT CONDITIONS 
        elif self.position.is_short:
            # Keltner midline exit
            if price > self.keltner[-1]:
                self.position.close()
                print(f"✨ KELTNER EXIT TRIGGERED @ {price:.2f}")
            
            # Bollinger Band re-entry exit
            elif price > self.lower_bb[-1]:
                self.position.close()
                print(f"🛑 EMERGENCY BB EXIT @ {price:.2f}")
            
            # Time-based exit
            elif (len(self.data) - self.entry_bar) >= self.time_exit_bars:
                self.position.close()
                print(f"⏳ TIME-BASED EXIT AFTER 4 HOURS @ {price:.2f}")

# 4. BACKTEST EXECUTION ⚙️
bt = Backtest(data, LiquidationBounce, cash=1_000_000, commission=.002)
stats = bt.run()

# 5. MOON DEV RESULTS DISPLAY 🌙
print("\n" + "="*50)
print("🌙✨ MOON DEV BACKTEST COMPL