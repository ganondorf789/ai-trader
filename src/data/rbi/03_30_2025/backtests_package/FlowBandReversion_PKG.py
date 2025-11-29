Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class FlowBandReversion(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Core indicators
        close = self.data.Close
        self.sma20 = self.I(talib.SMA, close, timeperiod=20, name='SMA20')
        self.stddev = self.I(talib.STDDEV, close, timeperiod=20, name='STDDEV')
        self.bb_upper = self.I(lambda sma, std: sma + 2*std, self.sma20, self.stddev, name='BB_Upper')
        self.bb_lower = self.I(lambda sma, std: sma - 2*std, self.sma20, self.stddev, name='BB_Lower')
        
        # Swing extremes
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        
        # On-chain data
        self.netflow = self.I(lambda x: x, self.data.Netflow, name='Netflow')
        
        print("🌙✨ Strategy initialized with Moon Dev precision! All indicators calibrated for lunar cycles!")

    def next(self):
        price = self.data.Close[-1]
        netflow = self.netflow[-1]
        
        # Calculate Bollinger Band contraction (5% threshold)
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1])/self.sma20[-1] if self.sma20[-1] else 0
        
        if not self.position:
            # Long entry logic
            if (price <= self.bb_lower[-1]) and (netflow < 0) and (bb_width < 0.05):
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Open[-1]
                sl_price = self.swing_low[-1]
                
                if (risk_per_share := (entry_price - sl_price)) > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.buy(size=size)
                        print(f"🌙🚀 LONG ENTRY! Moon gravity pulling price up! Size: {size} | Price: {entry_price:.2f} | Netflow: {netflow:.2f}")
            
            # Short entry logic
            elif (price >= self.bb_upper[-1]) and (netflow > 0) and (bb_width < 0.05):
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Open[-1]
                sl_price = self.swing_high[-1]
                
                if (risk_per_share := (sl_price - entry_price)) > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.sell(size=size)
                        print(f"🌙🚀 SHORT ENTRY! Moon phase indicates downward pressure! Size: {size} | Price: {entry_price:.2f} | Netflow: {netflow:.2f}")
        else:
            # Exit logic
            if self.position.is_long:
                if price >= self.sma20[-1]:
                    self.position.close()
                    print(f"🌙✨ LONG EXIT at SMA20: {price:.2f} - Moon profits secured!")
                elif self.data.Low[-1] < self.swing_low[-1]:
                    self.position.close()
                    print(f"🌙🚨 LONG STOP! Swing Low breached: {self.swing_low[-1]:.2f} - Moon dust protection activated!")
            
            elif self.position.is_short:
                if price <= self.sma20[-1]:
                    self.position.close()
                    print(f"🌙✨ SHORT EXIT at SMA20: {price:.2f} - Moon profits secured!")
                elif self.data.High[-1] > self.swing_high[-1]:
                    self.position.close()
                    print(f"🌙🚨 SHORT STOP! Swing High breached: {self.swing_high[-1]:.2f} - Moon dust protection