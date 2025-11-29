The code looks mostly correct, but I'll make a few technical improvements and add some Moon Dev-themed debug prints for better clarity. Here's the optimized version:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VoltaicSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 2
    lookback_period = 30
    
    def init(self):
        # Calculate Bollinger Bands components
        def bb_bands(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper, middle, lower
        
        self.upper, self.middle, self.lower = self.I(bb_bands, self.data.Close)
        
        # Calculate Bollinger Band Width
        self.band_width = self.I(lambda u, l, m: (u - l)/m, 
                                self.upper, self.lower, self.middle)
        
        # Find narrowest band width in lookback period
        self.min_band_width = self.I(talib.MIN, self.band_width, self.lookback_period)
        
        # Calculate ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Trade tracking variables
        self.entry_price = None
        self.stop_loss = None
        self.highest_high = None

    def next(self):
        price = self.data.Close[-1]
        upper = self.upper[-1]
        band_width = self.band_width[-1]
        min_band = self.min_band_width[-1]
        
        # Moon Dev debug prints ✨
        print(f"🌙 Current Close: {price:.2f} | Upper Band: {upper:.2f} | BW: {band_width:.4f} | Min BW: {min_band:.4f}")
        
        if not self.position:
            # Entry condition check
            if price > upper and abs(band_width - min_band) < 1e-5:
                equity = self.broker.equity()
                atr_value = self.atr[-1]
                
                if atr_value == 0:
                    print("🌙⚠️ ATR value is zero, skipping trade")
                    return
                
                # Risk management calculations 🧮
                risk_amount = equity * self.risk_per_trade
                stop_distance = self.atr_multiplier * atr_value
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.stop_loss = self.entry_price - stop_distance
                    self.highest_high = self.data.High[-1]
                    
                    # Moon Dev entry signal 🌙🚀
                    print(f"\n🌙✨🚀 VOLTAIC SQUEEZE ACTIVATED! 🌙✨🚀"
                          f"\nEntry Price: {self.entry_price:.2f}"
                          f"\nPosition Size: {position_size} units"
                          f"\nInitial Stop: {self.stop_loss:.2f}\n")
                else:
                    print("🌙⚠️ Position size too small, skipping trade")
        else:
            # Update trailing stop logic
            self.highest_high = max(self.highest_high, self.data.High[-1])
            new_stop = self.highest_high - self.atr_multiplier * self.atr[-1]
            self.stop_loss = max(self.stop_loss, new_stop)
            
            # Exit check
            if self.data.Low[-1] <= self.stop_loss:
                pl_pct = self.position.pl_pct * 100
                
                # Moon Dev exit signal 🌙💸
                print(f"\n🌙✨💸 SQUEEZE RELEASED! 💸✨🌙"
                      f"\nExit Price: {self.stop_loss:.2f}"
                      f"\nProfit: {self.position.pl:.2f} ({pl_pct:.1f}%)\n")
                
                self.position.close()

# Data preparation 🌙📊
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',