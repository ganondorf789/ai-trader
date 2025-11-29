Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints added:

```python
# 🌙✨ Moon Dev's LiquiClusterBreach Backtest Script 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquiClusterBreach(Strategy):
    risk_per_trade = 0.01  # 1% of equity
    atr_period = 14
    bb_period = 20
    
    def init(self):
        # Calculate Indicators 🌙
        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                   timeperiod=self.bb_period, nbdevup=2, nbdevdn=2,
                                                   name=['UpperBB', 'MiddleBB', 'LowerBB'])
        
        # ATR for Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                        timeperiod=self.atr_period, name='ATR')
        
        # Liquidation Zones (Swing Highs/Lows)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SwingLow')
        
        # Volume Analysis
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VolAvg')
        self.volume_ratio = self.data.Volume / self.volume_avg
        
    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_time = self.data.index[-1]
        
        # 🌙✨ Moon Dev Debug Prints
        print(f"🌙 Processing {current_time} | Close: {current_close:.2f} | UpperBB: {self.upper[-1]:.2f} | LowerBB: {self.lower[-1]:.2f}")
        print(f"🌙 Volume Ratio: {self.volume_ratio[-1]:.2f}x | ATR: {self.atr[-1]:.2f}")
        
        # Entry Conditions
        if not self.position:
            # Time Filter (London/NY Overlap 13-17 UTC)
            if current_time.hour not in range(13, 17):
                print("🌑 Outside trading hours (13-17 UTC)")
                return
                
            # Liquidation Zone Check
            upper_zone_dist = abs(current_close - self.swing_high[-1]) / self.swing_high[-1]
            lower_zone_dist = abs(current_close - self.swing_low[-1]) / self.swing_low[-1]
            
            # Long Entry Logic 🚀
            if (current_close > self.upper[-1] and 
                upper_zone_dist <= 0.15 and 
                self.volume_ratio[-1] >= 2):
                
                # Risk Management 🌙
                stop_loss = self.lower[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_close - stop_loss
                position_size = int(round(risk_amount / risk_per_share))
                
                # Entry Execution
                self.buy(size=position_size, 
                        sl=stop_loss,
                        tp=current_close + 1.5*self.atr[-1])
                print(f"🚀🌙 LONG ENTRY | Size: {position_size} | SL: {stop_loss:.2f} | TP: {current_close + 1.5*self.atr[-1]:.2f}")
            
            # Short Entry Logic 🌑    
            elif (current_close < self.lower[-1] and 
                  lower_zone_dist <= 0.15 and 
                  self.volume_ratio[-1] >= 2):
                
                stop_loss = self