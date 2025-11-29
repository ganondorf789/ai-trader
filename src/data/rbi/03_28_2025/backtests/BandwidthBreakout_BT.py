```python
# 🌙 Moon Dev's BandwidthBreakout Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
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
data = data.set_index('datetime')

class BandwidthBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌟 Core Indicators 🌟
        # Bollinger Bands (20,2)
        self.bb_upper, self.bb_middle, self.bb_lower = talib.BBANDS(
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        self.I(lambda: self.bb_upper, name='BB_UPPER')
        self.I(lambda: self.bb_middle, name='BB_MID')
        self.I(lambda: self.bb_lower, name='BB_LOWER')
        
        # 🔍 Bandwidth Calculation
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.I(lambda: self.bb_width, name='BB_WIDTH')
        
        # 📉 20-day Bandwidth Low
        self.bb_width_low = talib.MIN(self.bb_width, timeperiod=20)
        self.I(lambda: self.bb_width_low, name='BB_WIDTH_LOW')
        
        # 🔊 Volume Filter
        self.volume_sma = talib.SMA(self.data.Volume, timeperiod=10)
        self.I(lambda: self.volume_sma, name='VOL_SMA_10')
        
        # 🛑 ATR for Risk Management
        self.atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, 14)
        self.I(lambda: self.atr, name='ATR_14')

    def next(self):
        # 🕒 Wait for indicators to warm up
        if len(self.data) < 20:
            return
        
        # 🌙 Entry Logic
        if not self.position:
            # Long Entry Conditions 🚀
            if (self.bb_width[-1] == self.bb_width_low[-1] and
                self.data.Close[-1] > self.bb_upper[-1] and
                self.data.Volume[-1] > 1.5 * self.volume_sma[-1]):
                
                # Risk Management Calculations 🌙
                atr = self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = 2 * atr
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🌙✨ MOON DEV LONG ENTRY @ {self.data.Close[-1]:.2f} | Size: {position_size} BTC 🚀")
            
            # Short Entry Conditions 🌌
            elif (self.bb_width[-1] == self.bb_width_low[-1] and
                  self.data.Close[-1] < self.bb_lower[-1] and
                  self.data.Volume[-1] > 1.5 * self.volume_sma[-1]):
                
                # Risk Management Calculations 🌙
                atr = self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = 2 * atr
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size)
                    print(f"🌙✨ MOON DEV SHORT ENTRY @ {self.data.Close[-1]:.2f} | Size: {position_size} BTC 🌌")
        
        # 🌑 Exit Logic
        else:
            for trade in self.trades:
                if trade.is_long:
                    # Long Exit: 2x ATR below highest high since entry 🌙