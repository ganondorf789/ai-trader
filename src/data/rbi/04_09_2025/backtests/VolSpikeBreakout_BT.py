```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolSpikeBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands with 20-period, 2 std dev
        self.bb_upper, _, self.bb_lower = self.I(self.calculate_bbands, self.data.Close)
        
        # Volume 90th percentile (20-period lookback)
        self.vol_pct = self.I(self.calculate_volume_pct, self.data.Volume)
        
    def calculate_bbands(self, close):
        upper, middle, lower = talib.BBANDS(
            close, 
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        return upper, middle, lower
    
    def calculate_volume_pct(self, volume):
        return volume.rolling(20).quantile(0.9)
    
    def next(self):
        price = self.data.Close[-1]
        upper_band = self.bb_upper[-1]
        lower_band = self.bb_lower[-1]
        volume = self.data.Volume[-1]
        vol_threshold = self.vol_pct[-1]
        
        # Moon Dev Debug Prints 🌙
        print(f"🌙 Current Price: {price:.2f} | Upper BB: {upper_band:.2f} | Lower BB: {lower_band:.2f}")
        print(f"🌕 Volume: {volume:.2f} | 90th Percentile: {vol_threshold:.2f}")
        
        if not self.position:
            # Long entry logic
            if price > upper_band and volume > vol_threshold:
                band_width = upper_band - lower_band
                sl_price = price - 0.75 * band_width
                risk_per_share = price - sl_price
                position_size = (self.equity * self.risk_percent) / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    tp_price = price + 1.5 * band_width
                    self.buy(size=position_size, sl=sl_price, tp=tp_price)
                    print(f"\n🚀🌕 MOON DEV LONG SIGNAL 🚀")
                    print(f"Entry: {price:.2f} | Size: {position_size}")
                    print(f"TP: {tp_price:.2f} | SL: {sl_price:.2f}\n")
            
            # Short entry logic
            elif price < lower_band and volume > vol_threshold:
                band_width = upper_band - lower_band
                sl_price = price + 0.75 * band_width
                risk_per_share = sl_price - price
                position_size = (self.equity * self.risk_percent) / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    tp_price = price - 1.5 * band_width
                    self.sell(size=position_size, sl=sl_price, tp=tp_price)
                    print(f"\n🔻🌕 MOON DEV SHORT SIGNAL 🔻")
                    print(f"Entry: {price:.2f} | Size: {position_size}")
                    print(f"TP: {tp_price:.2f} | SL: {sl_price:.2f}\n")
        
        # Close position if price re-enters bands
        else:
            if self.position.is_long and price < upper_band:
                self.position.close()
                print(f"\n🌙📉 MOON DEV EXIT: Price returned inside bands at {price:.2f}\n")
            elif self.position.is_short and price > lower_band:
                self.position.close()
                print(f"\n🌙📈 MOON DEV EXIT: Price returned inside bands at {price:.