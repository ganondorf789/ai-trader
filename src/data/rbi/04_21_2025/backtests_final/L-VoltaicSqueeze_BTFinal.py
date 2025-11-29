from backtesting import Backtest, Strategy
import pandas as pd
import talib

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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
data.set_index('datetime', inplace=True)

class VoltaicSqueeze(Strategy):
    risk_pct = 0.01
    max_positions = 5
    
    def init(self):
        # Bollinger Bands
        close = self.data.Close
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, name='BB_UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, name='BB_LOWER', which=2)
        
        # Keltner Channels
        high, low = self.data.High, self.data.Low
        typical_price = (high + low + close) / 3
        kc_mid = self.I(talib.EMA, typical_price, timeperiod=20, name='KC_MID')
        atr = self.I(talib.ATR, high, low, close, timeperiod=20)
        self.kc_upper = self.I(lambda: kc_mid + 1.5*atr, name='KC_UPPER')
        self.kc_lower = self.I(lambda: kc_mid - 1.5*atr, name='KC_LOWER')
        
        # Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_SMA')
        
        # Chande Kroll Stop
        self.long_stop = self.I(lambda: talib.MAX(high, 9) - 3*talib.ATR(high, low, close, 10), name='LONG_STOP')
        self.short_stop = self.I(lambda: talib.MIN(low, 9) + 3*talib.ATR(high, low, close, 10), name='SHORT_STOP')
        
        self.trailing_stops = {}

    def next(self):
        if len(self.trades) >= self.max_positions:
            return
        
        # Check volatility squeeze
        if len(self.data.Close) < 21:
            return
            
        prev_bb_in_kc = (self.bb_upper[-2] < self.kc_upper[-2]) and (self.bb_lower[-2] > self.kc_lower[-2])
        vol_spike = self.data.Volume[-1] > 3 * self.vol_sma[-1]
        
        # Long entry
        if prev_bb_in_kc and self.data.Close[-1] > self.bb_upper[-1] and vol_spike:
            stop = self.long_stop[-1]
            risk = self.data.Close[-1] - stop
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size > 0:
                self.buy(size=size, sl=stop)
                print(f"🌙✨ MOON DEV LONG ENTRY! ✨ Size: {size} @ {self.data.Close[-1]:.2f} 🚀")
        
        # Short entry
        if prev_bb_in_kc and self.data.Close[-1] < self.bb_lower[-1] and vol_spike:
            stop = self.short_stop[-1]
            risk = stop - self.data.Close[-1]
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size > 0:
                self.sell(size=size, sl=stop)
                print(f"🌙✨ MOON DEV SHORT ENTRY! ✨ Size: {size} @ {self.data.Close[-1]:.2f} 🚀")
        
        # Update trailing stops
        for trade in self.t