```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as ta

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean data columns
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

class LiquidationBreakout(Strategy):
    def init(self):
        # Moon Dev Magic Indicators 🌙✨
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='SWING LOW')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        # Volume analysis
        self.volume_z = self.I(ta.zscore, self.data.Volume, length=10, name='VOL Z')
        
        # Volatility calculations
        price_change = self.I(talib.MOM, self.data.Close, 1, name='ΔPRICE')
        vol_std = self.I(talib.STDDEV, price_change, 20, name='VOL STD')
        self.vol_vol = self.I(lambda x,y: x*y, vol_std, self.data.Volume, name='VOL*STD')
        self.vol_vol_avg = self.I(talib.SMA, self.vol_vol, 20, name='VOL VOL AVG')
        
        print("🌙✨ Lunar indicators activated! Ready for quantum trading!")

    def next(self):
        # Avoid weekend liquidity voids 🌌
        if self.data.index[-1].weekday() >= 5:
            if self.position:
                self.position.close()
                print("🌙🌃 Closing positions before weekend stargazing...")
            return

        close = self.data.Close[-1]
        atr = self.atr[-1]
        vv = self.vol_vol[-1]
        vv_avg = self.vol_vol_avg[-1]
        
        # Moon Launch Entry Sequences 🚀
        if not self.position:
            # Long ignition sequence
            if close > self.swing_high[-1] and vv > 1.5*vv_avg and self.volume_z[-1] > 2:
                sl = close - atr
                risk = close - sl
                size = int(round((self.equity * 0.01) / risk))
                if size > 0:
                    self.buy(size=size, sl=sl, tag='🌕 LONG')
                    print(f"🚀🌙 LIFTOFF! Long {size} @ {close} | Cosmic SL: {sl}")
            
            # Short gravity well entry
            elif close < self.swing_low[-1] and vv > 1.5*vv_avg and self.volume_z[-1] > 2:
                sl = close + atr
                risk = sl - close
                size = int(round((self.equity * 0.01) / risk))
                if size > 0:
                    self.sell(size=size, sl=sl, tag='🌑 SHORT')
                    print(f"🌙🌌 BLACK HOLE! Short {size} @ {close} | Event Horizon SL: {sl}")

        # Stellar Position Management 🌠
        for trade in self.trades:
            if trade.is_long and close < self.swing_low[-1]:
                trade.close()
                print(f"🌙💫 LONG DOCKING! Support breached @ {close}")
            elif trade.is_short and close > self.swing_high[-1]:
                trade.close()
                print(f"🌙💥 SHORT EJECTION! Resistance broken @ {close}")
            
            # Trailing event horizon 🔄
            if trade.pnl_percent >= 1.5 * (atr/trade.entry_price)*100:
                trade.sl = trade.entry_price
                print(f"🌙🛡️ SHIELD UP! Breakeven activated @ {trade.entry_price}")