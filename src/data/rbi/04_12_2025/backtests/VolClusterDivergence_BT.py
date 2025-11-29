```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolClusterDivergence(Strategy):
    def init(self):
        # Calculate indicators with proper self.I() wrapper
        self.upper_cluster = self.I(talib.MAX, self.data.High, timeperiod=20, name='Upper Cluster 🌌')
        self.lower_cluster = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Lower Cluster 🕳️')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume MA 🔊')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14 ⚡')
        
        # Funding rate indicators
        self.funding_mean = self.I(talib.SMA, self.data.funding_rate, timeperiod=20, name='Funding Mean 🧮')
        self.funding_std = self.I(talib.STDDEV, self.data.funding_rate, timeperiod=20, name='Funding Std 📊')
        
        # Volatility index MA
        self.volatility_ma = self.I(talib.SMA, self.data.volatility_index, timeperiod=20, name='Vol MA 🌋')

    def next(self):
        # Moon Dev Debug Prints
        print(f"\n🌙 Lunar Cycle {len(self.data)} - Close: {self.data.Close[-1]:.2f}")
        
        # Skip initial bars
        if len(self.data) < 20:
            return

        # Current values
        close = self.data.Close[-1]
        funding = self.data.funding_rate[-1]
        vol_idx = self.data.volatility_index[-1]

        # Entry conditions
        long_cond = (close > self.upper_cluster[-1] and
                    self.data.Volume[-1] > self.volume_ma[-1] and
                    (funding > 0.05 or funding > (self.funding_mean[-1] + 2*self.funding_std[-1])) and
                    vol_idx > self.volatility_ma[-1])

        short_cond = (close < self.lower_cluster[-1] and
                     self.data.Volume[-1] > self.volume_ma[-1] and
                     (funding < -0.05 or funding < (self.funding_mean[-1] - 2*self.funding_std[-1])) and
                     vol_idx > self.volatility_ma[-1]

        # Risk management calculations
        risk_pct = 0.01  # 1% risk per trade
        atr_value = self.atr[-1]
        stop_loss = 0.75 * atr_value
        position_size = int(round((self.equity * risk_pct) / stop_loss))

        # Entry logic
        if not self.position and len(self.trades) < 3:
            if long_cond:
                tp = close + 1.5 * atr_value
                self.buy(size=position_size, sl=close-stop_loss, tp=tp)
                print(f"🚀🌕 MOONSHOT LONG! Size: {position_size} | SL: {close-stop_loss:.1f} | TP: {tp:.1f}")
                
            elif short_cond:
                tp = close - 1.5 * atr_value
                self.sell(size=position_size, sl=close+stop_loss, tp=tp)
                print(f"🌑🛸 DARK SIDE SHORT! Size: {position_size} | SL: {close+stop_loss:.1f} | TP: {tp:.1f}")

        # Exit logic
        for trade in self.trades:
            if trade.is_long:
                if (close < self.upper_cluster[-1] or 
                    abs(funding - self.funding_mean[-1]) < self.funding_std[-1]):
                    trade.close()
                    print(f"🌓 Closing LONG: Cluster/Funding Exit")
            else:
                if (close > self.lower_cluster[-1] or 
                    abs(funding - self.funding_mean[-1]) < self.funding_std[-1]):
                    trade.close()
                    print(f"🌗 Closing SHORT: Cluster/Funding Exit")

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col