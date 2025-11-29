```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

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

class LiquidationBreakout(Strategy):
    def init(self):
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        self.bb_mid = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_mid)
        
        # Liquidation zones
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        # Funding rate indicators
        self.funding = self.data['funding_rate']
        self.funding_ma = self.I(talib.SMA, self.funding, 20)

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if not self.position:
            # Entry conditions
            vol_contraction = self.bb_width[-1] < 0.5 * atr
            liq_cluster_long = price > self.swing_high[-1]
            funding_spike = self.funding[-1] > self.funding_ma[-1] * 1.3
            
            if vol_contraction and liq_cluster_long and funding_spike:
                sl = self.swing_low[-1]
                risk = price - sl
                position_size = int(round((0.01 * self.equity) / risk))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tp=price + 1.5*atr)
                    print(f"🌙🚀 LONG ENTRY! Price: {price:.2f}, Size: {position_size}, Moon Power: {self.funding[-1]:.4f}")
            
            liq_cluster_short = price < self.swing_low[-1]
            funding_drop = self.funding[-1] < self.funding_ma[-1] * 0.7
            
            if vol_contraction and liq_cluster_short and funding_drop:
                sl = self.swing_high[-1]
                risk = sl - price
                position_size = int(round((0.01 * self.equity) / risk))
                if position_size > 0:
                    self.sell(size=position_size, sl=sl, tp=price - 1.5*atr)
                    print(f"🌙🌑 SHORT ENTRY! Price: {price:.2f}, Size: {position_size}, Dark Energy: {self.funding[-1]:.4f}")
        else:
            # Exit conditions
            if self.position.pl_pct >= 0.75 * (atr/price):
                self.position.close()
                print(f"✨💫 PROFIT MOONSHOT! Closed at {price:.2f}")
            elif (self.position.is_long and self.funding[-1] < self.funding_ma[-1]) or \
                 (self.position.is_short and self.funding[-1] > self.funding_ma[-1]):
                self.position.close()
                print(f"🌙⚠️ FUNDING REVERSION! Closed at {price:.2f}")

bt = Backtest(data, LiquidationBreakout, cash=1_000_000)
stats = bt.run()
print