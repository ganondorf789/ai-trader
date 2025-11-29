```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATION VOLATILITY STRATEGY 🚀✨
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class LiquidationVolatility(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2     # Risk-reward ratio
    
    def init(self):
        # 🌙 BOLLINGER BANDWIDTH CALCULATION ✨
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_LOWER', which=2)
        self.bandwidth = self.I(lambda u, l: (u - l)/self.I(talib.SMA, self.data.Close, 20),
                              self.bb_upper, self.bb_lower)
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, 20)

        # 🚀 LIQUIDATION ZONES USING SWING HIGHS/LOWS ✨
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)

        # 🌙 VOLUME ANALYSIS ✨
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        print("🌙✨ Strategy initialized with Moon Dev magic! Ready to launch! 🚀")

    def next(self):
        if self.position:
            return  # Hold existing position

        # 🌙 STRATEGY CONDITIONS ✨
        bandwidth_cond = self.bandwidth[-1] < self.bandwidth_ma[-1]
        volume_cond = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]
        
        # 🚀 LONG SIGNAL: Breakout above liquidation zone ✨
        if (bandwidth_cond and volume_cond and
            self.data.Close[-1] > self.swing_high[-1]):
            
            sl_price = self.swing_low[-1]
            risk_per_share = self.data.Close[-1] - sl_price
            if risk_per_share <= 0: return
            
            position_size = int(round((self.equity * self.risk_pct) / risk_per_share))
            tp_price = self.data.Close[-1] + self.rr_ratio * risk_per_share
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price, tp=tp_price)
                print(f"🌙🚀 BULLISH BREAKOUT! Entry: {self.data.Close[-1]:.2f}, "
                      f"SL: {sl_price:.2f}, TP: {tp_price:.2f} ✨")

        # 🌑 SHORT SIGNAL: Breakdown below liquidation zone ✨
        elif (bandwidth_cond and volume_cond and
              self.data.Close[-1] < self.swing_low[-1]):
            
            sl_price = self.swing_high[-1]
            risk_per_share = sl_price - self.data.Close[-1]
            if risk_per_share <= 0: return
            
            position_size = int(round((self.equity * self.risk_pct) / risk_per_share))
            tp_price = self.data.Close[-1] - self.rr_ratio * risk_per_share
            
            if position_size > 0:
                self.sell(size=position_size, sl=sl_price, tp=tp_price)
                print(f"🌑🚀 BEARISH BREAKDOWN! Entry: {self.data.Close[-1]:.2f}, "
                      f"SL: {sl_price:.2f}, TP: {tp_price:.2f} ✨")

# 🌙 DATA PREPROCESSING MAGIC ✨
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set