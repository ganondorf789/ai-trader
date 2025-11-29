Here's the fully debugged and Moon Dev themed backtest code for the Liquidation Volatility strategy:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATION VOLATILITY STRATEGY 🚀✨
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class LiquidationVolatility(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2     # Risk-reward ratio
    
    def init(self):
        # 🌙 BOLLINGER BANDWIDTH CALCULATION ✨
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, which=2)
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.bandwidth = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.sma20)
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, timeperiod=20)

        # 🚀 LIQUIDATION ZONES USING SWING HIGHS/LOWS ✨
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)

        # 🌙 VOLUME ANALYSIS ✨
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
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
            if risk_per_share <= 0: 
                print("🌑 Warning: Invalid SL price for long trade")
                return
            
            position_size = int(round((self.equity * self.risk_pct) / risk_per_share))
            tp_price = self.data.Close[-1] + self.rr_ratio * risk_per_share
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price, tp=tp_price)
                print(f"🌙🚀 BULLISH BREAKOUT! Entry: {self.data.Close[-1]:.2f}, "
                      f"Size: {position_size} units, SL: {sl_price:.2f}, TP: {tp_price:.2f} ✨")

        # 🌑 SHORT SIGNAL: Breakdown below liquidation zone ✨
        elif (bandwidth_cond and volume_cond and
              self.data.Close[-1] < self.swing_low[-1]):
            
            sl_price = self.swing_high[-1]
            risk_per_share = sl_price - self.data.Close[-1]
            if risk_per_share <= 0:
                print("🌑 Warning: Invalid SL price for short trade")
                return
            
            position_size = int(round((self.equity * self.risk_pct) / risk_per_share))
            tp_price = self.data.Close[-1] - self.rr_ratio * risk_per_share
            
            if position_size > 0:
                self.sell(size=position_size, sl=sl_price, tp=tp_price)
                print(f"🌑🚀 BEARISH BREAKDOWN! Entry: {self.data.Close[-1]:.2f}, "
                      f"Size: {position_size} units, SL: {sl_price:.2f}, TP: {tp_price:.2f} ✨")

# 🌙 DATA LOADING AND PREPROCESSING ✨
print("🌙 Loading celestial market data...")
data = pd.read_csv('BTC-USD-15m.csv