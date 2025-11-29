```python
# 🌙✨ MOON DEV BACKTESTING SCRIPT FOR VOLTAIC SQUEEZE STRATEGY ✨🌙

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# 🚀 DATA PREPARATION FOR MOON DEV ANALYSIS
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Cleanse and align cosmic data energies 🌌
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['DateTime'] = pd.to_datetime(data['datetime'])
data.set_index('DateTime', inplace=True)

class VoltaicSqueeze(Strategy):
    def init(self):
        # 🌗 CELESTIAL INDICATORS CALCULATION
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[2], self.data.Close)
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.min_width = self.I(talib.MIN, self.bb_width, 2880)  # 30-day window
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 1920)  # 20-day average
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)

        # 🌠 INITIALIZE COSMIC PARAMETERS
        self.risk_per_trade = 0.01
        self.trailing_multiplier = 2

    def next(self):
        # 🌑 REQUIRE MINIMUM COSMIC ALIGNMENT (30 days of data)
        if len(self.data) < 2880:
            return

        # 🌕 CURRENT STAR ALIGNMENTS
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        width = self.bb_width[-1]
        min_width = self.min_width[-1]
        volume_spike = self.data.Volume[-1] > self.vol_ma[-1] * 1.5

        # 🌙✨ POSITION MANAGEMENT ORBIT
        for trade in self.trades:
            if trade.is_long:
                # 🌓 LONG EXIT CONDITIONS
                if price < upper or \
                price <= trade.sl:
                    print(f"🌙📉 MOON DEV LONG EXIT | Price: {price:.2f}")
                    trade.close()
            else:
                # 🌗 SHORT EXIT CONDITIONS
                if price > lower or \
                price >= trade.sl:
                    print(f"🌙📈 MOON DEV SHORT EXIT | Price: {price:.2f}")
                    trade.close()

        # 🌌 NEW STARGAZING OPPORTUNITIES
        if not self.position:
            # 🚀 LONG ENTRY PROTOCOL
            if width == min_width and price > upper and volume_spike:
                atr = self.atr[-1]
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / (1.5 * atr)))
                
                if position_size > 0:
                    sl = price - 1.5 * atr
                    print(f"🚀🌙 MOON DEV LONG ENTRY | Size: {position_size} | SL: {sl:.2f}")
                    self.buy(size=position_size, sl=sl, tag="VoltaicLong")

            # 🌑 SHORT ENTRY PROTOCOL
            elif width == min_width and price < lower and volume_spike:
                atr = self.atr[-1]
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / (1.5 * atr)))
                
                if position_size > 0:
                    sl = price + 1.5 * atr
                    print(f"🌑💫 MOON DEV SHORT ENTRY | Size: {position_size} | SL: {