```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

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

class LiquidityGapBreach(Strategy):
    max_positions = 5
    risk_per_trade = 0.01
    
    def init(self):
        # 🌙 Core Indicators
        # Bollinger Bands
        self.bb_upper, _, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                 timeperiod=20, nbdevup=2, nbdevdn=2, 
                                                 name=['BB_UP', 'BB_MID', 'BB_LOW'])
        
        # Volatility Indicators
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                           timeperiod=14, name='MOON_ATR')
        self.median_atr = self.I(lambda x: x.rolling(20).median(), self.atr14, 
                                name='MEDIAN_ATR')
        
        # Keltner Channel
        self.keltner_mid = self.I(talib.EMA, self.data.Close, timeperiod=20, 
                                 name='KC_MID')
        self.keltner_atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                                 timeperiod=20, name='KC_ATR')
        
        print("🌙✨ Lunar indicators activated! Ready to breach gaps! 🚀")

    def next(self):
        # 🌙 Trade management
        if len(self.trades) >= self.max_positions:
            return

        # Previous period values
        prev_idx = -1
        close = self.data.Close[prev_idx]
        atr = self.atr14[prev_idx]
        med_atr = self.median_atr[prev_idx]
        bb_upper = self.bb_upper[prev_idx]
        bb_lower = self.bb_lower[prev_idx]

        # 🌙 Entry logic
        if close > bb_upper and atr >= 1.5 * med_atr:
            self.moon_long()
        elif close < bb_lower and atr >= 1.5 * med_atr:
            self.moon_short()

        # 🌙 Exit logic
        for trade in self.trades:
            if trade.is_long:
                self.long_exits(trade)
            else:
                self.short_exits(trade)

    def moon_long(self):
        risk_amount = self.risk_per_trade * self.equity
        atr = self.atr14[-1]
        sl = self.data.Close[-1] - 1.5 * atr
        position_size = int(round(risk_amount / (1.5 * atr)))
        
        if position_size > 0:
            self.buy(size=position_size, sl=sl, 
                    tag=f"🌕 LONG | SL:{sl:.1f}",)
            print(f"🚀🌕 BLAST OFF! Long {position_size} units at {self.data.Open[0]}!")

    def moon_short(self):
        risk_amount = self.risk_per_trade * self.equity
        atr = self.atr14[-1]
        sl = self.data.Close[-1] + 1.5 * atr
        position_size = int(round(risk_amount / (1.5 * atr)))
        
        if position_size > 0:
            self.sell(size=position_size, sl=sl,
                     tag=f"🌑 SHORT | SL:{sl:.1f}")
            print(f"🌌🌑 BLACK HOLE ENGAGED! Short {position_size} units at {self.data.Open[0]}!")

    def long_exits(self, trade):
        current_low = self.data.Low[0]
        kc_mid = self.keltner_mid[0]
        
        # Mean reversion exit
        if current_low <= kc_mid:
            self.sell(size=trade.size, tag