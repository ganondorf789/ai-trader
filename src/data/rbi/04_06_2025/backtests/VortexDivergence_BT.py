```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preprocessing
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Vortex Indicator
        self.vi_plus = self.I(talib.VORTEX, self.data.High, self.data.Low, self.data.Close, 14, name='VI+', which=0)
        self.vi_minus = self.I(talib.VORTEX, self.data.High, self.data.Low, self.data.Close, 14, name='VI-', which=1)
        
        # Chaikin Money Flow
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 20, name='CMF')
        
        # Bollinger Bands Width
        upper = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_Upper', which=0)
        lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BB_Lower', which=2)
        self.bbw = self.I(lambda: (upper - lower) / self.I(talib.SMA, self.data.Close, 20), name='BBW')
        self.bbw_sma = self.I(talib.SMA, self.bbw, 20, name='BBW_SMA')
        
        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
    def next(self):
        price = self.data.Close[-1]
        
        # Exit conditions
        if self.position:
            if self.bbw[-1] < 0.5 * self.bbw_sma[-1]:
                self.position.close()
                print(f"🌙✨ Moon Dev Exit Signal! Closing position at {price:.2f} (BBW contraction)")
                
        # Entry conditions
        else:
            # Vortex crossover check
            vi_cross_bull = crossover(self.vi_plus, self.vi_minus)
            vi_cross_bear = crossover(self.vi_minus, self.vi_plus)
            
            # CMF divergence check
            current_low, prev_low = self.data.Low[-1], self.data.Low[-2]
            current_cmf, prev_cmf = self.cmf[-1], self.cmf[-2]
            bull_div = (current_low < prev_low) and (current_cmf > prev_cmf)
            
            current_high, prev_high = self.data.High[-1], self.data.High[-2]
            bear_div = (current_high > prev_high) and (current_cmf < prev_cmf)
            
            # Long entry logic
            if vi_cross_bull and bull_div:
                atr = self.atr[-1]
                entry_price = self.data.Open[-1]
                stop_loss = entry_price - 1.5 * atr
                risk_amount = self.equity * self.risk_per_trade
                position_size = int(round(risk_amount / (entry_price - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🚀🌙 Moon Dev LONG Entry! Size: {position_size} @ {entry_price:.2f}")
            
            # Short entry logic
            elif vi_cross_bear and bear_div:
                atr = self.atr[-1]
                entry_price = self.data.Open[-1]
                stop_loss = entry_price + 1.5 * atr
                risk_amount = self.equity * self.risk_per_trade
                position_size = int(round(risk_amount / (stop_loss - entry_price)))
                
                if position_size > 0: