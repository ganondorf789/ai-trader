```python
# backtesting.py implementation for SqueezeDivergence strategy
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preprocessing
def load_data():
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
    data = data.set_index('datetime')
    return data

class SqueezeDivergence(Strategy):
    risk_percent = 0.01
    volatility_expansion_mult = 1.5
    trailing_stop_pct = 0.03
    
    def init(self):
        # Bollinger Bands (20,2)
        self.upper_band = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='UPPER', priceic=0)
        self.lower_band = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='LOWER', priceic=2)
        self.bb_width = self.I(lambda upper, lower: upper - lower, self.upper_band, self.lower_band, name='BB_WIDTH')
        
        # Chaikin Money Flow (21)
        self.cmf = self.I(ta.cmf, self.data.High, self.data.Low, self.data.Close, self.data.Volume, length=21, name='CMF')
        
        # Swing highs/lows for trailing stops
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=5, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='SWING_LOW')

    def next(self):
        if not self.position:
            # Entry logic
            current_close = self.data.Close[-1]
            squeeze_condition = self.bb_width[-1] < self.bb_width[-5:].mean() * 0.5
            in_bands = self.lower_band[-1] < current_close < self.upper_band[-1]
            
            # CMF divergence check (simplified)
            bear_div = current_close > self.data.Close[-2] and self.cmf[-1] < self.cmf[-2]
            bull_div = current_close < self.data.Close[-2] and self.cmf[-1] > self.cmf[-2]
            
            if squeeze_condition and in_bands:
                # Long entry
                if bull_div and current_close > self.upper_band[-1]:
                    risk_amount = self.risk_percent * self.equity
                    stop_price = self.lower_band[-1]
                    risk_per_share = current_close - stop_price
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price)
                        self.entry_width = self.bb_width[-1]
                        print(f"🚀🌙 BULLISH BREAKOUT | Size: {position_size} | Entry: {current_close:.2f}")
                
                # Short entry    
                elif bear_div and current_close < self.lower_band[-1]:
                    risk_amount = self.risk_percent * self.equity
                    stop_price = self.upper_band[-1]
                    risk_per_share = stop_price - current_close
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_price)
                        self.entry_width = self.bb_width[-1]
                        print(f"🔻🌙 BEARISH BREAKOUT | Size: {position_size} | Entry: {current_close:.2f}")
        else:
            # Exit logic
            if self.position.is_long:
                # Volatility expansion exit
                if self.bb_width[-1] > self.entry_width * self.volatility_expansion_mult:
                    self.position.close()
                    print(f"✨🌙 VOL EXPANS