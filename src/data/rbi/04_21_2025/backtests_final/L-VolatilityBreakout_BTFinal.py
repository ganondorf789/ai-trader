```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilityBreakout(Strategy):
    def init(self):
        # Donchian Channel
        self.donchian_upper = self.I(talib.MAX, self.data.High, 20, name='Donchian Upper')
        self.donchian_lower = self.I(talib.MIN, self.data.Low, 20, name='Donchian Lower')
        
        # Bollinger Bands components
        def upper_bb(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.upper_bb = self.I(upper_bb, self.data.Close, name='Upper BB')
        
        def lower_bb(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.lower_bb = self.I(lower_bb, self.data.Close, name='Lower BB')
        
        def middle_bb(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return middle
        self.middle_bb = self.I(middle_bb, self.data.Close, name='Middle BB')
        
        # Bollinger Band Width
        def bb_width_calc(upper, lower, middle):
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width_calc, self.upper_bb, self.lower_bb, self.middle_bb, name='BB Width')
        
        # 6-month low of BB Width (12096 periods for 15m data)
        self.bb_width_min = self.I(talib.MIN, self.bb_width, 12096, name='BB Width Min')
        
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')

    def next(self):
        if not self.position:
            # Long entry logic
            if self.data.Close[-1] > self.donchian_upper[-1] and self.bb_width[-1] <= self.bb_width_min[-1]:
                risk_percent = 0.01
                risk_amount = self.equity * risk_percent
                entry_price = self.data.Close[-1]
                stop_loss = self.donchian_upper[-1]
                risk_per_share = entry_price - stop_loss
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        tp_price = entry_price + 1.5 * self.atr[-1]
                        self.buy(size=position_size, tp=tp_price, sl=stop_loss)
                        print(f"🚀🌙 Moon Dev LONG Entry! | Price: {entry_price:.2f} | Size: {position_size} | TP: {tp_price:.2f} | SL: {stop_loss:.2f}")
            
            # Short entry logic
            elif self.data.Close[-1] < self.donchian_lower[-1] and self.bb_width[-1] <= self.bb_width_min[-1]:
                risk_percent = 0.01
                risk_amount = self.equity * risk_percent
                entry_price = self.data.Close[-1]
                stop_loss = self.donchian_lower[-1]
                risk_per_share = stop_loss - entry_price
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        tp_price = entry_price - 1.5 * self.atr[-1]
                        self.sell(size=position_size, tp=tp_price, sl=stop_loss)
                        print(f