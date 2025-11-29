```python
import pandas as pd
from backtesting import Strategy, Backtest
import talib
import pandas_ta as ta

# Data Handling
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

class VortexFlow(Strategy):
    def init(self):
        # Clean data columns
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        
        # 1. Vortex Indicator (VI+, VI-)
        vortex = self.data.ta.vortex(length=14)
        self.vi_plus = self.I(lambda: vortex['VIPT_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VIMN_14'], name='VI-')
        
        # 2. Bollinger Bands + Bandwidth
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        self.bb_bandwidth = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_bandwidth_sma = self.I(talib.SMA, self.bb_bandwidth, 20)
        
        # 3. Funding Rate Momentum
        self.funding_roc = self.I(talib.ROC, self.data['funding_rate'], 3)
        
        # 4. Netflow Divergence Indicators
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        self.netflow_low = self.I(talib.MIN, self.data['netflow'], 20)
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.netflow_high = self.I(talib.MAX, self.data['netflow'], 20)
        
        # Volatility Stop (ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def next(self):
        # Moon Dev Debug Prints 🌙
        print(f"\n🌙 Moon Dev Status Update 🌙\nCurrent Close: {self.data.Close[-1]}")
        
        # Entry Conditions
        vi_cross_up = self.vi_plus[-1] > self.vi_minus[-1] and self.vi_plus[-2] <= self.vi_minus[-2]
        vi_cross_down = self.vi_plus[-1] < self.vi_minus[-1] and self.vi_plus[-2] >= self.vi_minus[-2]
        
        # Netflow Divergence Checks
        bull_div = (self.swing_low[-1] < self.swing_low[-21]) and (self.netflow_low[-1] > self.netflow_low[-21]) if len(self.swing_low) > 21 else False
        bear_div = (self.swing_high[-1] > self.swing_high[-21]) and (self.netflow_high[-1] < self.netflow_high[-21]) if len(self.swing_high) > 21 else False
        
        # Risk Management Calculations
        atr = self.atr[-1] if len(self.atr) > 0 else 0
        equity = self.equity
        
        # Long Entry
        if not self.position and vi_cross_up and bull_div:
            sl_price = self.data.Close[-1] - (atr * 1.5)
            risk_amount = equity * 0.01
            position_size = int(round(risk_amount / (self.data.Close[-1] - sl_price)))
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price, tp=self.data.Close[-1] + (atr * 3))
                print(f"🚀🌙 MOON DEV LONG ENTRY! Size: {position_size}")
        
        # Short Entry
        elif not self.position and vi_cross_down and bear_div:
            sl