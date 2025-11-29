Here's the debugged version of your VolatilityGapFiller strategy with Moon Dev themed improvements:

```python
import numpy as np
import pandas as pd
import talib
from backtesting import Strategy, Backtest

# Data preprocessing
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

class VolatilityGapFiller(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade (fractional sizing)
    max_drawdown_daily = 0.05  # 5% daily circuit breaker
    max_drawdown_weekly = 0.15  # 15% weekly circuit breaker
    
    def init(self):
        # Bollinger Bands components
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        
        # BB Width Percentile
        def calc_bb_width(close):
            upper, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / ((upper + lower)/2)
        self.bb_width = self.I(calc_bb_width, self.data.Close)
        self.bb_width_pct = self.I(lambda x: x.rolling(20).rank(pct=True)*100, self.bb_width)
        
        # Volume Profile Percentile
        self.volume_pct = self.I(lambda x: x.rolling(200).rank(pct=True)*100, self.data.Volume)
        
        # Fractal Levels
        self.fractal_high = self.I(talib.MAX, self.data.High, timeperiod=5)
        self.fractal_low = self.I(talib.MIN, self.data.Low, timeperiod=5)
        
        # Volatility Measure
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙 Moon Dev Indicators Initialized! Ready for cosmic analysis ✨")

    def next(self):
        if len(self.data.Close) < 20:
            return
            
        current_close = self.data.Close[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bb_width_pct = self.bb_width_pct[-1]
        volume_pct = self.volume_pct[-1]
        
        # Moon Dev Circuit Breaker Check
        if self.equity < self.initial_equity * (1 - self.max_drawdown_weekly):
            print("🚨 COSMIC ALERT: Weekly Drawdown Limit Reached! Closing all positions 🌌")
            self.position.close()
            return
            
        if not self.position:
            # Long Entry Logic
            if (current_close > bb_upper and 
                bb_width_pct > 70 and 
                volume_pct < 30):
                
                sl = self.fractal_low[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_close - sl
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl, tag="🌙 LONG: Galactic Breakout!")
                        print(f"✨ Moon Dev Trade Opened: Size={position_size}, SL={sl:.2f}")
            
            # Short Entry Logic        
            elif (current_close < bb_lower and 
                  bb_width_pct > 70 and 
                  volume_pct