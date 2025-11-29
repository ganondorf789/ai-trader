I'll complete and fix the LiquiditySqueeze strategy code while maintaining all the original strategy logic. Here's the fully debugged version with Moon Dev improvements:

```python
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

# Load and prepare data with cosmic precision 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns with lunar accuracy 🌙
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

class LiquiditySqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade - cosmic risk management 🌠
    
    def init(self):
        # Bollinger Bands components - stellar volatility indicators 🌟
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close)
        
        # Bollinger Bandwidth calculations - measuring cosmic compression 🌌
        self.bb_bandwidth = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.bb_bandwidth_ma = self.I(talib.SMA, self.bb_bandwidth, 20)
        
        # Keltner Channels components - intergalactic trend indicators 🚀
        self.ema = self.I(talib.EMA, self.data.Close, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.kc_upper = self.I(lambda e, a: e + 1.5*a, self.ema, self.atr)
        self.kc_lower = self.I(lambda e, a: e - 1.5*a, self.ema, self.atr)
        
        # Liquidity zones using swing levels - lunar gravity wells 🌑
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("🌙✨ Moon Dev Indicators Activated! Trading systems at warp speed 🚀🌌")

    def next(self):
        if len(self.data) < 20:
            return
            
        price = self.data.Close[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        kc_upper = self.kc_upper[-1]
        kc_lower = self.kc_lower[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        
        # Calculate squeeze condition - cosmic compression detected! 🌠
        squeeze_active = (self.bb_bandwidth[-1] < 0.5*self.bb_bandwidth_ma[-1])
        
        if not self.position:
            # Long entry logic - preparing for lunar launch 🚀
            if squeeze_active and price > bb_upper and price >= 0.99*swing_high:
                sl = bb_lower * 0.995
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = price - sl
                size = int(round(risk_amount / risk_per_share))
                if size > 0:
                    self.buy(size=size, sl=sl)
                    print(f"🌙🚀 BLASTOFF LONG! Entry: {price:.2f}, Size: {size} contracts (Cosmic Profit Potential: {(swing_high-price)/price*100:.2f}%) 🌌")
            
            # Short entry logic - comet short strategy ☄️
            elif squeeze_active and price < bb_l