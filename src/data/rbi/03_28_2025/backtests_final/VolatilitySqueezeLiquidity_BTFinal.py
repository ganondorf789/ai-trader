I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's Volatility Squeeze Liquidity Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# ---------------------------
# DATA PREPARATION
# ---------------------------
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ---------------------------
# STRATEGY IMPLEMENTATION
# ---------------------------
class VolatilitySqueezeLiquidity(Strategy):
    bb_period = 20
    bb_dev = 2
    swing_window = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Bollinger Bands Calculation
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close)
        
        # 🌓 Middle Band for Width Calculation
        def bb_middle(close):
            _, middle, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return middle
        self.bb_middle = self.I(bb_middle, self.data.Close)
        
        # 🌗 BB Width Percentile (100-period)
        def bb_width_pct(close):
            upper, middle, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            width = (upper - lower) / middle
            return ta.percentile_rank(width, 100)  # Fixed typo in function name
        self.bb_width_pct = self.I(bb_width_pct, self.data.Close)
        
        # 🌕 Volume Analysis
        self.volume_median = self.I(ta.median, self.data.Volume, 50)
        
        # 🌑 Trend Filter (200 EMA)
        self.ema200 = self.I(talib.EMA, self.data.Close, 200)
        
        # 🌒 Liquidity Gaps (Swing Highs/Lows)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        print("🌙✨ Moon Dev System Activated! Ready to hunt volatility squeezes! 🚀")

    def next(self):
        # Moon-themed debug prints
        if len(self.data) % 100 == 0:
            print(f"🌕 Moon Pulse: Close={self.data.Close[-1]:.2f} | BB%={self.bb_width_pct[-1]:.1f} | Vol={self.data.Volume[-1]:.0f}")
        
        # Risk Management Guardrails
        if self.equity < 0.95 * 1_000_000:
            print("💥 CRITICAL MOON ALERT: 5% Daily Loss Limit Breached!")
            self.position.close()
            return  # Fixed missing return statement
        
        if not self.position:
            # 🌙 Long Entry Conditions
            if (self.bb_width_pct[-1] < 20 and
                self.data.Volume[-1] > 1.25 * self.volume_median[-1] and
                self.data.Close[-1] > self.bb_upper[-1] and
                self.data.Close[-1] > self.ema200[-1]):
                
                risk_amount = self.equity * self.risk_pct
                risk_per_share = self.data.Close[-1] - self.bb_lower[-1]
                
                if risk_per_share >