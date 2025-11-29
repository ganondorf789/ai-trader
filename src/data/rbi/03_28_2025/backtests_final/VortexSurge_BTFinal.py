I'll analyze the code and ensure it meets all backtesting requirements while maintaining the original strategy logic. Here's the debugged version with Moon Dev enhancements:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as pta

# Data Preparation - Lunar Data Loading 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
print("🌑 Loading celestial market data from:", data_path)
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns - Moon Dust Removal ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
print("🌒 Data purification complete! Ready for lunar analysis")

class VortexSurge(Strategy):
    def init(self):
        # Calculate indicators - Moon Phase Calculations 🌗
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        
        # Vortex Indicator (VI+, VI-) - Cosmic Energy Flow 🌌
        vi_plus, vi_minus = pta.vortex(high=high, low=low, close=close, length=14)
        self.vi_plus = self.I(lambda: vi_plus, name='VI_PLUS')
        self.vi_minus = self.I(lambda: vi_minus, name='VI_MINUS')
        
        # Volume SMA - Lunar Tidal Patterns 🌊
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # ATR for exit conditions - Cosmic Volatility ☄️
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)
        
        print("🌙 Moon Dev Indicators Ready! Vortex Surge Activated 🚀")

    def next(self):
        # Skip early bars where indicators are NaN - Waiting for Full Moon 🌕
        if len(self.data) < 20:
            return

        # Get current values - Reading Celestial Signals 🌠
        vi_plus = self.vi_plus[-1]
        vi_plus_prev = self.vi_plus[-2]
        vi_minus = self.vi_minus[-1]
        vi_minus_prev = self.vi_minus[-2]
        
        volume = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        atr = self.atr[-1]

        # Entry condition - Moon Launch Sequence 🚀
        if not self.position:
            # Manual crossover detection (no backtesting.lib!)
            bullish_crossover = (vi_plus_prev <= vi_minus_prev) and (vi_plus > vi_minus)
            
            if bullish_crossover and (volume > vol_sma):
                # Risk management - Cosmic Capital Preservation 🌍
                risk_pct = 0.01  # 1% risk per trade
                equity = self.equity
                risk_amount = equity * risk_pct
                
                if atr == 0:  # Avoid black hole division
                    return
                
                position_size = risk_amount / atr
                position_size = int(round(position_size))  # Ensure whole units
                
                if position_size > 0:
                    tp_price = self.data.Close[-1] + 1.5 * atr  # Fixed TP as price level
                    self.buy(
                        size=position_size,
                        sl=None,  # Using VI reversal for exit
                        tp=tp_price,
                        tag={
                            'entry_price': self.data.Close[-1],
                            'atr': atr,
                            'tp_price': tp_price
                        }
                    )
                    print(f"🚀 MOON DEV ENTRY! Size: {position_size} units @ {self.data.Close[-1]:.2f} 🌙 TP: {tp_price:.2f}")

        # Exit conditions - Re-entry Protocol 🛸
        elif self.position:
            # Manual crossover detection (no backtesting.lib!)
            bearish_crossover = (vi_plus_prev >= vi_minus_prev) and (vi_plus < vi_minus)
            
            if bearish_crossover:
                self.position.close()
                print(f"🌘 MOON DEV EXIT! VI Reversal Signal @ {self.data.Close[-1]:.2f}")

# Run back