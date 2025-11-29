```python
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime']
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}).set_index('datetime')

class VortexVolumeBreakout(Strategy):
    vi_period = 14
    volume_ma_period = 1920  # 20 days in 15m intervals
    atr_period = 14
    atr_multiplier = 1.5
    sar_acceleration = 0.02
    sar_maximum = 0.2
    risk_percent = 0.01
    stop_loss_pct = 0.03

    def init(self):
        # Calculate indicators using TA-Lib and pandas_TA
        vi = self.data.df.ta.vortex(length=self.vi_period)
        self.vi_plus = self.I(lambda: vi[f'VORTEXP_{self.vi_period}'], name='VI+ 🌙')
        self.vi_minus = self.I(lambda: vi[f'VORTEXM_{self.vi_period}'], name='VI- 🌑')
        
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period, name='Vol MA 🌊')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR 🌪️')
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, 
                         acceleration=self.sar_acceleration, maximum=self.sar_maximum, name='SAR 🎯')
        self.avg_atr = self.I(talib.SMA, self.atr, 50, name='ATR MA ⚖️')
        
        self.equity_history = []
        self.stop_trading = False

    def next(self):
        # Moon Dev system heartbeat
        if len(self.data) % 100 == 0:
            print(f"🌙 Lunar Pulse detected on bar {len(self.data)}. Equity: ${self.equity():,.2f} ✨")
        
        # Check weekly drawdown
        self.equity_history.append(self.equity())
        if len(self.equity_history) > 672:  # 1 week of 15m data
            self.equity_history.pop(0)
        
        if len(self.equity_history) >= 672:
            peak = max(self.equity_history)
            trough = min(self.equity_history)
            drawdown = (peak - trough)/peak
            if drawdown >= 0.05:
                print(f"🌪️ CELESTIAL ALERT! 5% drawdown detected. Shutting down thrusters!")
                self.stop_trading = True
                return
        
        if not self.position and not self.stop_trading:
            # Entry conditions
            vi_bullish = crossover(self.vi_plus, self.vi_minus)
            volume_boost = self.data.Volume[-1] > self.volume_ma[-1]
            volatility_spike = (self.data.High[-1] - self.data.Low[-1]) > (self.atr[-1] * self.atr_multiplier)
            
            if vi_bullish and volume_boost and volatility_spike:
                # Calculate position size with risk management
                risk_amount = self.equity() * self.risk_percent
                entry_price = self.data.Close[-1]
                stop_price = entry_price * (1 - self.stop_loss_pct)
                risk_per_share = entry_price - stop_price
                
                if risk_per_share <= 0:
                    print("⚠️ Abort launch! Negative risk detected")
                    return
                
                position_size = risk_amount / risk_per_share
                
                # Volatility adjustment
                if self.atr[-1] > self.avg_atr[-1]:
                    position_size *= 0.5
                    print("🌊 Solar Flare Warning! Reducing payload by 50%")
                
                position_size = int(round(position