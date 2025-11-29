Here's the debugged and completed version of your VwapDivergenceSqueeze strategy with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Load and preprocess data
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VwapDivergenceSqueeze(Strategy):
    vwap_period = 288  # 3 days in 15m intervals (3*24*4*15/15)
    roc_period = 288   # 3-day ROC
    bb_period = 20
    bb_std = 2
    bandwidth_median_period = 1920  # 20 days (20*96)
    atr_period = 14
    risk_pct = 0.01
    max_positions = 5

    def init(self):
        # Compute VWAP with pandas_ta
        self.vwap = self.I(ta.vwap,
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume,
            length=self.vwap_period,
            name='VWAP'
        )

        # VWAP ROC using TA-Lib
        self.vwap_roc = self.I(talib.ROC,
            self.vwap,
            timeperiod=self.roc_period,
            name='VWAP_ROC'
        )

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS,
            self.data.Close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # Calculate Bandwidth
        def calc_bandwidth(data):
            return ((data.BB_Upper - data.BB_Lower) / data.BB_Middle) * 100
        
        self.bandwidth = self.I(calc_bandwidth, name='Bandwidth')

        # Bandwidth Median
        self.bw_median = self.I(talib.MEDIAN,
            self.bandwidth,
            timeperiod=self.bandwidth_median_period,
            name='BW_Median'
        )

        # ATR for volatility stops
        self.atr = self.I(talib.ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            timeperiod=self.atr_period,
            name='ATR'
        )

        print("🌙 Lunar Indicators Activated! Ready for Quantum Backtest 🚀")

    def next(self):
        # Moon Dev Debug Prints
        if len(self.data) % 1000 == 0:
            print(f"✨ Processing Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} 🌙")

        # Exit conditions
        for trade in self.trades:
            atr = self.atr[-1]
            if trade.is_long:
                sl = trade.entry_price - 2 * atr
                if self.data.Low[-1] < sl:
                    trade.close()
                    print(f"🌙 CLOSE LONG {self.data.index[-1]} | SL: {sl:.2f} 🛑")
            else:
                sl = trade.entry_price + 2 * atr
                if self.data.High[-1] > sl:
                    trade.close()
                    print(f"🌙 CLOSE SHORT {self.data.index[-1]} | SL: {sl:.2f} 🛑")

        # Don't open new trades if max positions reached
        if len(self.trades) >= self.max_positions:
            return

        # Need at least 2 periods for divergence check
        if len(self.data) < 2:
            return

        # Detect divergences
        current_low = self.data.Low[-1]
        prev_low = self.data.Low