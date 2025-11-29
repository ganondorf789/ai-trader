I'll fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VolatilityFadeVWAP(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_daily_trades = 3
    atr_multiplier = 1.5
    exit_bars = 5

    def init(self):
        # Clean data handled in preprocessing
        self._setup_indicators()
        
    def _setup_indicators(self):
        # Bollinger Bands components
        def bb_upper(close): return talib.BBANDS(close, 20, 2, 2, 0)[0]
        def bb_middle(close): return talib.BBANDS(close, 20, 2, 2, 0)[1]
        def bb_lower(close): return talib.BBANDS(close, 20, 2, 2, 0)[2]
        
        self.upper = self.I(bb_upper, self.data.Close)
        self.middle = self.I(bb_middle, self.data.Close)
        self.lower = self.I(bb_lower, self.data.Close)
        
        # Bollinger Band width calculations
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.upper, self.lower, self.middle)
        self.bb_pct = self.I(lambda x: x.rolling(20).quantile(0.9), self.bb_width)

        # Volume-weighted indicators
        self.vwma = self.I(ta.vwma, self.data.Close, self.data.Volume, 20)
        self.vwma_slope = self.I(lambda x: x.pct_change()*100, self.vwma)

        # Volatility and momentum
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)

        # Volume analysis
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.std_dev = self.I(talib.STDDEV, self.data.Close, 20)

    def next(self):
        current_bar = len(self.data)-1
        current_date = self.data.index[-1].date()

        # Entry logic
        if not self.position:
            trades_today = sum(1 for t in self.trades 
                             if self.data.index[t.entry_bar].date() == current_date)
            
            if trades_today < self.max_daily_trades:
                self._check_short_entry(current_date)

        # Exit logic
        for trade in self.trades:
            if not trade.is_short:
                continue
                
            if self._check_slope_exit(trade):
                continue
                
            self._check_time_exit(trade, current_bar)

    def _check_short_entry(self, current_date):
        # Indicator values
        close = self.data.Close[-1]
        atr = self.atr[-1]
        volatility_filter = (atr/close) <= 0.03
        
        entry_conditions = (
            self.bb_width[-1] > self.bb_pct[-1] and
            abs(self.vwma_slope[-1]) < 0.1 and
            abs(close - self.vwma[-1]) <= 0.5*self.std_dev[-1] and
            self.data.Volume[-1] > self.vol_sma[-1] and
            self.rsi[-1] > 60 and
            volatility_filter
        )

        if entry_conditions:
            risk_amount = self.equity * self.risk_per_trade
            position_size = int(round(risk_amount / atr))
            
            if position_size > 0:
                self.sell(size=position_size,
                        sl=close + atr,
                        tp=close - self.atr_multiplier*atr,
                        tag="🌙 SHORT ENTRY")
                print(f"🌙 MOON DEV ROCKET LAUNCH 🌙 | Entry: {close:.2f} | Size: {position_size}")

    def _check_slope_exit(self, trade):
        if abs(self.vwma_slope[-1]) > 0.3:
            trade.close()
            print(f"✨ MOON DEV SLOPE BAILOUT ✨ | Exit: {self.data.Close[-1]:.2f}")