Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper implementations:

```python
# moon_dev_volatility_cascade.py
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

class VolatilityCascade(Strategy):
    # Strategy parameters
    oi_lookback = 16       # 4 hours for 15m data
    hvol_period = 20
    hvol_std = 2
    iv_lookback = 252
    max_trade_duration = 192  # 48 hours
    
    def init(self):
        # Clean and prepare data columns 🌙
        self.open_interest = self.data.df['openinterest']
        self.funding_rate = self.data.df['fundingrate']
        self.vix_front = self.data.df['vix_front']
        self.vix_second = self.data.df['vix_second']
        self.iv = self.data.df['iv']

        # Calculate OI change rate using TA-Lib
        self.oi_pct_change = self.I(
            lambda x: x.pct_change(self.oi_lookback), 
            self.open_interest, 
            name='OI Change'
        )

        # Calculate HVOL bands with proper TA-Lib wrappers
        self.hvol_upper = self.I(
            lambda close: ta.bbands(close, length=self.hvol_period, std=self.hvol_std)['BBU_{}_{}'.format(self.hvol_period, self.hvol_std)],
            self.data.Close,
            name='HVOL Upper'
        )
        
        self.hvol_lower = self.I(
            lambda close: ta.bbands(close, length=self.hvol_period, std=self.hvol_std)['BBL_{}_{}'.format(self.hvol_period, self.hvol_std)],
            self.data.Close,
            name='HVOL Lower'
        )

        # Funding rate polarity tracker
        self.funding_polarity = self.I(
            lambda x: np.where(x > 0, 1, -1),
            self.funding_rate,
            name='Funding Polarity'
        )

        # VIX term structure calculation
        self.vix_term = self.I(
            lambda front, second: front - second,
            self.vix_front,
            self.vix_second,
            name='VIX Term'
        )

        # IV Percentile calculation
        self.iv_rank = self.I(
            lambda iv: iv.rolling(self.iv_lookback).rank(pct=True),
            self.iv,
            name='IV Rank'
        )

        # Track entry timing for time-based exits
        self.entry_bar = 0

    def next(self):
        current_bar = len(self.data) - 1
        
        # Moon-themed debug prints ✨
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Dev Pulse: Bar {current_bar} | Price: {self.data.Close[-1]:.2f} | Equity: {self.equity:,.2f} 🚀")

        # Exit conditions
        if self.position:
            term_structure = self.vix_term[-1]
            prev_term = self.vix_term[-2] if len(self.vix_term) > 2 else 0
            
            # VIX term structure reversal
            term_reversal = (term_structure * prev_term) < 0
            
            # HVOL reconvergence
            hvol_width = self.hvol_upper[-1] - self.hvol_lower[-1]
            hvol_contraction = hvol_width < hvol_width[-50:].mean() * 0.7
            
            # Time-based exit
            time_exit = current_bar - self.entry_bar >= self.max_trade_duration

            if term_reversal or hvol_contraction or time_exit:
                self.position.close()
                print(f"🌕 Moon Dev Exit: Term Reversal={term_reversal} | HVOL Contraction={hvol_contraction} | Time Exit={time_exit} ✂️")

        # Entry conditions
        else:
            # Liquidation cascade detection
            oi_crash = self.oi_pct_change[-1] <= -0.2
            
            # Volatility filter
            iv_valid = self.iv_rank[-1] >= 0.3
            
            # Funding rate flip detection
            funding_flip = (
                len(self.funding_polarity) > 1 and 
                self.funding_polarity[-1] != self.funding_polarity[-2]
            )

            # Price breakout detection (replaced crossover with manual