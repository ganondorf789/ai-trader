```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

class VortexContraction(Strategy):
    risk_per_trade = 0.01
    max_open_positions = 5
    atr_period = 14
    bb_period = 20
    bb_dev = 2
    vi_period = 14
    adx_period = 14
    swing_period = 20

    def init(self):
        # Calculate indicators using TA-Lib and pandas_ta
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        
        # Vortex Indicator
        vi = ta.vortex(high=high, low=low, close=close, length=self.vi_period)
        self.vi_plus = self.I(lambda: vi[f'VORTIC_{self.vi_period}'], name='VI+')
        self.vi_minus = self.I(lambda: vi[f'VORTICm_{self.vi_period}'], name='VI-')

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = talib.BBANDS(
            close, timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev
        )
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_width_ma = talib.SMA(self.bb_width, timeperiod=20)
        
        # ADX and DI
        self.adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        self.plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        self.minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        
        # ATR for risk management
        self.atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

    def next(self):
        # Moon Dev position tracking 🌙
        if len(self.positions) >= self.max_open_positions:
            print("🌙 MAX POSITIONS REACHED: 5 open trades active 🛑")
            return

        # Long entry logic 🚀
        if not self.position:
            if self._long_conditions():
                swing_low = talib.MIN(self.data.Low, timeperiod=self.swing_period)[-1]
                entry_price = self.data.Close[-1]
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = entry_price - swing_low
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(
                            size=position_size, 
                            sl=swing_low, 
                            tp=self.bb_upper[-1]
                        )
                        print(f"🌙 MOON DEV ENTRY 🚀 | Price: {entry_price:.2f} | Size: {position_size} | SL: {swing_low:.2f} | TP: {self.bb_upper[-1]:.2f} ✨")

        # Exit management 🔄
        for trade in self.trades:
            if trade.is_long and not trade.is_closed:
                self._manage_exits(trade)

    def _long_conditions(self):
        # Core strategy logic alignment ✨
        vi_ok = self.vi_plus[-1] > self.vi_minus[-1]
        bb_ok = self.bb_width[-1] < self.bb_width_ma[-1]
        adx_rising = (self.adx[-1] > self.adx[-2]) and (self.adx[-2] > self.adx[-3])
        adx_strong = self.adx[-1] > 25
        price_ok = self.data.Close[-1] > self.bb_middle[-1]
        
        return all([vi_ok, bb_ok, adx_rising, adx_strong, price_ok])

    def _manage_exits(self, trade):
        # Partial profit at BB upper 🎯
        if self.data.High[-1] >= self.bb_upper[-1]:
            trade.close(portion=0.5)
            print(f"🌙 PARTIAL PROFIT 🎯 | Closed 50%