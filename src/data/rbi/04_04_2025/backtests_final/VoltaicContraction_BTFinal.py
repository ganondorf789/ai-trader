I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaicContraction(Strategy):
    bb_period = 20
    bb_dev = 2
    bbw_threshold = 0.5
    atr_period = 14
    risk_pct = 0.01
    liquidation_drop = -0.02
    volume_multiplier = 1.5

    def init(self):
        # Bollinger Band Width
        def calculate_bbw(close, period, dev):
            upper, middle, lower = talib.BBANDS(close, timeperiod=period, 
                                                nbdevup=dev, nbdevdn=dev)
            return (upper - lower) / middle
        self.bbw = self.I(calculate_bbw, self.data.Close, self.bb_period, self.bb_dev)
        
        # Volume SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.bb_period)
        
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         self.atr_period)
        
        # VWMA
        self.vwma = self.I(ta.vwma, self.data.Close, self.data.Volume, length=self.bb_period)

        print("🌙 MOON DEV INIT COMPLETE 🌙 | Indicators loaded successfully!")

    def next(self):
        if len(self.data) < self.bb_period + 2:
            return

        current_idx = len(self.data) - 1
        prev_idx = current_idx - 1

        # Liquidation cascade check 🌊
        price_drop = (self.data.Close[prev_idx] - self.data.Close[prev_idx-1])/self.data.Close[prev_idx-1]
        liquidation_price = price_drop <= self.liquidation_drop
        liquidation_vol = self.data.Volume[prev_idx] > self.volume_multiplier * self.volume_sma[prev_idx]
        liquidation_cascade = liquidation_price and liquidation_vol

        # BBW threshold cross check (replaces crossover) 📉
        bbw_cross = (self.bbw[-2] >= self.bbw_threshold and 
                    self.bbw[-1] < self.bbw_threshold)

        # Reversal candle conditions 🕯️
        reversal_candle = (self.data.Close[-1] > self.data.Open[-1])
        volume_boost = self.data.Volume[-1] > self.volume_sma[-1]

        # Basis divergence (assuming 'basis' column exists) 📈📉
        basis_divergence = (self.data.Close[-1] > self.data.Close[-2] and 
                           self.data['basis'][-1] < self.data['basis'][-2])

        # Entry logic 🚀
        if not self.position and all([
            liquidation_cascade,
            bbw_cross,
            reversal_candle,
            volume_boost,
            basis_divergence
        ]):
            atr_value = self.atr[-1]
            if atr_value == 0:
                print("🌙 MOON DEV WARNING 🌙 | Zero ATR value detected, skipping trade")
                return

            risk_amount = self.equity * self.risk_pct
            stop_loss = self.data.Close[-1] - 0.5 * atr_value
            risk_per_share = self.data.Close[-1] - stop_loss
            
            position_size = int(round(risk_amount / risk_per_share))
            if position_size > 0:
                self.buy(size=position_size, 
                        sl=stop_loss, 
                        tp=self.data.Close[-1] + 1.5 * atr_value)
                print(f"🌙 MOON DEV ENTRY SIGNAL 🌙\n"
                      f"✨ Size: {position_size} units\n"
                      f"✨ Entry: {