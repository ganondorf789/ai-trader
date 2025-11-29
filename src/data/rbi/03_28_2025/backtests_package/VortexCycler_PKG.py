I'll fix the code by removing all backtesting.lib imports and replacing crossover functions with proper array comparisons. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

class VortexCycler(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    trailing_atr_multiplier = 2
    max_trade_duration = 20
    atr_period = 14
    dpo_period = 20
    vi_period = 14
    volume_sma_period = 50

    def init(self):
        # Primary indicators with self.I()
        self.dpo = self.I(talib.DPO, self.data.Close, self.dpo_period, name='DPO')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing Low')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_sma_period, name='Volume SMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')

        # Vortex Indicator calculations
        high = self.data.High
        low = self.data.Low
        
        # Calculate VM+ and VM- using shifted values
        vm_plus = high - low.shift(1)
        vm_minus = high.shift(1) - low
        
        # Sum components
        self.sum_vm_plus = self.I(talib.SUM, vm_plus, self.vi_period, name='SUM VM+')
        self.sum_vm_minus = self.I(talib.SUM, vm_minus, self.vi_period, name='SUM VM-')
        self.sum_tr = self.I(talib.SUM, self.I(talib.TRANGE, high, low, self.data.Close), self.vi_period, name='SUM TR')

    def next(self):
        # Moon Dev debug prints
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {len(self.data)} - Equity: ${self.equity:,.2f}")

        if self.position:
            # Exit conditions check
            current_close = self.data.Close[-1]
            exit_condition = False

            # Calculate current VI values
            sum_tr = self.sum_tr[-1] or 1  # Avoid division by zero
            vi_plus = self.sum_vm_plus[-1]/sum_tr
            vi_minus = self.sum_vm_minus[-1]/sum_tr

            # Condition 1: VI crossover exit (replaced backtesting.lib.crossover)
            if vi_minus[-2] < vi_plus[-2] and vi_minus[-1] > vi_plus[-1]:  # Bearish crossover
                print(f"🌪️ VI reversal detected! Closing position at {current_close}")
                self.position.close()
                exit_condition = True

            # Condition 2: Volume filter
            if not exit_condition and self.data.Volume[-1] < self.volume_sma[-1]:
                print(f"📉 Volume contraction! Closing position at {current_close}")
                self.position.close()
                exit_condition = True

            # Trailing stop logic
            if not exit_condition:
                trade = self.position
                if current_close >= trade.entry_price + 2*self.atr[-1]:
                    new_sl = trade.entry_price
                    if trade.sl != new_sl:
                        print(f"🔐 Moving stop loss to breakeven @ ${new_sl}")
                        trade.sl = new_sl

                # Max duration check
                if trade.duration >= self.max_trade_duration:
                    print(f"⏳ Max trade duration reached! Closing at {current_close}")
                    self.position.close()

        else:
            # Entry conditions
            dpo_condition = self.dpo[-1] > 0
            vi_plus = self.sum_vm_plus[-1]/(self.sum_tr[-1] or 1)
            vi_minus = self.sum_vm_minus[-1]/(self.sum_tr[-1] or 1)
            
            # Entry signal (replaced backtesting.lib.crossover)
            if dpo_condition and (vi_plus[-2] < vi_minus[-2] and vi_plus[-1] > vi_minus[-1]):  # Bullish crossover
                # Risk management calculations
                sl_price = self.swing_low[-1]
                risk_per_share =