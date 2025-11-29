```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Moon Dev Data Preparation Ritual 🌌
def prepare_data(path):
    data = pd.read_csv(path)
    # Cleanse column names
    data.columns = data.columns.str.strip().str.lower()
    # Banish unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Align with cosmic conventions
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# Cosmic Vortex Strategy Implementation 🌪️
class VortexCycler(Strategy):
    vi_period = 14
    dpo_period = 20
    swing_window = 20
    risk_pct = 0.01
    dpo_cycle_window = 5

    def init(self):
        # Calculate Vortex Energy Fields
        self.vi_plus = self.I(self._calculate_vi_plus, name='VI+')
        self.vi_minus = self.I(self._calculate_vi_minus, name='VI-')
        
        # Initiate DPO Cycle Scanner 🌗
        self.dpo = self.I(self._calculate_dpo, name='DPO')
        self.dpo_peaks = self.I(talib.MAX, self.dpo, self.dpo_cycle_window, name='DPO_PEAK')
        self.dpo_troughs = self.I(talib.MIN, self.dpo, self.dpo_cycle_window, name='DPO_TROUGH')

    def _calculate_vi_plus(self):
        vi_p, _ = talib.VORTEX(self.data.High, self.data.Low, self.data.Close, self.vi_period)
        return vi_p

    def _calculate_vi_minus(self):
        _, vi_m = talib.VORTEX(self.data.High, self.data.Low, self.data.Close, self.vi_period)
        return vi_m

    def _calculate_dpo(self):
        return ta.dpo(self.data.Close, length=self.dpo_period)

    def next(self):
        # Wait for cosmic alignment of indicators
        if len(self.data) < max(self.vi_period, self.dpo_period, self.swing_window) + 2:
            return

        # Current cosmic readings 🌠
        vi_p, vi_m = self.vi_plus[-1], self.vi_minus[-1]
        dpo = self.dpo[-1]
        prev_vi_p, prev_vi_m = self.vi_plus[-2], self.vi_minus[-2]

        # Calculate stellar positions
        swing_low = talib.MIN(self.data.Low, self.swing_window)[-1]
        swing_high = talib.MAX(self.data.High, self.swing_window)[-1]

        # Long Entry: Vortex Ascension + DPO Trough
        if (crossover(self.vi_plus, self.vi_minus) and dpo <= self.dpo_troughs[-1]:
            if not self.position.is_long:
                risk_amount = self.equity * self.risk_pct
                risk_per_share = self.data.Close[-1] - swing_low
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.buy(size=size, sl=swing_low, tag="🌌 CELESTIAL LONG")
                        print(f"🚀🌕 MOON LIFT-OFF! Long {size} @ {self.data.Close[-1]:.2f}")
        
        # Short Entry: Vortex Descent + DPO Peak
        if (crossover(self.vi_minus, self.vi_plus)) and dpo >= self.dpo_peaks[-1]:
            if not self.position.is_short:
                risk_amount = self.equity * self.risk_pct
                risk_per_share = swing_high - self.data.Close[-1]
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.sell(size=size, sl=swing_high, tag="🌑 VOID SHORT")
                        print(f"🌑🌪️ ENTERING THE VOID! Short {size} @ {self.data.Close[-