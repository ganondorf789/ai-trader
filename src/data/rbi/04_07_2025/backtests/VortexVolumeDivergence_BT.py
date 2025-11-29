```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VortexVolumeDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators with proper self.I() wrapping
        # Vortex Indicator (VI+ and VI-)
        vi = ta.vortex(self.data.High, self.data.Low, self.data.Close, length=14)
        self.vi_plus = self.I(lambda: vi.iloc[:,0], name='VI+')
        self.vi_minus = self.I(lambda: vi.iloc[:,1], name='VI-')
        
        # ADX for trend strength filtering
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Volume EMA for declining confirmation
        self.volume_ema = self.I(talib.EMA, self.data.Volume, 5)
        
        # ATR for risk calculation
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def next(self):
        # Wait for sufficient data
        if len(self.data) < 15:
            return
        
        # Current indicator values
        vi_plus = self.vi_plus[-1]
        vi_minus = self.vi_minus[-1]
        adx_value = self.adx[-1]
        atr_value = self.atr[-1]
        volume = self.data.Volume[-1]
        volume_ema = self.volume_ema[-1]
        
        # Moon Dev status monitoring 🌙
        print(f"🌙 Vortex Check | VI+={vi_plus:.2f}, VI-={vi_minus:.2f} | ADX={adx_value:.2f} | Vol/EMA={volume:.0f}/{volume_ema:.0f}")
        
        # Entry/Exit conditions
        long_entry = crossover(self.vi_plus, self.vi_minus) and (volume < volume_ema) and (adx_value < 20)
        short_entry = crossover(self.vi_minus, self.vi_plus) and (volume < volume_ema) and (adx_value < 20)
        
        long_exit = crossover(self.vi_minus, self.vi_plus) or (adx_value > 25)
        short_exit = crossover(self.vi_plus, self.vi_minus) or (adx_value > 25)
        
        # Risk management calculations
        if not self.position:
            if long_entry:
                # Calculate position size based on ATR stop
                risk_amount = self.equity * self.risk_percent
                stop_loss = self.data.Close[-1] - 2 * atr_value
                risk_per_share = self.data.Close[-1] - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        print(f"🚀🌕 BLAST OFF LONG! | Size={position_size} | Entry={self.data.Close[-1]:.2f}")
                        self.buy(size=position_size, sl=stop_loss)
                        
            elif short_entry:
                risk_amount = self.equity * self.risk_percent
                stop_loss = self.data.Close[-1] + 2 * atr_value
                risk_per_share = stop_loss - self.data.Close[-1]
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        print(f"🌑🌘 BLACK HOLE SHORT! | Size={position_size} | Entry={self.data.Close