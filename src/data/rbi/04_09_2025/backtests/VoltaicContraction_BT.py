```python
# 🌙 Moon Dev's Voltaic Contraction Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as ta
from pandas_ta.statistics import percentile

# ======== DATA PREPARATION ========
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data 🌙
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping ✨
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data.index = pd.to_datetime(data['datetime'])
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# ======== VOLTAIC CONTRACTION STRATEGY ========
class VoltaicContraction(Strategy):
    risk_pct = 0.02  # 🌙 2% risk per trade
    tp_ratio = 2     # 🚀 2:1 reward:risk
    max_bars = 5      # ⏳ Time-based exit
    
    def init(self):
        # ======== INDICATORS ========
        # Bollinger Band Width 🌈
        def bbw_calc(close):
            upper, middle, lower = talib.BBANDS(close, 20, 2, 2)
            return (upper - lower) / middle
        self.bbw = self.I(bbw_calc, self.data.Close)
        
        # Futures Basis Spread (assuming 'basis' column exists) ⚡
        self.basis = self.data['basis']  # Requires basis column in CSV
        self.basis_ma = self.I(talib.SMA, self.basis, 5)
        
        # Contraction Range 🔭
        self.cont_high = self.I(talib.MAX, self.data.High, 20)
        self.cont_low = self.I(talib.MIN, self.data.Low, 20)
        
        # Volume Surge 🌊
        self.vol_pct = self.I(percentile, self.data.Volume, 20, 90)
        
        self.entry_bar = 0  # Track entry timing

    def next(self):
        price = self.data.Close[-1]
        
        # ======== ENTRY LOGIC ========
        if not self.position:
            # Long Entry Conditions 🌙
            if (self.bbw[-1] < 0.5 and
                self.basis[-1] > self.basis_ma[-1] and
                price > self.cont_high[-1] and
                self.data.Volume[-1] > self.vol_pct[-1]):
                
                # Risk Management 💰
                sl = self.cont_low[-1]
                risk_amount = self.equity * self.risk_pct
                risk_per_unit = price - sl
                
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    tp = price + self.tp_ratio * (price - sl)
                    
                    self.buy(size=size, sl=sl, tp=tp)
                    self.entry_bar = len(self.data)
                    
                    print(f"🌙✨🚀 LONG ENTRY @ {price:.2f}")
                    print(f"Size: {size}, SL: {sl:.2f}, TP: {tp:.2f}")

        # ======== EXIT LOGIC ========
        else:
            # Time Exit ⏳
            if (len(self.data) - self.entry_bar) >= self.max_bars:
                self.position.close()
                print(f"🌙🕒 TIME EXIT @ {price:.2f}")
            
            # Volatility Expansion Exit 🌪️
            if self.bbw[-1] > 1.5:
                self.position.close()
                print(f"🌙📈 BBW EXPANSION EXIT @ {price:.2f}")

    def notify_trade(self, trade):
        if trade.is_closed:
            print(f"🌙💸 TRADE CLOSED: PnL ${trade.pnl:.2f}")

# ======== BACKTEST EXECUTION ========
bt = Backtest(data, VoltaicContraction, cash=1_000_000