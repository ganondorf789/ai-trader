Here's the complete fixed code with Moon Dev themed debug prints and all necessary technical fixes:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION ✨

# 1. NECESSARY IMPORTS
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 2. DATA PREPARATION
def prepare_data(path):
    # Load and clean data
    data = pd.read_csv(
        path,
        parse_dates=['datetime'],
        index_col='datetime'
    )
    
    # Clean column names 🌙
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns 
                            if 'unnamed' in col.lower()])
    
    # Proper column mapping 🚀
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data = data.rename(columns=column_map)
    
    print("🌙✨ MOON DEV DATA PREPARATION COMPLETE! 🚀")
    return data

# 3. STRATEGY IMPLEMENTATION
class LiquidationBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # BOLLINGER BANDS SETUP ✨
        self.bb_upper = self.I(
            lambda close: talib.BBANDS(close, 20, 2, 2)[0],
            self.data.Close,
            name='BB_UPPER'
        )
        self.bb_lower = self.I(
            lambda close: talib.BBANDS(close, 20, 2, 2)[2],
            self.data.Close,
            name='BB_LOWER'
        )
        
        # VOLATILITY CONTRACTION DETECTION 🌙
        self.bb_width = self.I(
            lambda u, l: u - l,
            self.bb_upper, self.bb_lower,
            name='BB_WIDTH'
        )
        self.bb_width_sma = self.I(
            talib.SMA, self.bb_width, 50,
            name='BB_WIDTH_SMA'
        )
        
        # LIQUIDATION SURGE INDICATOR 🚀
        self.liq_ma = self.I(
            talib.SMA, self.data['liquidations'], 20,
            name='LIQ_MA'
        )
        
        # BASIS DIVERGENCE FILTER ✨
        self.basis_ma = self.I(
            talib.SMA, self.data['basis'], 20,
            name='BASIS_MA'
        )
        
        # VOLATILITY FOR STOP CALCULATION 🌙
        self.atr = self.I(
            talib.ATR, self.data.High, self.data.Low, self.data.Close, 14,
            name='ATR'
        )
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED! 🚀")

    def next(self):
        if self.position:
            return  # Hold existing position 🌙
            
        # CURRENT MARKET CONDITIONS ✨
        price = self.data.Close[-1]
        basis = self.data['basis'][-1]
        liq = self.data['liquidations'][-1]
        
        # INDICATOR VALUES 🚀
        bb_narrowing = self.bb_width[-1] < self.bb_width_sma[-1]
        liq_surge = liq > 1.5 * self.liq_ma[-1]
        basis_divergence = basis > self.basis_ma[-1] if basis > 0 \
            else basis < self.basis_ma[-1]
        
        # LONG ENTRY CONDITIONS 🌙
        if (price > self.bb_upper[-1] and
            bb_narrowing and
            liq_surge and
            basis > 0 and
            basis_divergence):
            
            # RISK MANAGEMENT CALCULATIONS ✨
            sl = price - self.atr[-1]
            risk_per_share = price - sl
            position_size = int(round((self.equity * self.risk_pct) / risk_per_share))
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl, tp=price + 2*self.atr[-1])
                print(f"🌕🚀 MOON DEV LONG! Entry: {price:.2f} | "
                      f"SL: {sl:.2f} | TP: {price + 2*self.atr[-1]:.2f} "
                      f"Size: