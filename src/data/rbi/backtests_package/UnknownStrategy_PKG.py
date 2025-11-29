import pandas as pd
import numpy as np
from backtrex import Backtrex, OrderType

class BacktestingStrategy:
    def __init__(self, data, name="Test Strategy"):
        self.data = data
        self.name = name
        self positioned = False
        self.trades = []
        
    def get_signals(self):
        df = self.data.copy()
        df['Signal'] = 0.0
        
        # Calculate indicators
        df['EMA20'] = df['Close'].rolling(window=20).mean()
        df['STDEV20'] = df['Close'].rolling(window=20).std()
        
        # Generate signals based on conditions
        df.loc[df['EMA20'] > df['Close'], 'Signal'] += 1  # Buy signal
        df.loc[df['EMA20'] < df['Close'], 'Signal'] -= 1  # Sell signal
        
        return df[['Close', 'STDEV20', ' Signal']]
    
    def execute_trades(self, backtrex):
        positions = []
        exit_orders = []
        
        for i in range(len(self.data)):
            close_price = self.data['Close'].iloc[i]
            stop_loss = max(0.98 * close_price, (close_price - 2 * self.data['STDEV20'].iloc[i]))
            take_profit = close_price + 1.5 * self.data['STDEV20'].iloc[i]
            
            if self.get_signals().Signal.iloc[i] == 1:
                lot_size = backtrex.position_size(close_price, capital=50000) / close_price
                order = backtrex.order('Buy', OrderType.Market, lot_size)
                positions.append({'Instrument': None, 'PositionSize': lot_size, 'EntryPrice': close_price,
                                 'ExitPrice': 0.0, 'ExitType': 'Market'})
                
            elif self.get_signals().Signal.iloc[i] == -1:
                for pos in backtrex.positions():
                    if pos['PositionSize'] != 0 and abs(pos['PositionSize']) > 2 * lot_size:
                        exit_price = max(abs(0.98 * close_price), (close_price + 2 * self.data['STDEV20'].iloc[i]))
                        order = backtrex.order('Sell', OrderType.Market, pos['PositionSize'])
                        exit_orders.append({'Instrument': None, 'ExitPrice': exit_price, 'ExitType': 'Market'})
                        break
        
        return positions, exit_orders

class BacktrexStrategy(BacktestingStrategy):
    def __init__(self, data, name="Test Strategy", period=30,
                 fast_ma_period=15, slow_ma_period=30,
                 capital=50000, risk_buffer=2.0,
                 max_positions=2):
        super().__init__(data, name)
        self.period = period
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.capital = capital
        self.risk_buffer = risk_buffer
        self.max_positions = max_positions
        
    def run_backtest(self):
        backtrex = Backtrex(self.data, 'Backtest')
        
        # Pre-calculate indicators
        df = self.get_signals()
        
        positions, exit_orders = self.execute_trades(backtrex)
        if positions or exit_orders:
            for pos in positions:
                lot_size = self.capital * 2 / (self.data['Close'].iloc[0] ** 2)
                backtrex.position(pos['PositionSize'], pos['EntryPrice'], 'long')
            
            for ex in exit_orders:
                total exiting = sum(abs(ex['ExitSize']) for ex in exit_orders)
                remaining_capital = self.capital - ((total exiting * abs(ex['ExitPrice'] - ex['ExitPrice'])) / 2
                                                   if ex['ExitType'] == 'Market' else 0)
                if remaining_capital >= 5000:
                    lot_size = min(remaining_capital // (self.data['Close'].iloc[0] ** 2), 
                                   max(int(backtrex.position_size(ex['ExitPrice'], capital=remaining_capital) / ex['ExitPrice']),1))
                    backtrex.position(ex['ExitSize'], ex['ExitPrice'], 'short')
        
        backtrex.execute_all_orders()
        return backtrex