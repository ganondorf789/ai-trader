"""
Bear Flag Pattern Strategy Backtest
Based on: https://www.tradingview.com/chart/BTCUSDT/qq6ih8sa-Bitcoin-Sell-this-upcoming-pump-New-Bear-Flag-Target-74k/

Strategy Logic:
- Detect Bear Flag pattern (sharp decline followed by upward consolidation)
- Short at upper resistance of the flag
- Target: Previous swing low (typically 15-25% below entry)
- Stop loss: Above flag high

Uses backtesting.py with pandas_ta for indicators
"""

import os
import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy


def detect_bear_flag(data, pole_lookback=20, flag_lookback=10, min_pole_drop=0.08):
    """
    Detect Bear Flag pattern

    Bear Flag = Sharp decline (pole) + Upward/sideways consolidation (flag)

    Parameters:
    - pole_lookback: Bars to look back for the pole (sharp decline)
    - flag_lookback: Bars for flag consolidation
    - min_pole_drop: Minimum percentage drop for valid pole (8%)
    """
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values
    n = len(close)

    bear_flag = np.zeros(n)
    flag_high = np.full(n, np.nan)
    flag_low = np.full(n, np.nan)
    target = np.full(n, np.nan)

    for i in range(pole_lookback + flag_lookback, n):
        # Find pole: sharp decline before the flag
        pole_start_idx = i - pole_lookback - flag_lookback
        pole_end_idx = i - flag_lookback

        pole_high = np.max(high[pole_start_idx:pole_end_idx])
        pole_low = np.min(low[pole_start_idx:pole_end_idx])
        pole_drop = (pole_high - pole_low) / pole_high

        # Check if pole is valid (significant drop)
        if pole_drop < min_pole_drop:
            continue

        # Flag period: consolidation after the pole
        flag_start_idx = pole_end_idx
        flag_end_idx = i

        flag_highs = high[flag_start_idx:flag_end_idx]
        flag_lows = low[flag_start_idx:flag_end_idx]
        flag_closes = close[flag_start_idx:flag_end_idx]

        current_flag_high = np.max(flag_highs)
        current_flag_low = np.min(flag_lows)
        flag_range = (current_flag_high - current_flag_low) / current_flag_low

        # Flag should be smaller than pole (consolidation)
        if flag_range > pole_drop * 0.5:
            continue

        # Flag should be trending slightly upward or sideways (retracement)
        flag_trend = (flag_closes[-1] - flag_closes[0]) / flag_closes[0]

        # Valid bear flag: slight upward retracement (counter-trend move)
        if 0 < flag_trend < pole_drop * 0.5:
            bear_flag[i] = 1
            flag_high[i] = current_flag_high
            flag_low[i] = current_flag_low
            # Target: measured move (pole length projected from flag low)
            pole_length = pole_high - pole_low
            target[i] = current_flag_low - pole_length

    return (pd.Series(bear_flag, index=data.index),
            pd.Series(flag_high, index=data.index),
            pd.Series(flag_low, index=data.index),
            pd.Series(target, index=data.index))


def find_swing_lows(data, lookback=20):
    """Find significant swing lows for target identification"""
    low = data['Low'].values
    n = len(low)
    swing_lows = np.full(n, np.nan)

    for i in range(lookback, n):
        window_low = np.min(low[i-lookback:i])
        swing_lows[i] = window_low

    return pd.Series(swing_lows, index=data.index)


class BearFlagStrategy(Strategy):
    """
    Bear Flag Pattern Trading Strategy

    Parameters:
    - pole_lookback: Lookback period for detecting the pole (sharp decline)
    - flag_lookback: Lookback period for flag consolidation
    - min_pole_drop: Minimum pole decline percentage
    - atr_period: ATR period for stop loss
    - stop_atr_mult: ATR multiplier for stop loss above flag high
    - risk_reward: Minimum risk/reward ratio
    """

    pole_lookback = 20
    flag_lookback = 10
    min_pole_drop = 0.08  # 8% minimum drop for pole
    atr_period = 14
    stop_atr_mult = 1.5
    risk_reward = 2.0
    swing_low_lookback = 50

    def init(self):
        # Detect bear flag patterns
        self.bear_flag, self.flag_high, self.flag_low, self.measured_target = \
            self.I(lambda: detect_bear_flag(
                self.data.df, self.pole_lookback, self.flag_lookback, self.min_pole_drop
            ), plot=False)

        # Swing lows for target identification
        self.swing_low = self.I(lambda: find_swing_lows(
            self.data.df, self.swing_low_lookback
        ), plot=False)

        # ATR for stop loss
        self.atr = self.I(ta.atr,
                          pd.Series(self.data.High),
                          pd.Series(self.data.Low),
                          pd.Series(self.data.Close),
                          self.atr_period)

        # RSI for confirmation
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), 14)

        # EMA for trend context
        self.ema_50 = self.I(ta.ema, pd.Series(self.data.Close), 50)
        self.ema_200 = self.I(ta.ema, pd.Series(self.data.Close), 200)

        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.lowest_since_entry = None

    def next(self):
        if len(self.data) < max(self.pole_lookback + self.flag_lookback, 50) + 10:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr == 0:
            return

        bear_flag = self.bear_flag[-1]
        flag_high = self.flag_high[-1]
        flag_low = self.flag_low[-1]
        measured_target = self.measured_target[-1]
        swing_low = self.swing_low[-1]
        rsi = self.rsi[-1] if not np.isnan(self.rsi[-1]) else 50
        ema_50 = self.ema_50[-1] if not np.isnan(self.ema_50[-1]) else price
        ema_200 = self.ema_200[-1] if not np.isnan(self.ema_200[-1]) else price

        # Position management
        if self.position:
            if self.position.is_short:
                # Track lowest price since entry
                if self.lowest_since_entry is None:
                    self.lowest_since_entry = price
                else:
                    self.lowest_since_entry = min(self.lowest_since_entry, price)

                # Trailing stop: move stop down as price falls
                trailing_stop = self.lowest_since_entry + self.stop_atr_mult * atr
                if trailing_stop < self.stop_loss:
                    self.stop_loss = trailing_stop

                # Exit conditions
                exit_signal = False

                # 1. Take profit at measured target or swing low
                if self.take_profit and price <= self.take_profit:
                    exit_signal = True

                # 2. Stop loss hit
                elif price >= self.stop_loss:
                    exit_signal = True

                # 3. RSI oversold exit (partial profit taking)
                elif rsi < 25:
                    exit_signal = True

                if exit_signal:
                    self.position.close()
                    self.reset_position_vars()

            elif self.position.is_long:
                # Long position management (for bull flag variant)
                if self.highest_since_entry is None:
                    self.highest_since_entry = price
                else:
                    self.highest_since_entry = max(self.highest_since_entry, price)

                trailing_stop = self.highest_since_entry - self.stop_atr_mult * atr

                if price <= trailing_stop or (self.take_profit and price >= self.take_profit):
                    self.position.close()
                    self.reset_position_vars()

        else:
            # Entry logic - Bear Flag Short
            if bear_flag == 1 and not np.isnan(flag_high):
                # Price should be near flag high (resistance)
                near_resistance = price >= flag_high * 0.98

                # Trend confirmation: below longer-term EMA suggests bearish context
                bearish_context = price < ema_200 or ema_50 < ema_200

                # RSI not oversold (room to fall)
                rsi_ok = rsi > 35

                # Calculate risk/reward
                potential_stop = flag_high + self.stop_atr_mult * atr

                # Use swing low or measured target, whichever is closer
                if not np.isnan(swing_low) and swing_low < price:
                    target = max(swing_low, measured_target) if not np.isnan(measured_target) else swing_low
                elif not np.isnan(measured_target):
                    target = measured_target
                else:
                    target = price * 0.85  # Default 15% target

                risk = potential_stop - price
                reward = price - target

                if risk > 0 and reward / risk >= self.risk_reward:
                    if near_resistance and rsi_ok:
                        self.entry_price = price
                        self.stop_loss = potential_stop
                        self.take_profit = target
                        self.lowest_since_entry = price
                        self.sell()

            # Also detect Bull Flag for long opportunities
            # (Inverse of bear flag - sharp rise + consolidation)
            self._check_bull_flag(price, atr, rsi, ema_50, ema_200)

    def _check_bull_flag(self, price, atr, rsi, ema_50, ema_200):
        """Check for bull flag patterns (inverse of bear flag)"""
        # Simplified bull flag detection using recent price action
        lookback = self.pole_lookback + self.flag_lookback
        if len(self.data) < lookback + 5:
            return

        highs = list(self.data.High)[-lookback:]
        lows = list(self.data.Low)[-lookback:]
        closes = list(self.data.Close)[-lookback:]

        # Check for pole (sharp rise)
        pole_start = 0
        pole_end = self.pole_lookback
        pole_low = min(lows[pole_start:pole_end])
        pole_high = max(highs[pole_start:pole_end])
        pole_rise = (pole_high - pole_low) / pole_low

        if pole_rise < self.min_pole_drop:
            return

        # Check for flag (consolidation)
        flag_start = pole_end
        flag_end = lookback
        flag_high = max(highs[flag_start:flag_end])
        flag_low = min(lows[flag_start:flag_end])
        flag_range = (flag_high - flag_low) / flag_low

        if flag_range > pole_rise * 0.5:
            return

        # Flag should be slightly downward (retracement)
        flag_trend = (closes[-1] - closes[self.pole_lookback]) / closes[self.pole_lookback]

        if -pole_rise * 0.5 < flag_trend < 0:
            # Bull flag detected
            near_support = price <= flag_low * 1.02
            bullish_context = price > ema_200 or ema_50 > ema_200
            rsi_ok = rsi < 65

            potential_stop = flag_low - self.stop_atr_mult * atr
            target = price + (pole_high - pole_low)  # Measured move

            risk = price - potential_stop
            reward = target - price

            if risk > 0 and reward / risk >= self.risk_reward:
                if near_support and rsi_ok and bullish_context:
                    self.entry_price = price
                    self.stop_loss = potential_stop
                    self.take_profit = target
                    self.highest_since_entry = price
                    self.buy()

    def reset_position_vars(self):
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.lowest_since_entry = None
        self.highest_since_entry = None


def load_and_prepare_data(file_path):
    """Load CSV and prepare for backtesting.py format"""
    df = pd.read_csv(file_path)

    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'timestamp': 'Datetime'
    })

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
    df = df.sort_index()

    return df


def run_backtest(file_path, cash=100000, commission=0.001):
    """Run backtest on a single file"""
    print(f"\n{'='*60}")
    print(f"Backtesting: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        df = load_and_prepare_data(file_path)

        if len(df) < 100:
            print(f"  Skipping: Not enough data ({len(df)} rows)")
            return None

        print(f"  Data range: {df.index[0]} to {df.index[-1]}")
        print(f"  Total bars: {len(df)}")
        print(f"  Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")

        bt = Backtest(
            df,
            BearFlagStrategy,
            cash=cash,
            commission=commission,
            exclusive_orders=True
        )

        stats = bt.run()
        return stats

    except Exception as e:
        print(f"  Error: {str(e)}")
        return None


def format_stats(stats):
    """Format backtest statistics for display"""
    if stats is None:
        return "No results"

    output = []
    output.append(f"  Return [%]:              {stats['Return [%]']:.2f}%")
    output.append(f"  Buy & Hold Return [%]:   {stats['Buy & Hold Return [%]']:.2f}%")
    output.append(f"  Max Drawdown [%]:        {stats['Max. Drawdown [%]']:.2f}%")
    output.append(f"  # Trades:                {stats['# Trades']}")

    if stats['# Trades'] > 0:
        output.append(f"  Win Rate [%]:            {stats['Win Rate [%]']:.2f}%")
        pf = stats['Profit Factor']
        output.append(f"  Profit Factor:           {pf:.2f}" if not np.isnan(pf) else "  Profit Factor:           N/A")
        sr = stats['Sharpe Ratio']
        output.append(f"  Sharpe Ratio:            {sr:.2f}" if not np.isnan(sr) else "  Sharpe Ratio:            N/A")
        output.append(f"  Avg Trade [%]:           {stats['Avg. Trade [%]']:.2f}%")
        output.append(f"  Best Trade [%]:          {stats['Best Trade [%]']:.2f}%")
        output.append(f"  Worst Trade [%]:         {stats['Worst Trade [%]']:.2f}%")

    output.append(f"  Final Equity:            ${stats['Equity Final [$]']:,.2f}")

    return "\n".join(output)


def save_results(all_results, summary_stats, output_dir):
    """Save backtest results to execution_results directory"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = os.path.join(output_dir, f"bear_flag_results_{timestamp}.json")
    output_data = {
        "strategy": "Bear Flag Pattern Strategy",
        "source": "https://www.tradingview.com/chart/BTCUSDT/qq6ih8sa-Bitcoin-Sell-this-upcoming-pump-New-Bear-Flag-Target-74k/",
        "execution_time": timestamp,
        "parameters": {
            "pole_lookback": 20,
            "flag_lookback": 10,
            "min_pole_drop": 0.08,
            "atr_period": 14,
            "stop_atr_mult": 1.5,
            "risk_reward": 2.0
        },
        "summary": summary_stats,
        "per_file_results": all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    csv_file = os.path.join(output_dir, f"bear_flag_summary_{timestamp}.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)

    print(f"\n  Results saved to: {results_file}")
    print(f"  Summary saved to: {csv_file}")

    return output_data


def main():
    """Main function to run backtests on all OHLCV files"""
    print("="*60)
    print("Bear Flag Pattern Strategy Backtest")
    print("Based on TradingView analysis")
    print("="*60)

    base_dir = os.path.dirname(__file__)
    ohlcv_dir = os.path.join(base_dir, "..", "ohlcv")
    output_dir = os.path.join(base_dir, "..", "execution_results")
    ohlcv_files = glob.glob(os.path.join(ohlcv_dir, "*.csv"))

    if not ohlcv_files:
        print(f"No CSV files found in {ohlcv_dir}")
        return None, None

    print(f"\nFound {len(ohlcv_files)} data file(s)")

    all_results = []

    for file_path in ohlcv_files:
        bt_stats = run_backtest(file_path)

        if bt_stats is not None:
            print("\n  Results:")
            print(format_stats(bt_stats))

            all_results.append({
                'file': os.path.basename(file_path),
                'return': float(bt_stats['Return [%]']),
                'buy_hold_return': float(bt_stats['Buy & Hold Return [%]']),
                'trades': int(bt_stats['# Trades']),
                'win_rate': float(bt_stats['Win Rate [%]']) if bt_stats['# Trades'] > 0 else 0,
                'max_dd': float(bt_stats['Max. Drawdown [%]']),
                'sharpe': float(bt_stats['Sharpe Ratio']) if not np.isnan(bt_stats['Sharpe Ratio']) else 0,
                'profit_factor': float(bt_stats['Profit Factor']) if not np.isnan(bt_stats['Profit Factor']) else 0,
                'avg_trade': float(bt_stats['Avg. Trade [%]']) if bt_stats['# Trades'] > 0 else 0,
                'final_equity': float(bt_stats['Equity Final [$]'])
            })

    if all_results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        df_results = pd.DataFrame(all_results)

        summary_stats = {
            'total_files': len(df_results),
            'avg_return': float(df_results['return'].mean()),
            'total_trades': int(df_results['trades'].sum()),
            'avg_win_rate': float(df_results['win_rate'].mean()),
            'avg_max_dd': float(df_results['max_dd'].mean()),
            'avg_sharpe': float(df_results['sharpe'].mean()),
            'avg_profit_factor': float(df_results['profit_factor'].mean())
        }

        print(f"\nTotal files tested: {summary_stats['total_files']}")
        print(f"Average Return: {summary_stats['avg_return']:.2f}%")
        print(f"Total Trades: {summary_stats['total_trades']}")
        print(f"Average Win Rate: {summary_stats['avg_win_rate']:.2f}%")
        print(f"Average Max Drawdown: {summary_stats['avg_max_dd']:.2f}%")
        print(f"Average Sharpe Ratio: {summary_stats['avg_sharpe']:.2f}")

        print("\n" + "-"*60)
        print("Per-file Summary (Top 10 by return):")
        print("-"*60)
        df_sorted = df_results.sort_values('return', ascending=False).head(10)
        for _, row in df_sorted.iterrows():
            print(f"  {row['file']:<25} Return: {row['return']:>8.2f}%  Trades: {row['trades']:>3}  Win: {row['win_rate']:>5.1f}%")

        output_data = save_results(all_results, summary_stats, output_dir)

        return all_results, output_data

    return None, None


if __name__ == "__main__":
    results, output_data = main()
