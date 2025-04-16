import time
import threading
import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import mplcursors
from matplotlib.widgets import Button
from matplotlib.patches import Circle

# ====== CONFIGURATION ======
CSV_FILE = "price_long2.csv"
TRADE_LOG_FILE = "trade_log2.csv"
SEARCH_TERM = "MINI L SP500 NORDNET 325".capitalize()

SMA_PERIOD = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
TARGET_PROFIT = 0.005  # +0.3% net
STOP_LOSS = 0.003      # -0.3%
MAX_POSITION_COUNT = 5
SCALE_IN_COOLDOWN = 5
MAX_PRICE_JUMP_PCT = 0.20
MEMORY = 500

# ====== STATE CLASS ======
class StrategyState:
    def __init__(self):
        self.active_trade = None
        self.all_closed_trades = []  # persistent record for performance metrics
        self.buy_signals = []        # list of (timestamp, price)
        self.sell_signals = []       # list of (timestamp, price, gain)
        self.last_processed_index = 0

state = StrategyState()

# >>>> GLOBAL FLAGS & TRACKERS <<<<
trading_enabled = False
most_recent_bid = None
most_recent_time = None
manual_exit_requested = False

# ====== SETUP SELENIUM ======
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

def selenium_setup():
    driver.get("https://www.nordnet.se/se")
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="cookie-accept-all-secondary"]'))).click()
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="app"]/div/header/div[2]/div/div[2]/div/div/div/div/div[4]/div/div/div/div[1]/div/ul/button[2]'))).click()
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="main-content"]/div/div/div[3]/div[2]/div/div[1]/div/div/div[1]/button'))).click()
    print("🟡 Waiting 10 seconds for manual login (BankID)...")
    time.sleep(10)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="app"]/div/header/div[2]/div/div[2]/div/div/div/div/div[4]/div/div/div/div[2]/button'))).click()
    search_input = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="2val-Drawer"]/form/div/label/span/span/div/input')))
    search_input.send_keys(SEARCH_TERM)
    time.sleep(1)
    wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div[2]/div/div/div[3]/div/div/div/div/div[2]/div/ul/li[1]/a/div'))).click()

def initialize_trade_log():
    with open(TRADE_LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "EventType", "Price", "PositionCount", "TotalShares",
                         "AverageEntryPrice", "RealizedNetReturn", "HoldTimeSeconds", "ScaleInCount"])

def log_trade_event(timestamp, event_type, price, pos_count, tot_shares, avg_price, realized_return=None, hold_time=None, scale_in_count=None):
    with open(TRADE_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, price, pos_count, tot_shares, avg_price, realized_return, hold_time, scale_in_count])

def start_scraping():
    price_xpath = '//*[@id="main-content"]/div/div[1]/div/div/header/div/div/div[2]/div[1]'
    prev_bid, prev_ask = None, None
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "bid", "ask"])
    while True:
        try:
            price_text = driver.find_element(By.XPATH, price_xpath).text
            lines = price_text.splitlines()
            bid = ask = None
            for i in range(len(lines)):
                if lines[i] == "Köp":
                    bid = lines[i + 1].strip().replace(",", ".")
                elif lines[i] == "Sälj":
                    ask = lines[i + 1].strip().replace(",", ".")
            if bid and ask and (bid != prev_bid or ask != prev_ask):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, bid, ask])
                print(f"[{timestamp}] Bid: {bid}, Ask: {ask}")
                prev_bid, prev_ask = bid, ask
            time.sleep(0.5)
        except Exception as e:
            print("Scraper error:", e)
            time.sleep(1)

def compute_sma(series, window=50):
    return series.rolling(window=window).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def clean_data(df):
    cleaned_rows = []
    last_valid_bid = None
    last_valid_ask = None
    for _, row in df.iterrows():
        bid = row["bid"]
        ask = row["ask"]
        if bid <= 0 or ask <= 0:
            continue
        if last_valid_bid is None:
            cleaned_rows.append(row)
            last_valid_bid = bid
            last_valid_ask = ask
        else:
            bid_jump = abs(bid - last_valid_bid) / last_valid_bid
            ask_jump = abs(ask - last_valid_ask) / last_valid_ask
            if (bid_jump <= MAX_PRICE_JUMP_PCT) and (ask_jump <= MAX_PRICE_JUMP_PCT):
                cleaned_rows.append(row)
                last_valid_bid = bid
                last_valid_ask = ask
    return pd.DataFrame(cleaned_rows)

def open_new_trade(time_open, ask_price):
    avg = ask_price
    return {
        "time_open": time_open,
        "time_close": None,
        "price_open": ask_price,
        "average_entry": avg,
        "share_count": 1,
        "scale_in_count": 1,
        "time_last_scale_in": time_open,
        "tp_price": avg * (1 + TARGET_PROFIT),
        "sl_price": avg * (1 - STOP_LOSS),
        "breakeven_price": avg
    }

def scale_in_trade(trade_dict, ask_price, current_time):
    old_count = trade_dict["share_count"]
    old_avg = trade_dict["average_entry"]
    new_avg = (old_avg * old_count + ask_price) / (old_count + 1)
    trade_dict["average_entry"] = new_avg
    trade_dict["share_count"] += 1
    trade_dict["scale_in_count"] += 1
    trade_dict["time_last_scale_in"] = current_time
    trade_dict["tp_price"] = new_avg * (1 + TARGET_PROFIT)
    trade_dict["sl_price"] = new_avg * (1 - STOP_LOSS)
    trade_dict["breakeven_price"] = new_avg

def close_trade(trade_dict, time_close, bid_price):
    trade_dict["time_close"] = time_close
    net_return = (bid_price - trade_dict["average_entry"]) / trade_dict["average_entry"]
    hold_seconds = (time_close - trade_dict["time_open"]).total_seconds()
    closed_info = {
        "time_open": trade_dict["time_open"],
        "time_close": time_close,
        "price_open": trade_dict["price_open"],
        "price_close": bid_price,
        "average_entry": trade_dict["average_entry"],
        "share_count": trade_dict["share_count"],
        "scale_in_count": trade_dict["scale_in_count"],
        "net_return": net_return,
        "hold_seconds": hold_seconds,
        "tp_price": trade_dict["tp_price"],
        "sl_price": trade_dict["sl_price"],
        "breakeven_price": trade_dict["breakeven_price"]
    }
    return closed_info

def force_close_trade(reason="Manual Exit"):
    global state, most_recent_bid, most_recent_time
    if state.active_trade is not None and most_recent_bid is not None:
        closed_info = close_trade(state.active_trade, most_recent_time, most_recent_bid)
        state.all_closed_trades.append(closed_info)
        state.buy_signals.append(("FORCED SELL", most_recent_bid))
        state.sell_signals.append((most_recent_time, most_recent_bid, closed_info["net_return"]*100))
        log_trade_event(most_recent_time, reason, most_recent_bid, 0, 0, closed_info["average_entry"],
                        closed_info["net_return"]*100, closed_info["hold_seconds"], closed_info["scale_in_count"])
        print(f"Trade forcibly closed due to: {reason}.")
        state.active_trade = None

def process_new_rows(df):
    global state, trading_enabled
    start_idx = state.last_processed_index
    n = len(df)
    for idx in range(start_idx, n):
        row = df.iloc[idx]
        current_time = row["Timestamp"]
        ask_price = row["ask"]
        bid_price = row["bid"]
        sma_val = row["SMA_ask"]
        rsi_val = row["RSI_ask"]
        macd_val = row["MACD"]
        macd_sig = row["MACD_signal"]
        synergy_valid = (ask_price > sma_val and macd_val > macd_sig and rsi_val < 50)
        if state.active_trade is None:
            if synergy_valid:
                if trading_enabled:
                    state.active_trade = open_new_trade(current_time, ask_price)
                    state.buy_signals.append((current_time, ask_price))
                    log_trade_event(current_time, "Entry", ask_price, 1, 1, ask_price, None, None, 1)
                else:
                    state.buy_signals.append((current_time, ask_price))
        else:
            net_return = (bid_price - state.active_trade["average_entry"]) / state.active_trade["average_entry"]
            if net_return >= TARGET_PROFIT:
                if trading_enabled:
                    closed_info = close_trade(state.active_trade, current_time, bid_price)
                    state.all_closed_trades.append(closed_info)
                    state.sell_signals.append((current_time, bid_price, closed_info["net_return"]*100))
                    log_trade_event(current_time, "ExitTP", bid_price, 0, 0, closed_info["average_entry"],
                                    closed_info["net_return"]*100, closed_info["hold_seconds"], closed_info["scale_in_count"])
                    state.active_trade = None
                else:
                    state.sell_signals.append((current_time, bid_price, net_return*100))
            elif net_return <= -STOP_LOSS:
                if trading_enabled:
                    closed_info = close_trade(state.active_trade, current_time, bid_price)
                    state.all_closed_trades.append(closed_info)
                    state.sell_signals.append((current_time, bid_price, closed_info["net_return"]*100))
                    log_trade_event(current_time, "ExitSL", bid_price, 0, 0, closed_info["average_entry"],
                                    closed_info["net_return"]*100, closed_info["hold_seconds"], closed_info["scale_in_count"])
                    state.active_trade = None
                else:
                    state.sell_signals.append((current_time, bid_price, net_return*100))
            else:
                if net_return < 0 and synergy_valid:
                    time_diff = (current_time - state.active_trade["time_last_scale_in"]).total_seconds()
                    if state.active_trade["share_count"] < MAX_POSITION_COUNT and time_diff >= SCALE_IN_COOLDOWN:
                        if trading_enabled:
                            scale_in_trade(state.active_trade, ask_price, current_time)
                            state.buy_signals.append((current_time, ask_price))
                            log_trade_event(current_time, "ScaleIn", ask_price, state.active_trade["share_count"],
                                            state.active_trade["share_count"], state.active_trade["average_entry"],
                                            None, None, state.active_trade["scale_in_count"])
                        else:
                            state.buy_signals.append((current_time, ask_price))
    state.last_processed_index = n

def start_plotting():
    global state, trading_enabled, most_recent_bid, most_recent_time, manual_exit_requested
    initialize_trade_log()

    fig, ax_price = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3)
    ax_price.set_title("Multi-Indicator + Scaling + Data Cleaning + TP/SL Lines + 5s Delay")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Price (SEK)")

    # ===== BUTTON 1: Toggle Trading =====
    button_ax = plt.axes([0.05, 0.90, 0.08, 0.05])
    toggle_button = Button(button_ax, "INACTIVE")
    toggle_button.ax.set_facecolor("#ffdddd")  # pastel red

    circle = Circle((0.95, 0.95), 0.02, transform=ax_price.transAxes,
                    facecolor='red', edgecolor='black', linewidth=2)
    ax_price.add_patch(circle)

    def update_button_style():
        if trading_enabled:
            toggle_button.label.set_text("ACTIVE")
            toggle_button.ax.set_facecolor("#ddffdd")  # pastel green
            circle.set_facecolor('lightgreen')
        else:
            toggle_button.label.set_text("INACTIVE")
            toggle_button.ax.set_facecolor("#ffdddd")  # pastel red
            circle.set_facecolor('red')
        plt.draw()  # <- force redraw to update color

    def toggle_trading(_):
        global trading_enabled, state
        prev = trading_enabled
        trading_enabled = not trading_enabled
        update_button_style()
        if not trading_enabled and prev and state.active_trade is not None:
            force_close_trade(reason="Disable Trading")

    toggle_button.on_clicked(toggle_trading)

    exit_button_ax = plt.axes([0.05, 0.84, 0.08, 0.05])
    exit_button = Button(exit_button_ax, "EXIT", color="red", hovercolor="white")
    def exit_now(_):
        global manual_exit_requested
        manual_exit_requested = True
    exit_button.on_clicked(exit_now)

    def animate(_):
        global state, trading_enabled, most_recent_bid, most_recent_time, manual_exit_requested
        try:
            df = pd.read_csv(CSV_FILE)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["bid"] = pd.to_numeric(df["bid"], errors='coerce')
            df["ask"] = pd.to_numeric(df["ask"], errors='coerce')
            df.dropna(inplace=True)
            df = clean_data(df)
            if df.empty:
                ax_price.clear()
                ax_price.set_title("No valid data yet (all rows filtered).")
                return
            df.sort_values("Timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            # Compute indicators on full df
            df["SMA_ask"] = compute_sma(df["ask"], SMA_PERIOD)
            df["RSI_ask"] = compute_rsi(df["ask"], RSI_PERIOD)
            macd_line, macd_signal = compute_macd(df["ask"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            df["MACD"] = macd_line
            df["MACD_signal"] = macd_signal
            # Update persistent state by processing only new rows
            process_new_rows(df)
            most_recent_time = df["Timestamp"].iloc[-1]
            most_recent_bid = df["bid"].iloc[-1]
            if manual_exit_requested and state.active_trade is not None:
                force_close_trade(reason="Manual ExitNow")
                manual_exit_requested = False
            # For plotting, use only the last MEMORY rows
            if len(df) > MEMORY:
                df_plot = df.iloc[-MEMORY:].copy()
            else:
                df_plot = df.copy()
            ax_price.clear()
            ax_price.plot(df_plot["Timestamp"], df_plot["bid"], label="Bid", lw=1)
            ax_price.plot(df_plot["Timestamp"], df_plot["ask"], label="Ask", lw=1)
            df_plot["SMA_ask"] = compute_sma(df_plot["ask"], SMA_PERIOD)
            ax_price.plot(df_plot["Timestamp"], df_plot["SMA_ask"], label=f"SMA({SMA_PERIOD}) on Ask", lw=1)
            start_time = df_plot["Timestamp"].iloc[0]
            end_time = df_plot["Timestamp"].iloc[-1]
            buy_signals_plot = [(t, p) for (t, p) in state.buy_signals if start_time <= t <= end_time]
            sell_signals_plot = [(t, p, g) for (t, p, g) in state.sell_signals if start_time <= t <= end_time]
            for t, p in buy_signals_plot:
                ax_price.plot(t, p, marker='^', color='green', markersize=10, label="Buy")
            for t, p, g in sell_signals_plot:
                ax_price.plot(t, p, marker='v', color='red', markersize=10, label="Sell")
            if sell_signals_plot:
                scatter_times = [t for (t, _, _) in sell_signals_plot]
                scatter_prices = [p for (_, p, _) in sell_signals_plot]
                sc = ax_price.scatter(scatter_times, scatter_prices, alpha=0.0)
                annotator = mplcursors.cursor(sc, hover=True)
                def on_add(sel):
                    idx_sel = sel.index
                    _, price_sel, gain_sel = sell_signals_plot[idx_sel]
                    sel.annotation.set_text(f"Sell\nGain: {gain_sel:.2f}%")
                annotator.connect("add", on_add)
            if state.active_trade is not None:
                tp_price = state.active_trade["tp_price"]
                sl_price = state.active_trade["sl_price"]
                be_price = state.active_trade["breakeven_price"]
                line_start = max(state.active_trade["time_open"], start_time)
                ax_price.plot([line_start, end_time], [tp_price, tp_price], color='green', linestyle='--', lw=2)
                ax_price.plot([line_start, end_time], [sl_price, sl_price], color='red', linestyle='--', lw=2)
                ax_price.plot([line_start, end_time], [be_price, be_price], color='blue', linestyle='--', lw=2)
            ax_price.legend(loc="upper left")
            ax_price.grid(True)
            total_closed = len(state.all_closed_trades)
            if total_closed > 0:
                wins = sum(1 for t in state.all_closed_trades if t["net_return"] >= 0)
                win_rate = (wins / total_closed) * 100
                avg_hold = np.mean([t["hold_seconds"] for t in state.all_closed_trades])
                avg_scale_in = np.mean([t["scale_in_count"] for t in state.all_closed_trades])
                performance_text = (f"Trades: {total_closed}\nWin rate: {win_rate:.1f}%\nAvg hold: {int(avg_hold)}s\nAvg scale-in: {avg_scale_in:.2f}")
            else:
                performance_text = "No closed trades yet."
            ax_price.text(0.05, 0.05, performance_text, transform=ax_price.transAxes, fontsize=10,
                          verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))
        except Exception as e:
            print("Plotting error:", e)
    ani = animation.FuncAnimation(fig, animate, interval=1500)
    plt.show()

if __name__ == "__main__":
    selenium_setup()
    scraper_thread = threading.Thread(target=start_scraping)
    scraper_thread.daemon = True
    scraper_thread.start()
    start_plotting()
