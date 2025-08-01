import time
import json
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from pocketoptionapi.stable_api import PocketOption
import pocketoptionapi.global_value as global_value
from xgboost import XGBClassifier

# Load environment variables
load_dotenv()

# Session configuration
start_counter = time.perf_counter()

ssid = '42["auth",{"session":"7onedcpaoiv8natb8jl7vp2728","isDemo":1,"uid":73357779,"platform":2,"isFastHistory":true,"isOptimized":true}]'
demo = True

# Bot Settings
min_payout = 80
period = 60 
expiration = 60
INITIAL_AMOUNT = 1
MARTINGALE_LEVEL = 3
MIN_ACTIVE_PAIRS = 2
PROB_THRESHOLD = 0.76
TAKE_PROFIT = 20
current_profit = 0


api = PocketOption(ssid, demo)
api.connect()

FEATURE_COLS = ['RSI', 'k_percent', 'r_percent', 'MACD', 'MACD_EMA', 'Price_Rate_Of_Change']

def get_payout():
    try:
        d = json.loads(global_value.PayoutData)
        for pair in d:
            name = pair[1]
            payout = pair[5]
            asset_type = pair[3]
            is_active = pair[14]

            if not name.endswith("_otc") and asset_type == "currency" and is_active:
                if payout >= min_payout:
                    global_value.pairs[name] = {'payout': payout, 'type': asset_type}
                elif name in global_value.pairs:
                    del global_value.pairs[name]
        return True
    except Exception as e:
        global_value.logger(f"[ERROR]: Failed to parse payout data - {str(e)}", "ERROR")
        return False

def get_df():
    try:
        for i, pair in enumerate(global_value.pairs, 1):
            df = api.get_candles(pair, period)
            global_value.logger(f'{pair} ({i}/{len(global_value.pairs)})', "INFO")
            time.sleep(1)
        return True
    except:
        return False

def prepare_data(df):
    df = df[['time', 'open', 'high', 'low', 'close']]
    df.rename(columns={'time': 'timestamp'}, inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    df['change_in_price'] = df['close'].diff()

    rsi_period = 14
    stochastic_period = 14
    macd_ema_long = 26
    macd_ema_short = 12
    macd_signal = 9
    roc_period = 9

    up_df = df['change_in_price'].where(df['change_in_price'] > 0, 0)
    down_df = abs(df['change_in_price'].where(df['change_in_price'] < 0, 0))
    ewma_up = up_df.ewm(span=rsi_period).mean()
    ewma_down = down_df.ewm(span=rsi_period).mean()
    rs = ewma_up / ewma_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    df['low_14'] = df['low'].rolling(window=stochastic_period).min()
    df['high_14'] = df['high'].rolling(window=stochastic_period).max()
    df['k_percent'] = 100 * ((df['close'] - df['low_14']) / (df['high_14'] - df['low_14']))
    df['r_percent'] = ((df['high_14'] - df['close']) / (df['high_14'] - df['low_14'])) * -100

    ema_26 = df['close'].ewm(span=macd_ema_long).mean()
    ema_12 = df['close'].ewm(span=macd_ema_short).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_EMA'] = df['MACD'].ewm(span=macd_signal).mean()

    df['Price_Rate_Of_Change'] = df['close'].pct_change(periods=roc_period)
    df['Prediction'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df

def train_and_predict(df):
    X_train = df[FEATURE_COLS].iloc[:-1]
    y_train = df['Prediction'].iloc[:-1]

    # ✅ XGBoost Classifier
    model = XGBClassifier()
    model.fit(X_train, y_train)

    X_test = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(X_test)
    call_conf = proba[0][1]
    put_conf = 1 - call_conf

    latest_close = df.iloc[-1]['close']
    latest_ema26 = df['close'].ewm(span=26).mean().iloc[-1]

    if call_conf > PROB_THRESHOLD and latest_close > latest_ema26:
        decision = "call"
        emoji = "🟢"
        confidence = call_conf
    elif put_conf > PROB_THRESHOLD and latest_close < latest_ema26:
        decision = "put"
        emoji = "🔴"
        confidence = put_conf
    else:
        global_value.logger("⏭️ Skipping trade due to low confidence or trend mismatch.", "INFO")
        return None

    global_value.logger(f"{emoji} === PREDICTED: {decision.upper()} | CONFIDENCE: {confidence:.2%}", "INFO")
    return decision

def perform_trade(amount, pair, action, expiration):
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    trade_id = result[1]

    if result[0] is False or trade_id is None:
        global_value.logger("❗Trade failed to execute. Attempting reconnection...", "ERROR")
        api.disconnect()
        time.sleep(2)
        api.connect()
        return None

    time.sleep(expiration)
    return api.check_win(trade_id)

def martingale_strategy(pair, action):
    global current_profit

    amount = INITIAL_AMOUNT
    level = 1
    result = perform_trade(amount, pair, action, expiration)

    if result is None:
        return

    if result[1] == 'win':
        current_profit += amount * (global_value.pairs[pair]['payout'] / 100)
    else:
        current_profit -= amount

    while result[1] == 'loose' and level < MARTINGALE_LEVEL:
        level += 1
        amount *= 2
        result = perform_trade(amount, pair, action, expiration)

        if result is None:
            return

        if result[1] == 'win':
            current_profit += amount * (global_value.pairs[pair]['payout'] / 100)
            global_value.logger(f"✅ WIN - Profit: {current_profit:.2f} USD", "INFO")
            break
        else:
            current_profit -= amount
            global_value.logger(f"❌ LOSS - Profit: {current_profit:.2f} USD", "INFO")

    if current_profit >= TAKE_PROFIT:
        global_value.logger(f"🎯 Take Profit Achieved! Cooling down for 1 hour... Final Profit: {current_profit:.2f} USD", "INFO")
        time.sleep(3600)
        current_profit = 0

    if result[1] != 'loose':
        global_value.logger("WIN - Resetting to base amount.", "INFO")
    else:
        global_value.logger("LOSS. Resetting.", "INFO")

def wait_until_next_candle(period_seconds=300, seconds_before=15):
    while True:
        now = datetime.now(timezone.utc)
        next_candle = ((now.timestamp() // period_seconds) + 1) * period_seconds
        if now.timestamp() >= next_candle - seconds_before:
            break
        time.sleep(0.2)

def wait_for_candle_start():
    while True:
        now = datetime.now(timezone.utc)
        if now.second == 0 and now.minute % (period // 60) == 0:
            break
        time.sleep(0.1)

def near_github_timeout():
    return (time.perf_counter() - start_counter) >= (6 * 3600 - 20 * 60)

def strategie():
    pairs_snapshot = list(global_value.pairs.keys())

    if len(pairs_snapshot) < MIN_ACTIVE_PAIRS:
        time.sleep(60)
        prepare()
        return

    for i, pair in enumerate(pairs_snapshot, 1):
        live_pairs = list(global_value.pairs.keys())
        if len(live_pairs) < MIN_ACTIVE_PAIRS:
            time.sleep(60)
            prepare()
            return

        if pair not in global_value.pairs:
            continue

        payout = global_value.pairs[pair].get('payout', 0)
        if payout < min_payout:
            continue

        wait_until_next_candle(period, 15)

        df = global_value.pairs[pair].get('dataframe')
        if df is None or df.empty:
            continue

        df = df.sort_values(by='time').reset_index(drop=True)

        global_value.logger(f"{len(df)} Candles collected for === {pair} === ({period // 60} mins timeframe)", "INFO")

        processed_df = prepare_data(df.copy())
        if processed_df.empty:
            continue

        decision = train_and_predict(processed_df)

        if decision:
            latest_rsi = processed_df.iloc[-1]['RSI']
            if (decision == "call" and latest_rsi > 70) or (decision == "put" and latest_rsi < 30):
                global_value.logger(f"Skipping {decision.upper()} due to RSI filter: RSI = {latest_rsi:.2f}", "INFO")
                continue

            if near_github_timeout():
                global_value.logger("🕒 Near GitHub timeout. Skipping new trade to avoid interruption.", "INFO")
                return
            wait_for_candle_start()
            martingale_strategy(pair, decision)

            wait_until_next_candle(period, 60)
            get_payout()
            get_df()

def prepare():
    try:
        return get_payout() and get_df()
    except:
        return False

def start():
    while not global_value.websocket_is_connected:
        time.sleep(0.1)
    time.sleep(2)

    if prepare():
        while True:
            strategie()

if __name__ == "__main__":
    start()
    end_counter = time.perf_counter()
    global_value.logger(f"CPU-bound Task Time: {int(end_counter - start_counter)} seconds", "INFO")
