import json
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# 1. 大盤基準：盈富基金
BENCHMARK = "2800.HK"
BENCHMARK_NAME = "盈富基金 (2800.HK)"

# 2. 板塊相對強度基準：南方恒生科技 ETF
SECTOR_ETFS = {
    "3033.HK": "南方恒生科技 ETF (3033.HK)",
}

# 3. 港股通科技、AI、人形機器人與半導體觀察池（Yahoo Finance 港股需使用 4 位數字代號）
WATCHLIST = {
    "0700.HK": {"name": "騰訊控股", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "9988.HK": {"name": "阿里巴巴-W", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "1810.HK": {"name": "小米集團-W", "sector": "3033.HK", "sub_sector": "智能生態與電動車"},
    "9880.HK": {"name": "優必選", "sector": "3033.HK", "sub_sector": "人形機器人"},
    "9660.HK": {"name": "地平線機器人-W", "sector": "3033.HK", "sub_sector": "智駕與車規晶片"},
    "2533.HK": {"name": "黑芝麻智能", "sector": "3033.HK", "sub_sector": "智駕與車規晶片"},
    "0981.HK": {"name": "中芯國際", "sector": "3033.HK", "sub_sector": "半導體製造"},
    "1347.HK": {"name": "華虹半導體", "sector": "3033.HK", "sub_sector": "半導體製造"},
    "0020.HK": {"name": "商湯-W", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
    "9698.HK": {"name": "萬國數據-SW", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL20"]
    df["BASE_HIGH_60"] = df["High"].shift(1).rolling(60).max()
    df["BASE_LOW_60"] = df["Low"].shift(1).rolling(60).min()
    # 向上跳空缺口：當日最低價大於前日最高價
    df["GAP_UP"] = df["Low"] > df["High"].shift(1)
    df["GAP_DOWN"] = df["High"] < df["Low"].shift(1)
    return df


def extract_symbol_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        df = raw[symbol].copy()
    else:
        df = raw.copy()
    df = df.dropna(subset=["Close", "Volume"])
    # 過濾停牌或零成交量交易日
    df = df[df["Volume"] > 0]
    return df


def classify_stock(
    ticker: str,
    meta: dict,
    df: pd.DataFrame,
    sector_strong: bool,
) -> dict:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    sma200_20d_ago = df["SMA200"].iloc[-20]

    close = float(latest["Close"])
    sma50 = float(latest["SMA50"])
    sma150 = float(latest["SMA150"])
    sma200 = float(latest["SMA200"])
    vol_ratio = float(latest["VOL_RATIO"]) if pd.notna(latest["VOL_RATIO"]) else 0.0
    base_high = float(latest["BASE_HIGH_60"])
    base_low = float(latest["BASE_LOW_60"])

    sma200_slope = (sma200 - sma200_20d_ago) / sma200_20d_ago * 100
    dist_50 = (close - sma50) / sma50 * 100
    dist_200 = (close - sma200) / sma200 * 100

    # 檢查近 5 個交易日是否存在未回補的向上跳空缺口
    recent_5 = df.iloc[-5:]
    unfilled_gap_level = None
    for i in range(len(recent_5)):
        row = recent_5.iloc[i]
        if row["GAP_UP"]:
            prev_high = float(df.iloc[-5 + i - 1]["High"])
            subsequent_low = float(recent_5.iloc[i:]["Low"].min())
            if subsequent_low > prev_high:
                unfilled_gap_level = round(prev_high, 2)

    sector_etf = meta["sector"]
    sub_sector = meta["sub_sector"]

    # 四階段分類與實戰風控建議
    stage = "Stage 1 底部震盪"
    signal_type = "neutral"
    action_note = f"觀察箱體整理 (HK${base_low:.2f}–HK${base_high:.2f})，等待帶量突破頸線與 200 日線"

    broke_out_today = close > base_high and prev["Close"] <= base_high
    above_200 = close > sma200

    if above_200 and (
        broke_out_today
        or (close > base_high * 0.98 and dist_200 < 12 and sma200_slope > -1.0)
    ):
        if vol_ratio >= 1.5:
            stage = "Stage 2A 理想突破"
            signal_type = "buy_2a"
            action_note = (
                f"油箱有老虎（量比 {vol_ratio:.1f}x）：突破建倉 50%，"
                f"回測 HK${base_high:.2f} 頸線守穩再加碼 50%"
            )
        else:
            stage = "Stage 2A 觀察（量能待補）"
            signal_type = "watch"
            action_note = f"已站上 200 日線與前高，但量比僅 {vol_ratio:.1f}x，防範假突破"
    elif above_200 and close < sma50:
        stage = "Stage 2 轉弱（破 50 日線）"
        signal_type = "exit_50"
        action_note = f"現代版防守警訊：跌破 50 日線 (HK${sma50:.2f})，宜果斷離場觀望"
    elif above_200 and dist_200 >= 20:
        stage = "Stage 2B 延伸末段"
        signal_type = "trim_2b"
        action_note = f"遠離 200 日線 (+{dist_200:.1f}%)，若出現後期缺口宜分批止盈減碼"
    elif above_200 and close >= sma50:
        stage = "Stage 2 上升趨勢"
        signal_type = "hold_2"
        action_note = f"穩守 50 日線 (HK${sma50:.2f}) 上方，續抱並以 50 日線作為移動防守線"
    elif not above_200 and prev["Close"] >= prev["SMA200"] and sma50 < sma150:
        stage = "Stage 4A 轉空警訊"
        signal_type = "short_4a"
        action_note = (
            f"跌破 200 日線轉弱；空單停損參考：站回 150 日線 (HK${sma150:.2f}) 減半，"
            f"站回 50 日線 (HK${sma50:.2f}) 全平"
        )
    elif not above_200 and close < sma50 and sma50 < sma200:
        stage = "Stage 4 下跌趨勢"
        signal_type = "avoid_4"
        action_note = "處於第四階段空頭格局，切忌逆勢抄底或向下攤平"
    else:
        stage = "Stage 3 / 過渡震盪"
        signal_type = "stage_3"
        action_note = f"均線糾結或頭部派發區，留意 200 日線 (HK${sma200:.2f}) 攻防"

    return {
        "ticker": ticker,
        "name": meta["name"],
        "sector_etf": sector_etf,
        "sector_name": SECTOR_ETFS.get(sector_etf, sector_etf),
        "sub_sector": sub_sector,
        "sector_strong": sector_strong,
        "close": round(close, 2),
        "change_pct": round((close - float(prev["Close"])) / float(prev["Close"]) * 100, 2),
        "sma50": round(sma50, 2),
        "sma150": round(sma150, 2),
        "sma200": round(sma200, 2),
        "dist_50": round(dist_50, 2),
        "dist_200": round(dist_200, 2),
        "vol_ratio": round(vol_ratio, 2),
        "base_high": round(base_high, 2),
        "base_low": round(base_low, 2),
        "unfilled_gap": unfilled_gap_level,
        "stage": stage,
        "signal_type": signal_type,
        "action_note": action_note,
    }


def main():
    all_symbols = list(set([BENCHMARK] + list(SECTOR_ETFS.keys()) + list(WATCHLIST.keys())))

    raw = yf.download(
        all_symbols,
        period="15mo",
        interval="1d",
        group_by="ticker",
        progress=False,
    )

    processed = {}
    for sym in all_symbols:
        df = extract_symbol_df(raw, sym)
        if len(df) >= 210:
            processed[sym] = compute_indicators(df)
        else:
            print(f"[Skip] {sym}: 有效交易日不足 210 日 ({len(df)} 日)")

    if BENCHMARK not in processed:
        raise RuntimeError(f"無法取得大盤基準 {BENCHMARK} 數據，終止掃描。")

    # 1. 大盤基準評估：盈富基金 (2800.HK)
    bench_df = processed[BENCHMARK]
    bench_latest = bench_df.iloc[-1]
    bench_ret_20d = (bench_latest["Close"] / bench_df["Close"].iloc[-20]) - 1
    market_above_200 = bool(bench_latest["Close"] > bench_latest["SMA200"])
    market_above_50 = bool(bench_latest["Close"] > bench_latest["SMA50"])

    # 2. 板塊相對強度評估：南方恒生科技 ETF (3033.HK) 對比 2800.HK
    sectors_summary = []
    sector_strength_map = {}
    for etf, label in SECTOR_ETFS.items():
        if etf in processed:
            sdf = processed[etf]
            sl = sdf.iloc[-1]
            s_ret_20d = (sl["Close"] / sdf["Close"].iloc[-20]) - 1
            rs_vs_bench = (s_ret_20d - bench_ret_20d) * 100
            is_strong = bool(
                sl["Close"] > sl["SMA50"]
                and sl["Close"] > sl["SMA200"]
                and rs_vs_bench >= 0
            )
            sector_strength_map[etf] = is_strong
            sectors_summary.append({
                "etf": etf,
                "name": label,
                "close": round(float(sl["Close"]), 2),
                "sma50": round(float(sl["SMA50"]), 2),
                "sma200": round(float(sl["SMA200"]), 2),
                "above_50": bool(sl["Close"] > sl["SMA50"]),
                "above_200": bool(sl["Close"] > sl["SMA200"]),
                "rs_20d": round(float(rs_vs_bench), 2),
                "is_strong": is_strong,
            })

    # 3. 掃描港股通科技、AI、人形機器人與半導體個股
    results = []
    for ticker, meta in WATCHLIST.items():
        if ticker in processed:
            item = classify_stock(
                ticker,
                meta,
                processed[ticker],
                sector_strength_map.get(meta["sector"], False),
            )
            results.append(item)

    hkt = timezone(timedelta(hours=8))
    payload = {
        "updated_at": datetime.now(hkt).strftime("%Y-%m-%d %H:%M HKT"),
        "market": {
            "symbol": BENCHMARK_NAME,
            "close": round(float(bench_latest["Close"]), 2),
            "sma50": round(float(bench_latest["SMA50"]), 2),
            "sma200": round(float(bench_latest["SMA200"]), 2),
            "above_50": market_above_50,
            "above_200": market_above_200,
            "regime": "多頭健康（站穩 50 & 200 日線）"
            if (market_above_50 and market_above_200)
            else ("震盪警戒（跌破 50 日線，考驗 200 日線）" if market_above_200 else "空頭警戒（跌破 200 日線）"),
        },
        "sectors": sectors_summary,
        "stocks": results,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/scan_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"HK Scan complete: {len(results)} stocks saved to data/scan_results.json")


if __name__ == "__main__":
    main()
