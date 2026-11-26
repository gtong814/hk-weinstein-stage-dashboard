import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import akshare as ak
except ImportError:
    ak = None

# 1. 大盤趨勢基準：盈富基金
BENCHMARK = "2800.HK"
BENCHMARK_NAME = "盈富基金 (2800.HK)"

# 2. 板塊相對強度基準：南方恒生科技 ETF
SECTOR_ETFS = {
    "3033.HK": "南方恒生科技 ETF (3033.HK)",
}

# 3. 港股通多賽道核心觀察池（作為細分賽道標籤庫與保底名單）
CORE_GGT_POOL = {
    "0700.HK": {"name": "騰訊控股", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "9988.HK": {"name": "阿里巴巴-W", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "3690.HK": {"name": "美團-W", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "9618.HK": {"name": "京東集團-SW", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "1024.HK": {"name": "快手-W", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "9888.HK": {"name": "百度集團-SW", "sector": "3033.HK", "sub_sector": "互聯網與 AI"},
    "1810.HK": {"name": "小米集團-W", "sector": "3033.HK", "sub_sector": "智能生態與機器人"},
    "9880.HK": {"name": "優必選", "sector": "3033.HK", "sub_sector": "智能生態與機器人"},
    "1211.HK": {"name": "比亞迪股份", "sector": "3033.HK", "sub_sector": "智能生態與機器人"},
    "2015.HK": {"name": "理想汽車-W", "sector": "3033.HK", "sub_sector": "智能生態與機器人"},
    "9868.HK": {"name": "小鵬汽車-W", "sector": "3033.HK", "sub_sector": "智能生態與機器人"},
    "9660.HK": {"name": "地平線機器人-W", "sector": "3033.HK", "sub_sector": "智駕與車規晶片"},
    "2533.HK": {"name": "黑芝麻智能", "sector": "3033.HK", "sub_sector": "智駕與車規晶片"},
    "2276.HK": {"name": "康耐特光學", "sector": "3033.HK", "sub_sector": "智駕與車規晶片"},
    "0981.HK": {"name": "中芯國際", "sector": "3033.HK", "sub_sector": "半導體製造"},
    "1347.HK": {"name": "華虹半導體", "sector": "3033.HK", "sub_sector": "半導體製造"},
    "0522.HK": {"name": "ASMPT", "sector": "3033.HK", "sub_sector": "半導體製造"},
    "0020.HK": {"name": "商湯-W", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
    "9698.HK": {"name": "萬國數據-SW", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
    "1686.HK": {"name": "新意網集團", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
    "0268.HK": {"name": "金蝶國際", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
    "3396.HK": {"name": "聯想控股", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
    "0992.HK": {"name": "聯想集團", "sector": "3033.HK", "sub_sector": "AI 大模型與算力"},
    "1833.HK": {"name": "平安好醫生", "sector": "3033.HK", "sub_sector": "醫療生技"},
    "6160.HK": {"name": "百濟神州", "sector": "3033.HK", "sub_sector": "醫療生技"},
    "2269.HK": {"name": "藥明生物", "sector": "3033.HK", "sub_sector": "醫療生技"},
    "1801.HK": {"name": "信達生物", "sector": "3033.HK", "sub_sector": "醫療生技"},
    "0388.HK": {"name": "香港交易所", "sector": "2800.HK", "sub_sector": "金融與高息"},
    "0939.HK": {"name": "建設銀行", "sector": "2800.HK", "sub_sector": "金融與高息"},
    "1658.HK": {"name": "郵儲銀行", "sector": "2800.HK", "sub_sector": "金融與高息"},
    "0941.HK": {"name": "中國移動", "sector": "2800.HK", "sub_sector": "金融與高息"},
    "0883.HK": {"name": "中國海洋石油", "sector": "2800.HK", "sub_sector": "金融與高息"},
    "0697.HK": {"name": "首程控股", "sector": "2800.HK", "sub_sector": "工業與基建"},
    "9981.HK": {"name": "沃爾核材", "sector": "2800.HK", "sub_sector": "工業與基建"},
}

SIGNAL_PRIORITY = {
    "buy_2a": 1,
    "watch": 2,
    "exit_50": 3,
    "trim_2b": 4,
    "short_4a": 5,
    "hold_2": 6,
    "stage_3": 7,
    "neutral": 8,
    "avoid_4": 9,
}


def normalize_hk_ticker(raw_code: str) -> str:
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if not digits:
        return ""
    if len(digits) == 5 and digits.startswith("0"):
        digits = digits[1:]
    else:
        digits = digits.zfill(4)
    return f"{digits}.HK"


def load_local_ggt_file(filepath: str = "ggt_list.txt") -> dict:
    """從倉庫根目錄的 ggt_list.txt 讀取自訂全市場港股通名單"""
    loaded = {}
    if not os.path.exists(filepath):
        return loaded

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,\t]+", line, maxsplit=1)
            yf_ticker = normalize_hk_ticker(parts[0])
            if not yf_ticker or len(yf_ticker) > 8:
                continue
            stock_name = parts[1].strip() if len(parts) > 1 else ""
            loaded[yf_ticker] = stock_name
    return loaded


def build_ggt_universe() -> tuple[dict, str]:
    """組合港股通股票池：優先讀取 ggt_list.txt，其次嘗試線上接口，最後以核心池保底"""
    universe = {k: dict(v) for k, v in CORE_GGT_POOL.items()}

    # 1. 優先讀取倉庫內的 ggt_list.txt（最穩定，不受海外 IP 封鎖影響）
    local_map = load_local_ggt_file("ggt_list.txt")
    if local_map:
        for yf_ticker, stock_name in local_map.items():
            if yf_ticker in universe:
                if stock_name:
                    universe[yf_ticker]["name"] = stock_name
            else:
                universe[yf_ticker] = {
                    "name": stock_name or yf_ticker,
                    "sector": "3033.HK",
                    "sub_sector": "港股通成份股",
                }
        return universe, f"全市場港股通名單 ({len(universe)} 隻)"

    # 2. 若無 ggt_list.txt，嘗試透過 akshare 線上同步
    if ak is not None:
        try:
            ggt_df = ak.stock_hk_ggt_components_em()
            if ggt_df is not None and not ggt_df.empty:
                code_col = "代碼" if "代碼" in ggt_df.columns else ggt_df.columns[1]
                name_col = "名稱" if "名稱" in ggt_df.columns else ggt_df.columns[2]
                for _, row in ggt_df.iterrows():
                    yf_ticker = normalize_hk_ticker(row[code_col])
                    if not yf_ticker or len(yf_ticker) > 8:
                        continue
                    stock_name = str(row[name_col]).strip()
                    if yf_ticker in universe:
                        if stock_name:
                            universe[yf_ticker]["name"] = stock_name
                    else:
                        universe[yf_ticker] = {
                            "name": stock_name or yf_ticker,
                            "sector": "3033.HK",
                            "sub_sector": "港股通成份股",
                        }
                return universe, f"全市場港股通動態名單 ({len(universe)} 隻)"
        except Exception as e:
            print(f"[GGT Warn] 線上接口超時，改用核心觀察池 ({len(universe)} 隻): {e}")

    return universe, f"港股通核心觀察池 ({len(universe)} 隻)"


def fetch_southbound_flows() -> dict:
    flow_map = {}
    if ak is None:
        return flow_map
    try:
        stats_df = ak.stock_hsgt_stock_statistics_em(symbol="南向持股")
        if stats_df is not None and not stats_df.empty:
            code_col = "代碼" if "代碼" in stats_df.columns else None
            chg_col = next((c for c in stats_df.columns if "持股市值變化" in c or "持股數量變化" in c), None)
            if code_col and chg_col:
                for _, row in stats_df.iterrows():
                    tk = normalize_hk_ticker(row[code_col])
                    val = pd.to_numeric(row[chg_col], errors="coerce")
                    if tk and pd.notna(val):
                        if val > 0:
                            flow_map[tk] = "北水增持"
                        elif val < 0:
                            flow_map[tk] = "北水減持"
    except Exception:
        pass
    return flow_map


def clean_hk_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return pd.DataFrame()
    latest_idx = df.index[-1]
    df = df[(df["Volume"] > 0) | (df.index == latest_idx)]
    return df


def fetch_all_histories(symbols: list[str], chunk_size: int = 50) -> dict:
    """分批高效下載全市場歷史日線"""
    history_map = {}
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        try:
            raw = yf.download(
                chunk,
                period="14mo",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            for sym in chunk:
                try:
                    sub_df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                    cleaned = clean_hk_df(sub_df)
                    if not cleaned.empty:
                        history_map[sym] = cleaned
                except Exception:
                    pass
        except Exception as e:
            print(f"[Batch Warn] 批次 {i // chunk_size + 1} 下載異常: {e}")
        time.sleep(0.25)

    # 針對核心觀察池中的遺漏標的進行單隻補抓
    missing_core = [s for s in CORE_GGT_POOL.keys() if s in symbols and s not in history_map]
    for sym in missing_core:
        try:
            df_single = yf.Ticker(sym).history(period="14mo", interval="1d", auto_adjust=False)
            cleaned = clean_hk_df(df_single)
            if not cleaned.empty:
                history_map[sym] = cleaned
        except Exception:
            pass

    return history_map


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["VOL20"] = df["Volume"].shift(1).rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL20"]
    df["BASE_HIGH_60"] = df["High"].shift(1).rolling(60).max()
    df["BASE_LOW_60"] = df["Low"].shift(1).rolling(60).min()
    df["GAP_UP"] = df["Low"] > df["High"].shift(1)
    return df


def classify_stock(
    ticker: str,
    meta: dict,
    df: pd.DataFrame,
    sector_strong: bool,
    southbound_status: str,
) -> dict:
    sector_etf = meta["sector"]
    sub_sector = meta["sub_sector"]

    if df.empty or len(df) < 210:
        close_val = round(float(df["Close"].iloc[-1]), 2) if not df.empty else 0.0
        return {
            "ticker": ticker,
            "name": meta["name"],
            "is_ggt": True,
            "southbound_flow": southbound_status,
            "sector_etf": sector_etf,
            "sector_name": SECTOR_ETFS.get(sector_etf, sector_etf),
            "sub_sector": sub_sector,
            "sector_strong": sector_strong,
            "close": close_val,
            "change_pct": 0.0,
            "sma50": 0.0,
            "sma150": 0.0,
            "sma200": 0.0,
            "dist_50": 0.0,
            "dist_200": 0.0,
            "vol_ratio": 0.0,
            "base_high": 0.0,
            "base_low": 0.0,
            "unfilled_gap": None,
            "stage": "上市未滿200日 / 數據不足",
            "signal_type": "neutral",
            "priority": SIGNAL_PRIORITY["neutral"],
            "action_note": f"有效交易日 {len(df)} 日，暫未形成 200 日均線基準",
        }

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

    recent_5 = df.iloc[-5:]
    unfilled_gap_level = None
    for i in range(len(recent_5)):
        row = recent_5.iloc[i]
        if row["GAP_UP"]:
            prev_high = float(df.iloc[-5 + i - 1]["High"])
            subsequent_low = float(recent_5.iloc[i:]["Low"].min())
            if subsequent_low > prev_high:
                unfilled_gap_level = round(prev_high, 2)

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
            f"跌破 200 日線轉弱；防守參考：站回 150 日線 (HK${sma150:.2f}) 減半，"
            f"站回 50 日線 (HK${sma50:.2f}) 解除"
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
        "is_ggt": True,
        "southbound_flow": southbound_status,
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
        "priority": SIGNAL_PRIORITY.get(signal_type, 99),
        "action_note": action_note,
    }


def main():
    ggt_universe, universe_source = build_ggt_universe()
    southbound_flows = fetch_southbound_flows()

    all_symbols = list(dict.fromkeys([BENCHMARK] + list(SECTOR_ETFS.keys()) + list(ggt_universe.keys())))
    raw_histories = fetch_all_histories(all_symbols, chunk_size=50)

    processed = {}
    for sym, df in raw_histories.items():
        if len(df) >= 210:
            processed[sym] = compute_indicators(df)
        else:
            processed[sym] = df

    if BENCHMARK not in processed or len(processed[BENCHMARK]) < 210:
        raise RuntimeError(f"無法取得大盤基準 {BENCHMARK} 完整數據，終止掃描。")

    bench_df = processed[BENCHMARK]
    bench_latest = bench_df.iloc[-1]
    bench_ret_20d = (bench_latest["Close"] / bench_df["Close"].iloc[-20]) - 1
    market_above_200 = bool(bench_latest["Close"] > bench_latest["SMA200"])
    market_above_50 = bool(bench_latest["Close"] > bench_latest["SMA50"])

    sectors_summary = []
    sector_strength_map = {}
    for etf, label in SECTOR_ETFS.items():
        if etf in processed and len(processed[etf]) >= 210:
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

    results = []
    for ticker, meta in ggt_universe.items():
        df_item = processed.get(ticker, pd.DataFrame())
        sb_status = southbound_flows.get(ticker, "港股通合資格")
        item = classify_stock(
            ticker,
            meta,
            df_item,
            sector_strength_map.get(meta["sector"], False),
            sb_status,
        )
        results.append(item)

    results.sort(key=lambda x: (x["priority"], -x["vol_ratio"]))

    valid_stocks = [r for r in results if r["sma200"] > 0]
    total_valid = max(len(valid_stocks), 1)
    stage2a_count = sum(1 for r in valid_stocks if r["signal_type"] in ("buy_2a", "watch"))
    stage2_total = sum(1 for r in valid_stocks if r["signal_type"] in ("buy_2a", "watch", "hold_2", "trim_2b"))
    stage4_total = sum(1 for r in valid_stocks if r["signal_type"] in ("short_4a", "avoid_4"))
    exit50_count = sum(1 for r in valid_stocks if r["signal_type"] == "exit_50")

    hkt = timezone(timedelta(hours=8))
    payload = {
        "updated_at": datetime.now(hkt).strftime("%Y-%m-%d %H:%M HKT"),
        "universe_source": universe_source,
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
        "breadth": {
            "total": len(valid_stocks),
            "stage2a_count": stage2a_count,
            "stage2_count": stage2_total,
            "stage4_count": stage4_total,
            "exit50_count": exit50_count,
            "stage2_pct": round(stage2_total / total_valid * 100, 1),
            "stage4_pct": round(stage4_total / total_valid * 100, 1),
        },
        "stocks": results,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/scan_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"GGT Scan complete: {len(results)} stocks saved ({universe_source})")


if __name__ == "__main__":
    main()
