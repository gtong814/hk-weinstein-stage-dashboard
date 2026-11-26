import time
from typing import Any, Dict, List, Tuple
import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf

from scanner_engine import analyze_single_stock, export_dashboard_json


# 備援核心港股通股票池（當外部成分股接口受海外 IP 限流時自動啟用，確保任務不中斷）
FALLBACK_UNIVERSE: List[Dict[str, str]] = [
    {"symbol": "0700.HK", "name": "騰訊控股", "sector": "互聯網科技"},
    {"symbol": "9988.HK", "name": "阿里巴巴-W", "sector": "電商 / 雲計算"},
    {"symbol": "3690.HK", "name": "美團-W", "sector": "本地生活服務"},
    {"symbol": "1810.HK", "name": "小米集團-W", "sector": "消費電子 / 智能汽車"},
    {"symbol": "9660.HK", "name": "地平線機器人-W", "sector": "智能駕駛 / AI芯片"},
    {"symbol": "0020.HK", "name": "商湯-W", "sector": "人工智能大模型"},
    {"symbol": "9880.HK", "name": "優必選", "sector": "人形機器人"},
    {"symbol": "2533.HK", "name": "黑芝麻智能", "sector": "自動駕駛芯片"},
    {"symbol": "0981.HK", "name": "中芯國際", "sector": "半導體製造"},
    {"symbol": "1347.HK", "name": "華虹半導體", "sector": "半導體製造"},
    {"symbol": "9698.HK", "name": "萬國數據-SW", "sector": "AI數據中心"},
    {"symbol": "1686.HK", "name": "新意網集團", "sector": "數據中心基建"},
    {"symbol": "0697.HK", "name": "首程控股", "sector": "機器人產業 / 基建"},
    {"symbol": "1833.HK", "name": "平安好醫生", "sector": "互聯網醫療"},
    {"symbol": "3396.HK", "name": "聯想控股", "sector": "科技投資"},
    {"symbol": "0992.HK", "name": "聯想集團", "sector": "AI PC / 伺服器"},
    {"symbol": "1211.HK", "name": "比亞迪股份", "sector": "新能源汽車"},
    {"symbol": "9868.HK", "name": "小鵬汽車-W", "sector": "智能電動車"},
    {"symbol": "2015.HK", "name": "理想汽車-W", "sector": "智能電動車"},
    {"symbol": "0388.HK", "name": "香港交易所", "sector": "金融交易所"},
    {"symbol": "0941.HK", "name": "中國移動", "sector": "電信營運商"},
    {"symbol": "0883.HK", "name": "中國海洋石油", "sector": "能源油氣"},
    {"symbol": "1658.HK", "name": "郵儲銀行", "sector": "內銀高息"},
    {"symbol": "0939.HK", "name": "建設銀行", "sector": "內銀高息"},
    {"symbol": "1024.HK", "name": "快手-W", "sector": "短視頻 / AI應用"},
    {"symbol": "9618.HK", "name": "京東集團-SW", "sector": "電商物流"},
    {"symbol": "9961.HK", "name": "攜程集團-S", "sector": "在線旅遊"},
    {"symbol": "2382.HK", "name": "舜宇光學科技", "sector": "車載與手機光學"},
    {"symbol": "0285.HK", "name": "比亞迪電子", "sector": "消費電子 / AI伺服器"},
    {"symbol": "2276.HK", "name": "康耐特光學", "sector": "智能眼鏡光學"},
]


def normalize_hk_symbol(raw_code: Any) -> str:
    """
    將 AKShare 的 5 位港股代碼（如 '00700', '09660'）統一轉換為 Yahoo Finance 4 位格式（'0700.HK', '9660.HK'）。
    """
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if not digits:
        return ""
    short_code = digits.lstrip("0").zfill(4)
    return f"{short_code}.HK"


def fetch_hk_connect_universe() -> List[Dict[str, str]]:
    """
    動態拉取最新港股通成分股名單；若上游接口波動則平滑切換至備援清單。
    """
    try:
        print("正在拉取港股通成分股名單...")
        df_ggt = ak.stock_hk_ggt_components_em()
        if df_ggt is None or df_ggt.empty:
            raise ValueError("港股通成分股返回空表")

        code_col = next((c for c in df_ggt.columns if "代碼" in c or "代码" in c), None)
        name_col = next((c for c in df_ggt.columns if "名稱" in c or "名称" in c), None)
        sector_col = next((c for c in df_ggt.columns if "行業" in c or "行业" in c or "板塊" in c), None)

        if not code_col or not name_col:
            raise ValueError(f"無法識別成分股欄位: {list(df_ggt.columns)}")

        universe = []
        seen = set()
        for _, row in df_ggt.iterrows():
            sym = normalize_hk_symbol(row[code_col])
            if not sym or sym in seen:
                continue
            seen.add(sym)
            universe.append(
                {
                    "symbol": sym,
                    "name": str(row[name_col]).strip(),
                    "sector": str(row[sector_col]).strip() if sector_col and pd.notna(row[sector_col]) else "港股通標的",
                }
            )

        if len(universe) >= 50:
            print(f"成功獲取 {len(universe)} 隻港股通標的。")
            return universe
    except Exception as exc:
        print(f"[警告] 動態獲取港股通名單失敗 ({exc})，改用內置核心備援清單（{len(FALLBACK_UNIVERSE)} 隻）。")

    return FALLBACK_UNIVERSE


def fetch_northbound_3d_flow_map() -> Dict[str, float]:
    """
    嘗試獲取港股通（南向資金）近 3 日增持/淨買入估算金額映射表（單位：港元/人民幣）。
    因外部數據源接口格式可能隨版本變動，失敗時返回空字典並由主程式平滑處理。
    """
    flow_map: Dict[str, float] = {}
    try:
        print("正在拉取港股通（北水）近 3 日資金流向數據...")
        df_nb = ak.stock_hsgt_hold_stock_em(market="港股通", indicator="3日排行")
        if df_nb is None or df_nb.empty:
            return flow_map

        code_col = next((c for c in df_nb.columns if "代碼" in c or "代码" in c), None)
        # 優先尋找「3日增持估計市值」或包含「增持」/「淨買」的數值欄位
        val_col = next(
            (c for c in df_nb.columns if ("3日" in c and ("增持" in c or "淨買" in c or "净买" in c))),
            None,
        )
        if not val_col:
            val_col = next((c for c in df_nb.columns if "增持" in c or "淨買" in c or "净买" in c), None)

        if code_col and val_col:
            for _, row in df_nb.iterrows():
                sym = normalize_hk_symbol(row[code_col])
                raw_val = pd.to_numeric(row[val_col], errors="coerce")
                if sym and pd.notna(raw_val):
                    # 部分排行欄位單位為「萬」，若絕對值較小可按需還原；此處保留原始比例供方向判定
                    flow_map[sym] = float(raw_val)
            print(f"成功匹配 {len(flow_map)} 隻標的的北水 3 日流向數據。")
    except Exception as exc:
        print(f"[提示] 北水排行接口暫不可用 ({exc})，本次掃描將跳過北水加權。")

    return flow_map


def download_yf_adjusted_ohlcv(
    tickers: List[str], period: str = "15mo", batch_size: int = 45
) -> Dict[str, pd.DataFrame]:
    """
    使用 yfinance 批量下載「前復權（auto_adjust=True）」日線 OHLCV 數據。
    """
    ohlcv_dict: Dict[str, pd.DataFrame] = {}
    unique_tickers = list(dict.fromkeys(tickers))

    for i in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[i : i + batch_size]
        print(f"正在下載前復權日線批次 {i // batch_size + 1}/{(len(unique_tickers) - 1) // batch_size + 1} ({len(batch)} 隻)...")
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=True,  # 關鍵：使用前復權價格，避免港股除淨跳空引發假破線警報
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if raw is None or raw.empty:
                continue

            if len(batch) == 1:
                tk = batch[0]
                sub = raw.copy()
                if isinstance(sub.columns, pd.MultiIndex):
                    sub.columns = sub.columns.get_level_values(0)
                sub = _clean_single_df(sub)
                if not sub.empty:
                    ohlcv_dict[tk] = sub
            else:
                for tk in batch:
                    if tk not in raw.columns.get_level_values(0):
                        continue
                    sub = _clean_single_df(raw[tk].copy())
                    if not sub.empty:
                        ohlcv_dict[tk] = sub
        except Exception as exc:
            print(f"[警告] 批次下載異常 ({exc})，繼續下一批次...")

        time.sleep(0.8)  # 禮貌性間隔，避免觸發 Yahoo Finance 速率限制

    return ohlcv_dict


def _clean_single_df(df: pd.DataFrame) -> pd.DataFrame:
    """標準化單隻股票日線欄位與時間索引"""
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    cleaned = df[required_cols].dropna(subset=["Close", "Volume"]).copy()
    cleaned = cleaned[cleaned["Volume"] > 0]
    if cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_localize(None)
    cleaned.sort_index(inplace=True)
    return cleaned


def build_macro_summary(
    df_2800: pd.DataFrame, df_3033: pd.DataFrame
) -> Tuple[Dict[str, Any], bool]:
    """
    計算頂部兩張大盤基準卡（2800.HK 趨勢階段 & 3033.HK 相對強度），並返回大盤是否處於多頭。
    """
    # 1. 評估 2800.HK（盈富基金）
    d1 = df_2800.copy()
    d1["MA50"] = d1["Close"].rolling(50).mean()
    d1["MA200"] = d1["Close"].rolling(200).mean()
    d1["Pct"] = d1["Close"].pct_change() * 100
    c1 = d1.iloc[-1]

    px_2800 = float(c1["Close"])
    pct_2800 = float(c1["Pct"])
    hsi_price_str = f"{px_2800:.2f} ({pct_2800:+.2f}%)"

    if c1["Close"] > c1["MA50"] and c1["MA50"] > c1["MA200"]:
        is_market_bullish = True
        hsi_stage = "📈 Stage 2 多頭"
        hsi_advice = "2800.HK 穩守 50 日及 200 日線上方，大盤處於多頭環境，可執行標準倉位進攻策略。"
    elif c1["Close"] >= c1["MA50"] and c1["Close"] > c1["MA200"]:
        is_market_bullish = True
        hsi_stage = "🚀 Stage 2A 轉強"
        hsi_advice = "2800.HK 站上 50 日與 200 日均線，趨勢初步轉強，可精選雙因子共振標的試探建倉。"
    elif c1["Close"] < c1["MA50"] and c1["Close"] >= c1["MA200"]:
        is_market_bullish = False
        hsi_stage = "⚠️ 跌破50日線（防守）"
        hsi_advice = "2800.HK 失守 50 日線，大盤進入震盪調整，個股一旦初破 50 日線應立即收緊止盈。"
    else:
        is_market_bullish = False
        hsi_stage = "🔻 Stage 4 空頭警戒"
        hsi_advice = "2800.HK 位於 50 日及 200 日線下方，大盤處於空頭防守期，宜大幅縮減倉位並嚴控風險。"

    # 2. 評估 3033.HK（南方恒生科技）對比 2800.HK 相對強度
    d2 = df_3033.copy()
    d2["Pct"] = d2["Close"].pct_change() * 100
    joined = d2[["Close", "Pct"]].join(
        d1[["Close"]].rename(columns={"Close": "Bench_Close"}), how="inner"
    )
    joined["RS"] = joined["Close"] / joined["Bench_Close"]
    joined["RS_MA50"] = joined["RS"].rolling(50).mean()
    c2 = joined.iloc[-1]

    px_3033 = float(c2["Close"])
    pct_3033 = float(c2["Pct"])
    hstech_price_str = f"{px_3033:.2f} ({pct_3033:+.2f}%)"

    if c2["RS"] > c2["RS_MA50"]:
        hstech_rs = "RS 站穩 50D 均線走強"
        hstech_advice = "3033.HK 跑贏盈富基金，資金偏好高彈性科技與成長賽道，選股可優先側重科技股。"
    else:
        hstech_rs = "RS 弱於 50D 均線（防禦）"
        hstech_advice = "3033.HK 相對大盤走弱，高彈性科技股承壓，選股宜偏向高息藍籌或縮減進攻比例。"

    macro_summary = {
        "hsi_price": hsi_price_str,
        "hsi_stage": hsi_stage,
        "hsi_advice": hsi_advice,
        "hstech_price": hstech_price_str,
        "hstech_rs": hstech_rs,
        "hstech_advice": hstech_advice,
    }
    return macro_summary, is_market_bullish


def main() -> None:
    # 1. 獲取港股通標的清單與北水資金流向
    universe = fetch_hk_connect_universe()
    nb_flow_map = fetch_northbound_3d_flow_map()

    # 2. 批量拉取基準 ETF（2800.HK, 3033.HK）及全部港股通前復權日線
    all_tickers = ["2800.HK", "3033.HK"] + [item["symbol"] for item in universe]
    ohlcv_map = download_yf_adjusted_ohlcv(all_tickers, period="15mo", batch_size=45)

    if "2800.HK" not in ohlcv_map or "3033.HK" not in ohlcv_map:
        raise RuntimeError("無法獲取基準 ETF（2800.HK 或 3033.HK）日線數據，終止本次更新。")

    df_2800 = ohlcv_map["2800.HK"]
    df_3033 = ohlcv_map["3033.HK"]

    # 3. 評估大盤基準與板塊相對強度
    macro_summary, is_market_bullish = build_macro_summary(df_2800, df_3033)
    print(f"大盤基準判定: {macro_summary['hsi_stage']} (is_market_bullish={is_market_bullish})")

    # 4. 逐一分析港股通個股
    results: List[Dict[str, Any]] = []
    for item in universe:
        sym = item["symbol"]
        if sym not in ohlcv_map:
            continue

        stock_df = ohlcv_map[sym].copy()

        # 將 3 日北水淨流向對齊寫入最近 3 個交易日（使 tail(3).sum() 與 analyze_single_stock 邏輯完全一致）
        nb_3d_total = nb_flow_map.get(sym, 0.0)
        stock_df["Northbound_Net_Buy"] = 0.0
        if len(stock_df) >= 3 and nb_3d_total != 0.0:
            stock_df.iloc[-3:, stock_df.columns.get_loc("Northbound_Net_Buy")] = nb_3d_total / 3.0

        record = analyze_single_stock(
            symbol=sym,
            name=item["name"],
            sector=item["sector"],
            df=stock_df,
            bench_2800_df=df_2800,
            is_market_bullish=is_market_bullish,
        )
        if record:
            results.append(record)

    # 5. 按實戰優先級排序（黃金共振 Stage 2A 突破排最前，其餘按階段與 20 日量比降序排列）
    stage_priority = {
        "stage2a": 1,
        "stage2": 2,
        "warning": 3,
        "neutral": 4,
        "stage4": 5,
    }
    results.sort(
        key=lambda x: (
            stage_priority.get(x.get("stage_code", "neutral"), 9),
            0 if x.get("is_golden_resonance") else 1,
            -float(x.get("vol_ratio", 0.0)),
        )
    )

    # 6. 輸出 data.json 供前端儀表板渲染
    export_dashboard_json(macro_summary, results, output_path="data.json")
    print(f"掃描完成！共分析 {len(results)} 隻有效港股通標的，已寫入 data.json。")


if __name__ == "__main__":
    main()
