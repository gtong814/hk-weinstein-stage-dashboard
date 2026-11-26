import html
import json
import math
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
import yfinance as yf


UNIVERSE_CACHE_FILE = "hk_connect_universe.json"

# 僅在首次運行且所有外部網絡均斷開時才啟用的緊急種子清單
EMERGENCY_SEED_UNIVERSE: List[Dict[str, str]] = [
    {"symbol": "0700.HK", "name": "騰訊控股", "sector": "互聯網科技"},
    {"symbol": "9988.HK", "name": "阿里巴巴-W", "sector": "電商 / 雲計算"},
    {"symbol": "3690.HK", "name": "美團-W", "sector": "本地生活服務"},
    {"symbol": "1810.HK", "name": "小米集團-W", "sector": "消費電子 / 智能汽車"},
    {"symbol": "1816.HK", "name": "中廣核電力", "sector": "核電 / 清潔能源"},
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
    {"symbol": "1658.HK", "name": "郵儲銀行", "sector": "內銀高息"},
    {"symbol": "0941.HK", "name": "中國移動", "sector": "電信營運商"},
    {"symbol": "0883.HK", "name": "中國海洋石油", "sector": "能源油氣"},
    {"symbol": "1211.HK", "name": "比亞迪股份", "sector": "新能源汽車"},
]


def normalize_hk_symbol(raw_code: Any) -> str:
    """將 5 位港股代碼（如 '01816', '00700'）統一轉為 Yahoo Finance 4 位格式（'1816.HK', '0700.HK'）"""
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if not digits:
        return ""
    return f"{digits.lstrip('0').zfill(4)}.HK"


def sanitize_for_json(obj: Any) -> Any:
    """遞歸清理 NaN / Inf 與 NumPy 類型，確保輸出合法標準 JSON"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return 0.0 if (math.isnan(val) or math.isinf(val)) else val
    return obj


def _strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = html.unescape(clean)
    for prefix in [
        "Stock Code:",
        "Stock Name:",
        "Shareholding in CCASS:",
        "% of the total number of Issued Shares:",
    ]:
        clean = clean.replace(prefix, "")
    return " ".join(clean.split()).strip()


def _parse_hkex_mutualmarket_html(page_html: str) -> Dict[str, Dict[str, Any]]:
    """解析港交所官方 Mutual Market (Southbound) 持股頁面表格"""
    records: Dict[str, Dict[str, Any]] = {}
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, flags=re.DOTALL | re.IGNORECASE)
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL | re.IGNORECASE)
        if len(cells) < 3:
            continue
        code_txt = _strip_html(cells[0])
        name_txt = _strip_html(cells[1])
        shares_txt = _strip_html(cells[2]).replace(",", "")

        if not code_txt.isdigit():
            continue
        sym = normalize_hk_symbol(code_txt)
        try:
            shares_val = int(float(shares_txt))
        except ValueError:
            shares_val = 0

        if sym:
            records[sym] = {
                "symbol": sym,
                "name": name_txt,
                "shares": shares_val,
            }
    return records


def _extract_input_value(page_html: str, field_name: str) -> str:
    """兼容不同屬性順序，提取 ASP.NET 表單欄位 value"""
    m = re.search(
        rf'<input[^>]*?(?:id|name)="{re.escape(field_name)}"[^>]*?value="([^"]*)"',
        page_html,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)
    m_rev = re.search(
        rf'<input[^>]*?value="([^"]*)"[^>]*?(?:id|name)="{re.escape(field_name)}"',
        page_html,
        flags=re.IGNORECASE,
    )
    return m_rev.group(1) if m_rev else ""


def fetch_hkex_southbound_universe_and_holdings() -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    直接從港交所官方（HKEXnews）拉取：
    1. 全部 500+ 隻港股通合資格成分股名單（不受海外 IP 限制）
    2. 近 3 個交易日的南向（北水）持股股數變動（shares_delta_3d）
    """
    url = "https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8",
    }
    session = requests.Session()
    universe: List[Dict[str, str]] = []
    shares_delta_map: Dict[str, int] = {}

    try:
        print("正在連接港交所官方（HKEXnews）拉取完整港股通成分股與北水持股...")
        resp_today = session.get(url, headers=headers, timeout=25)
        resp_today.raise_for_status()
        html_today = resp_today.text

        today_map = _parse_hkex_mutualmarket_html(html_today)
        if len(today_map) < 300:
            raise ValueError(f"HKEXnews 解析筆數過少 ({len(today_map)})")

        for sym, info in today_map.items():
            universe.append({
                "symbol": sym,
                "name": info["name"],
                "sector": "港股通標的",
            })

        vs_val = _extract_input_value(html_today, "__VIEWSTATE")
        vsg_val = _extract_input_value(html_today, "__VIEWSTATEGENERATOR")
        ev_val = _extract_input_value(html_today, "__EVENTVALIDATION")
        date_str = _extract_input_value(html_today, "txtShareholdingDate")

        if vs_val and re.match(r"^\d{4}/\d{2}/\d{2}$", date_str):
            latest_dt = datetime.strptime(date_str, "%Y/%m/%d")
            prev_dt = latest_dt - timedelta(days=4)
            if prev_dt.weekday() == 5:
                prev_dt -= timedelta(days=1)
            elif prev_dt.weekday() == 6:
                prev_dt -= timedelta(days=2)

            payload = {
                "__VIEWSTATE": vs_val,
                "__VIEWSTATEGENERATOR": vsg_val,
                "__EVENTVALIDATION": ev_val,
                "today": latest_dt.strftime("%Y%m%d"),
                "sortBy": "stockcode",
                "sortDirection": "asc",
                "alertMsg": "",
                "txtShareholdingDate": prev_dt.strftime("%Y/%m/%d"),
                "btnSearch": "Search",
            }
            resp_prev = session.post(url, data=payload, headers=headers, timeout=25)
            if resp_prev.ok:
                prev_map = _parse_hkex_mutualmarket_html(resp_prev.text)
                if len(prev_map) >= 300:
                    for sym, info in today_map.items():
                        prev_shares = prev_map.get(sym, {}).get("shares", info["shares"])
                        shares_delta_map[sym] = info["shares"] - prev_shares
                    print(f"成功對比港交所 {prev_dt.strftime('%Y-%m-%d')} 至 {latest_dt.strftime('%Y-%m-%d')} 北水持股變動。")

        print(f"成功從港交所官方獲取 {len(universe)} 隻港股通成分股！")
    except Exception as exc:
        print(f"[警告] HKEXnews 拉取異常: {exc}")

    return universe, shares_delta_map


def fetch_sina_hk_connect_names() -> Dict[str, str]:
    """從新浪財經港股通節點（ggt_sh / ggt_sz）補充中文股票名稱"""
    cn_name_map: Dict[str, str] = {}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://vip.stock.finance.sina.com.cn/"}
    for node in ["ggt_sh", "ggt_sz"]:
        for page in range(1, 8):
            api_url = (
                f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                f"Market_Center.getHKStockData?page={page}&num=100&sort=symbol&asc=1&node={node}"
            )
            try:
                r = requests.get(api_url, headers=headers, timeout=12)
                if not r.ok or len(r.text.strip()) < 10:
                    break
                items = r.json()
                if not items:
                    break
                for it in items:
                    sym = normalize_hk_symbol(it.get("symbol", ""))
                    name = str(it.get("name", "")).strip()
                    if sym and name:
                        cn_name_map[sym] = name
                if len(items) < 100:
                    break
            except Exception:
                break
    if cn_name_map:
        print(f"成功從新浪財經匹配 {len(cn_name_map)} 隻港股通中文名稱。")
    return cn_name_map


def get_full_hk_connect_universe() -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """三層容錯獲取完整 500+ 港股通名單並持久化快取"""
    universe, shares_delta_map = fetch_hkex_southbound_universe_and_holdings()
    cn_names = fetch_sina_hk_connect_names()

    if len(universe) >= 300:
        for item in universe:
            if item["symbol"] in cn_names:
                item["name"] = cn_names[item["symbol"]]
        with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(universe, f, ensure_ascii=False, indent=2)
        return universe, shares_delta_map

    if len(cn_names) >= 300:
        universe = [
            {"symbol": sym, "name": name, "sector": "港股通標的"}
            for sym, name in sorted(cn_names.items())
        ]
        with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(universe, f, ensure_ascii=False, indent=2)
        return universe, shares_delta_map

    if os.path.exists(UNIVERSE_CACHE_FILE):
        try:
            with open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached) >= 100:
                print(f"已從本地快取 {UNIVERSE_CACHE_FILE} 載入 {len(cached)} 隻港股通標的。")
                return cached, shares_delta_map
        except Exception as exc:
            print(f"[警告] 讀取快取失敗: {exc}")

    print("[警告] 使用緊急種子清單。")
    return EMERGENCY_SEED_UNIVERSE, shares_delta_map


def analyze_single_stock(
    symbol: str,
    name: str,
    sector: str,
    df: pd.DataFrame,
    bench_2800_df: pd.DataFrame,
    is_market_bullish: bool = True,
) -> Dict[str, Any]:
    """
    對單隻港股通標的執行 Stan Weinstein 現代版四階段分析、大升浪共振評級與分級止盈判定。
    """
    if len(df) < 60 or len(bench_2800_df) < 200:
        return {}

    data = df.copy()
    bench = bench_2800_df[["Close"]].rename(columns={"Close": "Bench_Close"})
    data = data.join(bench, how="inner")
    if len(data) < 60:
        return {}

    # 1. 核心均線與斜率（兼容上市滿 60-199 日的次新港股通標的）
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200, min_periods=100).mean()
    data["MA50_Slope"] = data["MA50"] - data["MA50"].shift(5)
    data["MA200_Slope_Pct"] = np.where(
        data["MA200"].shift(20).notna() & (data["MA200"].shift(20) > 0),
        (data["MA200"] - data["MA200"].shift(20)) / data["MA200"].shift(20) * 100,
        0.0,
    )

    # 2. 乖離率與 20 日量比
    data["Bias_50D"] = np.where(
        data["MA50"] > 0, (data["Close"] - data["MA50"]) / data["MA50"] * 100, 0.0
    )
    data["Bias_200D"] = np.where(
        data["MA200"].notna() & (data["MA200"] > 0),
        (data["Close"] - data["MA200"]) / data["MA200"] * 100,
        0.0,
    )
    data["Vol_MA20"] = data["Volume"].rolling(20).mean()
    data["Vol_Ratio_20D"] = np.where(
        data["Vol_MA20"] > 0, data["Volume"] / data["Vol_MA20"], 0.0
    )

    # 3. 相對強度（RS vs 2800.HK）
    data["RS"] = data["Close"] / data["Bench_Close"]
    data["RS_MA50"] = data["RS"].rolling(50).mean()
    data["RS_Prev_High_40"] = data["RS"].shift(1).rolling(40).max()

    # 4. K 線結構與區間高低點
    data["Pct_Change"] = data["Close"].pct_change().fillna(0.0) * 100
    body = (data["Close"] - data["Open"]).abs()
    upper_shadow = data["High"] - np.maximum(data["Open"], data["Close"])
    day_range = data["High"] - data["Low"]

    data["Close_Pos"] = np.where(day_range > 0, (data["Close"] - data["Low"]) / day_range, 0.5)
    data["Upper_Shadow_Ratio"] = np.where(
        body > 0, upper_shadow / body, np.where(upper_shadow > 0, 99.0, 0.0)
    )
    data["Prev_High_40"] = data["High"].shift(1).rolling(40).max()
    data["Prev_Low_60"] = data["Low"].shift(1).rolling(60, min_periods=40).min()
    data["Max_Bias_5D"] = data["Bias_50D"].rolling(5).max()

    # 5. 未補向上跳空缺口檢測（近 15 日內）
    gap_info = "無"
    start_idx = max(1, len(data) - 15)
    for idx in range(start_idx, len(data)):
        if data["Low"].iloc[idx] > data["High"].iloc[idx - 1] * 1.005:
            gap_low = round(float(data["High"].iloc[idx - 1]), 2)
            gap_high = round(float(data["Low"].iloc[idx]), 2)
            if data["Low"].iloc[idx:].min() > gap_low:
                gap_info = f"HK${gap_low:.2f}-{gap_high:.2f}"

    # 6. 北水動向（近 3 日淨買賣估算金額，單位：港元）
    has_nb = "Northbound_Net_Buy" in data.columns
    nb_3d = float(data["Northbound_Net_Buy"].tail(3).sum()) if has_nb else 0.0
    if nb_3d > 5_000_000:
        nb_status = f"北水淨買 +{nb_3d / 1e6:.1f}M"
    elif nb_3d < -5_000_000:
        nb_status = f"北水淨賣 {nb_3d / 1e6:.1f}M"
    else:
        nb_status = "北水持平"

    curr = data.iloc[-1]
    prev = data.iloc[-2]

    close_px = round(float(curr["Close"]), 2)
    pct_chg = round(float(curr["Pct_Change"]), 2)
    bias_50 = round(float(curr["Bias_50D"]), 2)
    has_ma200 = pd.notna(curr["MA200"]) and float(curr["MA200"]) > 0
    bias_200 = round(float(curr["Bias_200D"]), 2) if has_ma200 else 0.0
    vol_ratio = round(float(curr["Vol_Ratio_20D"]), 2)
    ma50_val = round(float(curr["MA50"]), 2)
    ma200_slope_pct = round(float(curr["MA200_Slope_Pct"]), 2)

    # 200 日線斜率與 RS 共振定性
    ma200_up_or_flat = (not has_ma200) or (ma200_slope_pct >= -0.3)
    rs_leading = (curr["RS"] > curr["RS_MA50"]) and (curr["RS"] >= curr["RS_Prev_High_40"] * 0.98)
    rs_above_ma = curr["RS"] > curr["RS_MA50"]

    slope_desc = f"200D({ma200_slope_pct:+.1f}%)" if has_ma200 else "次新股(無200D)"
    if ma200_up_or_flat and rs_leading:
        resonance_tier = "黃金共振"
        slope_rs_label = f"{slope_desc} · RS領先創高"
    elif (not ma200_up_or_flat) and rs_leading:
        resonance_tier = "早期先鋒"
        slope_rs_label = f"{slope_desc} · RS強勢領先"
    elif ma200_up_or_flat and (not rs_above_ma):
        resonance_tier = "補漲落後"
        slope_rs_label = f"{slope_desc} · RS走平轉弱"
    else:
        resonance_tier = "弱勢結構"
        slope_rs_label = f"{slope_desc} · RS弱於大市"

    is_fake_breakout_risk = bool(
        (vol_ratio < 1.2)
        or (vol_ratio > 5.0 and curr["Upper_Shadow_Ratio"] > 1.8)
        or (bias_50 > 12.0)
        or (has_nb and nb_3d < -5_000_000)
    )
    is_golden_entry_zone = bool(
        (2.0 <= bias_50 <= 8.0) and (1.5 <= vol_ratio <= 3.5) and (not is_fake_breakout_risk)
    )

    tags: List[str] = []
    if gap_info != "無":
        tags.append("gap")
    if nb_3d > 5_000_000:
        tags.append("northbound")

    below_ma200 = (not has_ma200) or (curr["MA50"] < curr["MA200"]) or (curr["Close"] < curr["MA200"])

    # 1. Stage 4A / 空頭
    if (
        curr["Close"] < curr["MA50"]
        and curr["MA50_Slope"] < 0
        and pd.notna(curr["Prev_Low_60"])
        and curr["Close"] < curr["Prev_Low_60"]
        and below_ma200
    ):
        tags.append("stage4")
        return {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "nb_status": nb_status,
            "close": close_px,
            "pct_change": pct_chg,
            "stage": "🔻 Stage 4A / 空頭",
            "stage_code": "stage4",
            "resonance_tier": resonance_tier,
            "slope_rs_label": slope_rs_label,
            "bias_50d": bias_50,
            "bias_200d": bias_200,
            "vol_ratio": vol_ratio,
            "gap": gap_info,
            "is_golden_resonance": False,
            "is_golden_entry": False,
            "is_fake_risk": True,
            "advice": "【Level 4 清倉 100%】50日線下彎並跌破近60日箱體底部，結構已破壞，宜離場觀望不抄底。",
            "tags": tags,
        }

    # 2. 右側破 50 日線分級防守
    if curr["Close"] < curr["MA50"]:
        tags.append("warning")
        cond_vol_break = (bias_50 < -2.5) and (vol_ratio > 1.5)
        cond_two_days = (curr["Close"] < curr["MA50"]) and (prev["Close"] < prev["MA50"])
        cond_slope_rs = (curr["MA50_Slope"] < 0) and (not rs_above_ma)

        reasons = []
        if cond_vol_break:
            reasons.append(f"放量擊穿50D線(乖離{bias_50}%/量比{vol_ratio}x)")
        if cond_two_days:
            reasons.append("連續2日收於50D線下方")
        if cond_slope_rs:
            reasons.append("50D線下彎且RS跌破均線")

        if reasons or (not is_market_bullish):
            reason_txt = "；".join(reasons) if reasons else "大盤基準轉弱且個股破50D線"
            return {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "nb_status": nb_status,
                "close": close_px,
                "pct_change": pct_chg,
                "stage": "⚠️ 破50日線（確認轉弱）",
                "stage_code": "warning",
                "resonance_tier": resonance_tier,
                "slope_rs_label": slope_rs_label,
                "bias_50d": bias_50,
                "bias_200d": bias_200,
                "vol_ratio": vol_ratio,
                "gap": gap_info,
                "is_golden_resonance": False,
                "is_golden_entry": False,
                "is_fake_risk": True,
                "advice": f"【Level 3 減倉 70%-75%】{reason_txt}。僅留底倉觀察箱體下緣，再破即全清。",
                "tags": tags,
            }
        return {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "nb_status": nb_status,
            "close": close_px,
            "pct_change": pct_chg,
            "stage": "⚠️ 初破50日線（觀察）",
            "stage_code": "warning",
            "resonance_tier": resonance_tier,
            "slope_rs_label": slope_rs_label,
            "bias_50d": bias_50,
            "bias_200d": bias_200,
            "vol_ratio": vol_ratio,
            "gap": gap_info,
            "is_golden_resonance": False,
            "is_golden_entry": False,
            "is_fake_risk": False,
            "advice": f"【Level 2 觀察2日 / 減倉0-20%】縮量微破50D線(HK${ma50_val})，若第3日未收復則執行減倉。",
            "tags": tags,
        }

    # 3. 左側 2B 止盈
    ma50_above_ma200 = (not has_ma200) or (curr["MA50"] > curr["MA200"])
    is_bull_extended = (
        ma50_above_ma200
        and curr["MA50_Slope"] > 0
        and curr["Max_Bias_5D"] >= 15.0
    )
    if is_bull_extended:
        cond_2b = (
            pd.notna(curr["Prev_High_40"])
            and curr["High"] > curr["Prev_High_40"]
            and curr["Close"] < curr["Prev_High_40"]
            and curr["Close_Pos"] < 0.35
        )
        cond_churn = (
            bias_50 > 18.0
            and vol_ratio > 2.0
            and (abs(curr["Pct_Change"]) <= 1.5 or curr["Upper_Shadow_Ratio"] > 2.0)
        )
        if cond_2b or cond_churn:
            tags.append("warning")
            trigger = "2B假突破前高回落" if cond_2b else f"50D乖離達{bias_50}%高位放量滯漲"
            nb_note = "且近3日北水轉淨賣出" if (has_nb and nb_3d < -5_000_000) else ""
            return {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "nb_status": nb_status,
                "close": close_px,
                "pct_change": pct_chg,
                "stage": "⚠️ 2B止盈（高位派發）",
                "stage_code": "warning",
                "resonance_tier": resonance_tier,
                "slope_rs_label": slope_rs_label,
                "bias_50d": bias_50,
                "bias_200d": bias_200,
                "vol_ratio": vol_ratio,
                "gap": gap_info,
                "is_golden_resonance": resonance_tier == "黃金共振",
                "is_golden_entry": False,
                "is_fake_risk": True,
                "advice": f"【Level 1 主動減倉 30%-50%】{trigger}{nb_note}。停止追高，止損上移至前波段低點。",
                "tags": tags,
            }

    # 4. Stage 2A 突破
    is_stage2a_breakout = (
        pd.notna(curr["Prev_High_40"])
        and curr["Close"] > curr["Prev_High_40"]
        and curr["Close"] > curr["MA50"]
        and curr["MA50_Slope"] >= 0
        and bias_50 <= 15.0
    )
    if is_stage2a_breakout:
        tags.extend(["stage2a", "stage2"])
        stop_loss = round(ma50_val * 0.985, 2)
        if is_golden_entry_zone and resonance_tier == "黃金共振":
            advice = f"【首選買點 · 標準倉 10-15%】乖離與量比均在黃金區且雙因子共振；止損參考 HK${stop_loss}（50D線下1.5%）。"
        elif is_fake_breakout_risk:
            advice = f"【謹慎防假突破】量比({vol_ratio}x)/乖離({bias_50}%)或北水未完全配合，建議等回踩頸線縮量企穩再部署。"
        else:
            pos_tip = "輕倉試探 5-7%" if resonance_tier == "早期先鋒" else "分批建倉"
            advice = f"【Stage 2A 突破 · {pos_tip}】結構定性為「{resonance_tier}」，止損設於 HK${stop_loss}。"

        return {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "nb_status": nb_status,
            "close": close_px,
            "pct_change": pct_chg,
            "stage": "🚀 Stage 2A 突破",
            "stage_code": "stage2a",
            "resonance_tier": resonance_tier,
            "slope_rs_label": slope_rs_label,
            "bias_50d": bias_50,
            "bias_200d": bias_200,
            "vol_ratio": vol_ratio,
            "gap": gap_info,
            "is_golden_resonance": resonance_tier == "黃金共振",
            "is_golden_entry": is_golden_entry_zone,
            "is_fake_risk": is_fake_breakout_risk,
            "advice": advice,
            "tags": tags,
        }

    # 5. Stage 2 多頭延續
    if curr["Close"] >= curr["MA50"] and ma50_above_ma200:
        tags.append("stage2")
        pullback_tip = (
            "正處縮量回踩50D線低吸觀察窗（量比低於0.8x）"
            if (0 <= bias_50 <= 3.5 and vol_ratio < 0.85)
            else f"沿上升趨勢持有，移動止損參考50D線 HK${ma50_val}"
        )
        return {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "nb_status": nb_status,
            "close": close_px,
            "pct_change": pct_chg,
            "stage": "📈 Stage 2 多頭",
            "stage_code": "stage2",
            "resonance_tier": resonance_tier,
            "slope_rs_label": slope_rs_label,
            "bias_50d": bias_50,
            "bias_200d": bias_200,
            "vol_ratio": vol_ratio,
            "gap": gap_info,
            "is_golden_resonance": resonance_tier == "黃金共振",
            "is_golden_entry": is_golden_entry_zone,
            "is_fake_risk": False,
            "advice": f"【多頭持有 · {resonance_tier}】{pullback_tip}。",
            "tags": tags,
        }

    # 6. Stage 1 / 3 震盪觀察
    return {
        "symbol": symbol,
        "name": name,
        "sector": sector,
        "nb_status": nb_status,
        "close": close_px,
        "pct_change": pct_chg,
        "stage": "⏸️ Stage 1 / 3 震盪",
        "stage_code": "neutral",
        "resonance_tier": resonance_tier,
        "slope_rs_label": slope_rs_label,
        "bias_50d": bias_50,
        "bias_200d": bias_200,
        "vol_ratio": vol_ratio,
        "gap": gap_info,
        "is_golden_resonance": False,
        "is_golden_entry": False,
        "is_fake_risk": False,
        "advice": "均線尚未理順多頭排列，繼續觀察底部放量突破訊號。",
        "tags": tags,
    }


def _clean_single_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    cleaned = df[required_cols].dropna(subset=["Close", "Volume"]).copy()
    cleaned = cleaned[cleaned["Volume"] > 0]
    if cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_localize(None)
    cleaned.sort_index(inplace=True)
    return cleaned


def download_yf_adjusted_ohlcv(
    tickers: List[str], period: str = "15mo", batch_size: int = 55
) -> Dict[str, pd.DataFrame]:
    """批量下載前復權日線，支援全量 550+ 隻港股通並發拉取"""
    ohlcv_dict: Dict[str, pd.DataFrame] = {}
    unique_tickers = list(dict.fromkeys(tickers))
    total_batches = (len(unique_tickers) - 1) // batch_size + 1

    for i in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[i : i + batch_size]
        print(f"正在下載前復權日線批次 {i // batch_size + 1}/{total_batches} ({len(batch)} 隻)...")
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if raw is None or raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0)
                for tk in batch:
                    if tk in lvl0:
                        sub = _clean_single_df(raw[tk].copy())
                        if not sub.empty:
                            ohlcv_dict[tk] = sub
            elif len(batch) == 1:
                sub = _clean_single_df(raw.copy())
                if not sub.empty:
                    ohlcv_dict[batch[0]] = sub
        except Exception as exc:
            print(f"[警告] 批次下載異常 ({exc})")
        time.sleep(0.7)

    return ohlcv_dict


def build_macro_summary(df_2800: pd.DataFrame, df_3033: pd.DataFrame) -> Tuple[Dict[str, Any], bool]:
    d1 = df_2800.copy()
    d1["MA50"] = d1["Close"].rolling(50).mean()
    d1["MA200"] = d1["Close"].rolling(200).mean()
    d1["Pct"] = d1["Close"].pct_change().fillna(0.0) * 100
    c1 = d1.iloc[-1]

    hsi_price_str = f"{float(c1['Close']):.2f} ({float(c1['Pct']):+.2f}%)"
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

    d2 = df_3033.copy()
    d2["Pct"] = d2["Close"].pct_change().fillna(0.0) * 100
    joined = d2[["Close", "Pct"]].join(d1[["Close"]].rename(columns={"Close": "Bench_Close"}), how="inner")
    joined["RS"] = joined["Close"] / joined["Bench_Close"]
    joined["RS_MA50"] = joined["RS"].rolling(50).mean()
    c2 = joined.iloc[-1]

    hstech_price_str = f"{float(c2['Close']):.2f} ({float(c2['Pct']):+.2f}%)"
    if c2["RS"] > c2["RS_MA50"]:
        hstech_rs = "RS 站穩 50D 均線走強"
        hstech_advice = "3033.HK 跑贏盈富基金，資金偏好高彈性科技與成長賽道，選股可優先側重科技股。"
    else:
        hstech_rs = "RS 弱於 50D 均線（防禦）"
        hstech_advice = "3033.HK 相對大盤走弱，高彈性科技股承壓，選股宜偏向高息藍籌或縮減進攻比例。"

    return {
        "hsi_price": hsi_price_str,
        "hsi_stage": hsi_stage,
        "hsi_advice": hsi_advice,
        "hstech_price": hstech_price_str,
        "hstech_rs": hstech_rs,
        "hstech_advice": hstech_advice,
    }, is_market_bullish


def main() -> None:
    universe, shares_delta_map = get_full_hk_connect_universe()
    print(f"本次目標掃描股票池總數: {len(universe)} 隻")

    all_tickers = ["2800.HK", "3033.HK"] + [item["symbol"] for item in universe]
    ohlcv_map = download_yf_adjusted_ohlcv(all_tickers, period="15mo", batch_size=55)

    if "2800.HK" not in ohlcv_map or "3033.HK" not in ohlcv_map:
        raise RuntimeError("無法獲取基準 ETF（2800.HK 或 3033.HK）日線數據。")

    macro_summary, is_market_bullish = build_macro_summary(ohlcv_map["2800.HK"], ohlcv_map["3033.HK"])

    results: List[Dict[str, Any]] = []
    for item in universe:
        sym = item["symbol"]
        if sym not in ohlcv_map:
            continue
        stock_df = ohlcv_map[sym].copy()

        delta_shares_3d = shares_delta_map.get(sym, 0)
        latest_close = float(stock_df["Close"].iloc[-1])
        nb_3d_hkd = delta_shares_3d * latest_close

        stock_df["Northbound_Net_Buy"] = 0.0
        if len(stock_df) >= 3 and nb_3d_hkd != 0.0:
            stock_df.iloc[-3:, stock_df.columns.get_loc("Northbound_Net_Buy")] = nb_3d_hkd / 3.0

        record = analyze_single_stock(
            symbol=sym,
            name=item["name"],
            sector=item["sector"],
            df=stock_df,
            bench_2800_df=ohlcv_map["2800.HK"],
            is_market_bullish=is_market_bullish,
        )
        if record:
            results.append(record)

    stage_priority = {"stage2a": 1, "stage2": 2, "warning": 3, "neutral": 4, "stage4": 5}
    results.sort(
        key=lambda x: (
            stage_priority.get(x.get("stage_code", "neutral"), 9),
            0 if x.get("is_golden_resonance") else 1,
            -float(x.get("vol_ratio", 0.0)),
        )
    )

    payload = sanitize_for_json({
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M HKT"),
        "universe_count": len(results),
        "macro": macro_summary,
        "stocks": results,
    })

    for out_file in ["scan_results.json", "data.json"]:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"掃描完成！共輸出 {len(results)} 隻有效港股通標的至 scan_results.json。")


if __name__ == "__main__":
    main()
