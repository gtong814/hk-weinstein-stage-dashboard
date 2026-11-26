import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf


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


def analyze_single_stock(
    symbol: str,
    name: str,
    sector: str,
    df: pd.DataFrame,
    bench_2800_df: pd.DataFrame,
    is_market_bullish: bool = True,
) -> Dict[str, Any]:
    if len(df) < 200 or len(bench_2800_df) < 200:
        return {}

    data = df.copy()
    bench = bench_2800_df[["Close"]].rename(columns={"Close": "Bench_Close"})
    data = data.join(bench, how="inner")
    if len(data) < 200:
        return {}

    # 1. 核心均線與斜率
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["MA50_Slope"] = data["MA50"] - data["MA50"].shift(5)
    data["MA200_Slope_Pct"] = np.where(
        data["MA200"].shift(20) > 0,
        (data["MA200"] - data["MA200"].shift(20)) / data["MA200"].shift(20) * 100,
        0.0,
    )

    # 2. 乖離率與 20 日量比
    data["Bias_50D"] = (data["Close"] - data["MA50"]) / data["MA50"] * 100
    data["Bias_200D"] = (data["Close"] - data["MA200"]) / data["MA200"] * 100
    data["Vol_MA20"] = data["Volume"].rolling(20).mean()
    data["Vol_Ratio_20D"] = np.where(
        data["Vol_MA20"] > 0, data["Volume"] / data["Vol_MA20"], 0.0
    )

    # 3. 相對強度（RS vs 2800.HK）
    data["RS"] = data["Close"] / data["Bench_Close"]
    data["RS_MA50"] = data["RS"].rolling(50).mean()
    data["RS_Prev_High_40"] = data["RS"].shift(1).rolling(40).max()

    # 4. K 線結構與區間高低點
    data["Pct_Change"] = data["Close"].pct_change() * 100
    body = (data["Close"] - data["Open"]).abs()
    upper_shadow = data["High"] - np.maximum(data["Open"], data["Close"])
    day_range = data["High"] - data["Low"]

    data["Close_Pos"] = np.where(day_range > 0, (data["Close"] - data["Low"]) / day_range, 0.5)
    data["Upper_Shadow_Ratio"] = np.where(
        body > 0, upper_shadow / body, np.where(upper_shadow > 0, 99.0, 0.0)
    )
    data["Prev_High_40"] = data["High"].shift(1).rolling(40).max()
    data["Prev_Low_60"] = data["Low"].shift(1).rolling(60).min()
    data["Max_Bias_5D"] = data["Bias_50D"].rolling(5).max()

    # 5. 未補向上跳空缺口檢測（近 15 日內）
    gap_info = "無"
    for idx in range(len(data) - 15, len(data)):
        if idx <= 0:
            continue
        if data["Low"].iloc[idx] > data["High"].iloc[idx - 1] * 1.005:
            gap_low = round(float(data["High"].iloc[idx - 1]), 2)
            gap_high = round(float(data["Low"].iloc[idx]), 2)
            if data["Low"].iloc[idx:].min() > gap_low:
                gap_info = f"HK${gap_low}-{gap_high}"

    # 6. 北水動向
    has_nb = "Northbound_Net_Buy" in data.columns
    nb_3d = float(data["Northbound_Net_Buy"].tail(3).sum()) if has_nb else 0.0
    nb_status = (
        "北水連續淨買入" if nb_3d > 5_000_000 else ("北水淨流出" if nb_3d < -5_000_000 else "北水持平")
    )

    curr = data.iloc[-1]
    prev = data.iloc[-2]

    close_px = round(float(curr["Close"]), 2)
    pct_chg = round(float(curr["Pct_Change"]), 2)
    bias_50 = round(float(curr["Bias_50D"]), 2)
    bias_200 = round(float(curr["Bias_200D"]), 2)
    vol_ratio = round(float(curr["Vol_Ratio_20D"]), 2)
    ma50_val = round(float(curr["MA50"]), 2)
    ma200_slope_pct = round(float(curr["MA200_Slope_Pct"]), 2)

    # 200 日線斜率與 RS 共振定性
    ma200_up_or_flat = ma200_slope_pct >= -0.3
    rs_leading = (curr["RS"] > curr["RS_MA50"]) and (curr["RS"] >= curr["RS_Prev_High_40"] * 0.98)
    rs_above_ma = curr["RS"] > curr["RS_MA50"]

    if ma200_up_or_flat and rs_leading:
        resonance_tier = "黃金共振"
        slope_rs_label = f"200D走平/向上 ({ma200_slope_pct:+.1f}%) · RS領先創高"
    elif (not ma200_up_or_flat) and rs_leading:
        resonance_tier = "早期先鋒"
        slope_rs_label = f"200D仍微降 ({ma200_slope_pct:+.1f}%) · RS強勢領先"
    elif ma200_up_or_flat and (not rs_above_ma):
        resonance_tier = "補漲落後"
        slope_rs_label = f"200D向上 ({ma200_slope_pct:+.1f}%) · RS走平轉弱"
    else:
        resonance_tier = "弱勢結構"
        slope_rs_label = f"200D向下 ({ma200_slope_pct:+.1f}%) · RS弱於大市"

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

    # Stage 4A / 空頭
    if (
        curr["Close"] < curr["MA50"]
        and curr["MA50_Slope"] < 0
        and curr["Close"] < curr["Prev_Low_60"]
        and (curr["MA50"] < curr["MA200"] or curr["Close"] < curr["MA200"])
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
            "advice": "【Level 4 清倉 100%】50日線下彎並跌破近60日箱體底部，結構已破壞，無條件離場不抄底。",
            "tags": tags,
        }

    # 右側破 50 日線分級防守
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

    # 左側 2B 止盈
    is_bull_extended = (
        curr["MA50"] > curr["MA200"]
        and curr["MA50_Slope"] > 0
        and curr["Max_Bias_5D"] >= 15.0
    )
    if is_bull_extended:
        cond_2b = (
            curr["High"] > curr["Prev_High_40"]
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

    # Stage 2A 突破
    is_stage2a_breakout = (
        curr["Close"] > curr["Prev_High_40"]
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

    # Stage 2 多頭延續
    if curr["Close"] >= curr["MA50"] and curr["MA50"] > curr["MA200"]:
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

    # Stage 1 / 3 震盪觀察
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


def normalize_hk_symbol(raw_code: Any) -> str:
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if not digits:
        return ""
    return f"{digits.lstrip('0').zfill(4)}.HK"


def fetch_hk_connect_universe() -> List[Dict[str, str]]:
    try:
        df_ggt = ak.stock_hk_ggt_components_em()
        if df_ggt is not None and not df_ggt.empty:
            code_col = next((c for c in df_ggt.columns if "代碼" in c or "代码" in c), None)
            name_col = next((c for c in df_ggt.columns if "名稱" in c or "名称" in c), None)
            sector_col = next((c for c in df_ggt.columns if "行業" in c or "行业" in c or "板塊" in c), None)
            if code_col and name_col:
                universe, seen = [], set()
                for _, row in df_ggt.iterrows():
                    sym = normalize_hk_symbol(row[code_col])
                    if sym and sym not in seen:
                        seen.add(sym)
                        universe.append({
                            "symbol": sym,
                            "name": str(row[name_col]).strip(),
                            "sector": str(row[sector_col]).strip() if sector_col and pd.notna(row[sector_col]) else "港股通標的",
                        })
                if len(universe) >= 50:
                    return universe
    except Exception as exc:
        print(f"[警告] 動態獲取港股通名單失敗 ({exc})，改用內置核心備援清單。")
    return FALLBACK_UNIVERSE


def fetch_northbound_3d_flow_map() -> Dict[str, float]:
    flow_map: Dict[str, float] = {}
    try:
        df_nb = ak.stock_hsgt_hold_stock_em(market="港股通", indicator="3日排行")
        if df_nb is not None and not df_nb.empty:
            code_col = next((c for c in df_nb.columns if "代碼" in c or "代码" in c), None)
            val_col = next((c for c in df_nb.columns if ("3日" in c and ("增持" in c or "淨買" in c or "净买" in c))), None)
            if not val_col:
                val_col = next((c for c in df_nb.columns if "增持" in c or "淨買" in c or "净买" in c), None)
            if code_col and val_col:
                for _, row in df_nb.iterrows():
                    sym = normalize_hk_symbol(row[code_col])
                    raw_val = pd.to_numeric(row[val_col], errors="coerce")
                    if sym and pd.notna(raw_val):
                        flow_map[sym] = float(raw_val)
    except Exception as exc:
        print(f"[提示] 北水排行接口暫不可用 ({exc})，跳過北水加權。")
    return flow_map


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


def download_yf_adjusted_ohlcv(tickers: List[str], period: str = "15mo", batch_size: int = 45) -> Dict[str, pd.DataFrame]:
    ohlcv_dict: Dict[str, pd.DataFrame] = {}
    unique_tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[i : i + batch_size]
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
            if len(batch) == 1:
                sub = raw.copy()
                if isinstance(sub.columns, pd.MultiIndex):
                    sub.columns = sub.columns.get_level_values(0)
                sub = _clean_single_df(sub)
                if not sub.empty:
                    ohlcv_dict[batch[0]] = sub
            else:
                for tk in batch:
                    if tk in raw.columns.get_level_values(0):
                        sub = _clean_single_df(raw[tk].copy())
                        if not sub.empty:
                            ohlcv_dict[tk] = sub
        except Exception as exc:
            print(f"[警告] 批次下載異常 ({exc})")
        time.sleep(0.8)
    return ohlcv_dict


def build_macro_summary(df_2800: pd.DataFrame, df_3033: pd.DataFrame) -> Tuple[Dict[str, Any], bool]:
    d1 = df_2800.copy()
    d1["MA50"] = d1["Close"].rolling(50).mean()
    d1["MA200"] = d1["Close"].rolling(200).mean()
    d1["Pct"] = d1["Close"].pct_change() * 100
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
    d2["Pct"] = d2["Close"].pct_change() * 100
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
    universe = fetch_hk_connect_universe()
    nb_flow_map = fetch_northbound_3d_flow_map()
    all_tickers = ["2800.HK", "3033.HK"] + [item["symbol"] for item in universe]
    ohlcv_map = download_yf_adjusted_ohlcv(all_tickers, period="15mo", batch_size=45)

    if "2800.HK" not in ohlcv_map or "3033.HK" not in ohlcv_map:
        raise RuntimeError("無法獲取基準 ETF（2800.HK 或 3033.HK）日線數據。")

    macro_summary, is_market_bullish = build_macro_summary(ohlcv_map["2800.HK"], ohlcv_map["3033.HK"])

    results: List[Dict[str, Any]] = []
    for item in universe:
        sym = item["symbol"]
        if sym not in ohlcv_map:
            continue
        stock_df = ohlcv_map[sym].copy()
        nb_3d_total = nb_flow_map.get(sym, 0.0)
        stock_df["Northbound_Net_Buy"] = 0.0
        if len(stock_df) >= 3 and nb_3d_total != 0.0:
            stock_df.iloc[-3:, stock_df.columns.get_loc("Northbound_Net_Buy")] = nb_3d_total / 3.0

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

    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M HKT"),
        "universe_count": len(results),
        "macro": macro_summary,
        "stocks": results,
    }
    # 同時寫入 scan_results.json（配合你現有工作流）與 data.json（雙保險）
    for out_file in ["scan_results.json", "data.json"]:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"掃描完成！共分析 {len(results)} 隻有效港股通標的，已寫入 scan_results.json。")


if __name__ == "__main__":
    main()
