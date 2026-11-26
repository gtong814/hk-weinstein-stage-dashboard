import json
from datetime import datetime
from typing import Any, Dict, List
import numpy as np
import pandas as pd


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
    注意：df 與 bench_2800_df 必須按日期升序排列，且使用「前復權」價格。
    """
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
    data["MA200_Slope"] = data["MA200"] - data["MA200"].shift(20)
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
            # 檢查缺口之後至今最低價是否未跌破缺口下緣
            subsequent_min = data["Low"].iloc[idx:].min()
            if subsequent_min > gap_low:
                gap_info = f"HK${gap_low}-{gap_high}"

    # 6. 北水動向（若無傳入欄位則預設 0）
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

    # =========================================================================
    # 評估 200 日線斜率與 RS 共振狀態
    # =========================================================================
    ma200_up_or_flat = ma200_slope_pct >= -0.3  # 走平轉升或穩步向上
    rs_leading = (curr["RS"] > curr["RS_MA50"]) and (
        curr["RS"] >= curr["RS_Prev_High_40"] * 0.98
    )
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

    # =========================================================================
    # 評估假突破風險標記（供前端「過濾假突破」開關使用）
    # =========================================================================
    is_fake_breakout_risk = bool(
        (vol_ratio < 1.2)
        or (vol_ratio > 5.0 and curr["Upper_Shadow_Ratio"] > 1.8)
        or (bias_50 > 12.0)
        or (has_nb and nb_3d < -5_000_000)
    )
    is_golden_entry_zone = bool(
        (2.0 <= bias_50 <= 8.0) and (1.5 <= vol_ratio <= 3.5) and (not is_fake_breakout_risk)
    )

    # =========================================================================
    # 狀態機判定（優先級互斥）
    # =========================================================================
    tags: List[str] = []
    if gap_info != "無":
        tags.append("gap")
    if nb_3d > 5_000_000:
        tags.append("northbound")

    # 1. Stage 4A / 空頭
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

    # 2. 右側破 50 日線分級防守（Level 2 / Level 3）
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
        else:
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

    # 3. 左側 2B 止盈 / 極端乖離放量滯漲（Level 1）
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

    # 4. Stage 2A 突破（剛突破 40 日箱體頂部，且站穩 50 日線）
    is_stage2a_breakout = (
        curr["Close"] > curr["Prev_High_40"]
        and curr["Close"] > curr["MA50"]
        and curr["MA50_Slope"] >= 0
        and bias_50 <= 15.0
    )
    if is_stage2a_breakout:
        tags.append("stage2a")
        tags.append("stage2")
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


def export_dashboard_json(
    macro_summary: Dict[str, Any],
    stock_records: List[Dict[str, Any]],
    output_path: str = "data.json",
) -> None:
    """輸出供 GitHub Pages 前端讀取的標準化 JSON"""
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M HKT"),
        "universe_count": len(stock_records),
        "macro": macro_summary,
        "stocks": stock_records,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
