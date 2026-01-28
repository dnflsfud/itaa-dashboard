# -*- coding: utf-8 -*-
"""
Enhanced Black-Litterman Portfolio Optimization Dashboard — ITAA Master 전용
v2025-11-20-ss3 (리스크 시뮬레이션 & P&L 분포 기능 추가)

주요 개선사항:
1. 일별 성과 탭에서 리스크 제약 방법 변경 시 성과지표 재계산
2. 포지션 크기 변화 테이블 추가
3. 원본 vs 제약 적용 비교 차트

ss3 버전 추가 기능 (Actual Portfolio 탭):
1. 포지션 크기 조정 → 리스크 변화 시뮬레이션
2. 목표 리스크 변화 → 포지션 크기 역산
3. -3σ ~ +3σ 시나리오별 P&L 분포 그래프 (현재/조정 포지션 비교)
"""

import os
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import rankdata, norm
from scipy.optimize import minimize, nnls
import warnings

import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional, Union, Any

warnings.filterwarnings('ignore')


# === PATCH C helpers (module-scope; avoid UnboundLocalError) ===
import numpy as _np
import pandas as _pd

def _pc_ensure_decimal_returns(df: _pd.DataFrame) -> _pd.DataFrame:
    x = df.replace([_np.inf, -_np.inf], _np.nan).astype(float)
    med = x.abs().stack(dropna=True).median()
    # 수익률이 %스케일(≈1=1%)이면 소수로 변환
    return x / 100.0 if (med is not None and _np.isfinite(med) and med > 0.2) else x

def _pc_build_recent_cov_constant_corr(ret_df: _pd.DataFrame,
                                       window: int = 63, rho: float = 0.25) -> _pd.DataFrame:
    X = ret_df.tail(window)
    std = X.std(ddof=1).fillna(0.0)
    S = _np.outer(std, std) * float(rho)
    _np.fill_diagonal(S, std.values ** 2)
    return _pd.DataFrame(S, index=ret_df.columns, columns=ret_df.columns)

def _pc_te_bp_from_cov(w, cov_df: _pd.DataFrame, ann_factor: int = 252) -> float:
    w = _np.asarray(_pd.Series(w).values, float).reshape(-1)
    var = float(w @ cov_df.values @ w)
    sigma_ann = (var ** 0.5) * (ann_factor ** 0.5)
    return sigma_ann * 1e4  # → basis points


# =============================================================================
# Streamlit 설정
# =============================================================================
st.set_page_config(
    page_title="ITAA Black-Litterman Portfolio Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    .stButton>button { width: 100%; }
    .metric-container { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .timeline-event { background-color: #e8f4f8; padding: 10px; border-left: 3px solid #1f77b4; margin: 10px 0; border-radius: 5px; }
    .rebalance-log { background-color: #f8f8f8; padding: 8px; border-left: 3px solid #ff7f0e; margin: 5px 0; border-radius: 3px; }
    .rank-change { padding: 5px 10px; border-radius: 3px; font-weight: bold; }
    .new-pair { background-color: #e7f3ff; padding: 10px; border-radius: 5px; margin: 10px 0; }
    .risk-indicator { background-color: #fff3cd; padding: 10px; border-left: 3px solid #ffc107; margin: 10px 0; border-radius: 5px; }
    .scenario-box { background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0; border: 2px solid #4682b4; }
    .constraint-box { background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0; border: 2px solid #4caf50; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 유틸리티 함수
# =============================================================================
def _to_float(x):
    """안전한 float 변환"""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s == "":
            return np.nan
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    return np.nan


def normalize_score(scores, min_val=0, max_val=1):
    """점수 정규화"""
    scores = np.array(scores, dtype=float)
    if scores.size == 0:
        return scores
    score_min = np.nanmin(scores)
    score_max = np.nanmax(scores)
    if not np.isfinite(score_min) or not np.isfinite(score_max):
        return np.zeros_like(scores)
    if score_max == score_min:
        return np.full_like(scores, (min_val + max_val) / 2.0)
    normalized = (scores - score_min) / (score_max - score_min)
    return normalized * (max_val - min_val) + min_val


def _norm(s: str) -> str:
    """문자열 정규화"""
    return str(s).strip().casefold()


def _safe_dt(x):
    """안전한 datetime 변환"""
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def is_cash_asset(asset_name: str) -> bool:
    """현금 자산 여부 확인"""
    if asset_name is None:
        return False
    asset_lower = str(asset_name).lower().strip()
    return asset_lower in ['cash', '현금', 'usd cash', 'krw cash']


def normalize_str(x: Union[str, float, int]) -> str:
    """문자열 정규화 (고급)"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()

# 리스크 제약 표시용 공통 맵 (전역)
CONSTRAINT_DISPLAY_MAP = {
    "3Y_MDD": "3년 MDD",
    "-3STD": "-3 표준편차 (3M)",
    "-2STD": "-2 표준편차 (3M)",
    "-1STD": "-1 표준편차 (3M)",
}

# =============================================================================
# 폰트 크기 설정 함수 (1단계 증가)
# =============================================================================
def apply_chart_font_settings(fig, title_size=20, axis_title_size=18, tick_size=16, legend_size=16):
    """차트에 통일된 폰트 크기 적용 (1단계 증가)"""
    fig.update_layout(
        title_font_size=title_size,
        font=dict(size=tick_size),
        legend=dict(font=dict(size=legend_size))
    )
    fig.update_xaxes(title_font_size=axis_title_size, tickfont_size=tick_size)
    fig.update_yaxes(title_font_size=axis_title_size, tickfont_size=tick_size)
    return fig


# =============================================================================
# 연율화 계산 함수
# =============================================================================
def calculate_annualized_metrics(returns_series: pd.Series, trading_days_per_year: int = 252) -> Dict[str, float]:
    """
    수익률 시계열로부터 연율화 수익률 및 변동성 계산

    Args:
        returns_series: 일별 수익률 시계열 (소수 형태)
        trading_days_per_year: 연간 거래일 수

    Returns:
        dict: {
            'cumulative_return': 누적 수익률 (소수),
            'annualized_return': 연율화 수익률 (소수),
            'annualized_volatility': 연율화 변동성 (소수),
            'n_days': 실제 거래일 수
        }
    """
    if returns_series.empty or len(returns_series) == 0:
        return {
            'cumulative_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'n_days': 0
        }

    # 누적 수익률
    cumulative = (1 + returns_series).prod() - 1

    # 거래일 수
    n_days = len(returns_series)

    # 연율화 수익률
    if n_days > 0:
        annualized_return = (1 + cumulative) ** (trading_days_per_year / n_days) - 1
    else:
        annualized_return = 0.0

    # 연율화 변동성
    daily_vol = returns_series.std()
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)

    return {
        'cumulative_return': float(cumulative),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'n_days': int(n_days)
    }


# =============================================================================
# 상관관계/리스크 계산 함수
# =============================================================================
def calculate_rolling_correlation(returns_df: pd.DataFrame, window: int = 60):
    """Rolling correlation 계산 (NaN 처리 개선)"""
    if returns_df.empty or len(returns_df.columns) < 2:
        return {}

    rolling_corr = {}
    assets = returns_df.columns.tolist()

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            asset1, asset2 = assets[i], assets[j]
            pair_name = f"{asset1} vs {asset2}"

            valid_data = returns_df[[asset1, asset2]].dropna()

            if len(valid_data) < window:
                continue

            corr = returns_df[asset1].rolling(window=window).corr(returns_df[asset2])
            corr = corr.dropna()

            if not corr.empty:
                rolling_corr[pair_name] = corr

    return rolling_corr


def calculate_correlation_stability(returns_df: pd.DataFrame, window: int = 60):
    """Correlation stability 계산 (NaN 처리 개선)"""
    if returns_df.empty or len(returns_df) < window * 2:
        return pd.DataFrame()

    valid_threshold = 0.8
    valid_assets = []
    for col in returns_df.columns:
        valid_ratio = returns_df[col].notna().sum() / len(returns_df)
        if valid_ratio >= valid_threshold:
            valid_assets.append(col)

    if len(valid_assets) < 2:
        return pd.DataFrame()

    returns_clean = returns_df[valid_assets].copy()

    n_windows = len(returns_clean) // window
    correlations = []

    for i in range(n_windows):
        start_idx = i * window
        end_idx = (i + 1) * window
        window_data = returns_clean.iloc[start_idx:end_idx]

        if window_data.notna().sum().min() >= window * 0.7:
            corr_matrix = window_data.corr()
            if not corr_matrix.isnull().all().all():
                correlations.append(corr_matrix)

    if not correlations:
        return pd.DataFrame()

    corr_stack = np.stack([corr.values for corr in correlations], axis=0)
    stability = pd.DataFrame(
        np.nanstd(corr_stack, axis=0),
        index=valid_assets,
        columns=valid_assets
    )

    return stability


def calculate_portfolio_weights(views_df: pd.DataFrame, weights_df: pd.DataFrame) -> np.ndarray:
    """
    Views와 Benchmark를 기반으로 포트폴리오 가중치 계산
    TE와 Vol 계산에서 동일하게 사용
    """
    if weights_df.empty:
        return np.array([])

    assets = weights_df["Asset"].astype(str).tolist()
    n_assets = len(assets)

    # 1. Optimal_Weight가 있으면 그것 사용
    if "Optimal_Weight" in weights_df.columns:
        opt_weights = pd.to_numeric(weights_df["Optimal_Weight"], errors="coerce").fillna(0.0).values
        return opt_weights

    # 2. Benchmark + View adjustments
    if "Benchmark_Weight" in weights_df.columns:
        bm_weights = pd.to_numeric(weights_df["Benchmark_Weight"], errors="coerce").fillna(0.0).values
        view_adjustments = np.zeros(n_assets)

        if not views_df.empty:
            for _, view in views_df.iterrows():
                long_asset = str(view.get("Long_Asset", ""))
                short_asset = str(view.get("Short_Asset", ""))
                signal = float(view.get("Signal", 0.0))
                conviction = float(view.get("Conviction", 1.0))
                strength = signal * conviction * 0.05

                if long_asset in assets:
                    idx_long = assets.index(long_asset)
                    view_adjustments[idx_long] += strength
                if short_asset in assets:
                    idx_short = assets.index(short_asset)
                    view_adjustments[idx_short] -= strength

        opt_weights = bm_weights + view_adjustments
        opt_weights = np.clip(opt_weights, 0, 1)
        s = opt_weights.sum()
        if s > 0:
            opt_weights = opt_weights / s
        return opt_weights

    return np.array([])


# =============================================================================
# 3M Rolling Return 계산
# =============================================================================
def calculate_pair_3m_rolling_returns(
        returns_by_asset: pd.DataFrame,
        long_asset: str,
        short_asset: str,
        signal: float,  # ✅ Signal 추가
        lookback_years: int = 3,
        rolling_window: int = 63
) -> pd.Series:
    """
    페어의 3개월 롤링 리턴 계산
    ✅ Signal 방향에 따라 스프레드 방향 결정

    Signal > 0: spread = Long - Short
    Signal < 0: spread = Short - Long
    """
    lookback_days = int(252 * lookback_years) + rolling_window
    recent_data = returns_by_asset.iloc[-lookback_days:]

    if long_asset not in recent_data.columns or short_asset not in recent_data.columns:
        return pd.Series(dtype=float)

    # ✅ Signal 방향에 따라 스프레드 계산
    if signal >= 0:
        spread_daily = (recent_data[long_asset] - recent_data[short_asset]).dropna()
    else:
        spread_daily = (recent_data[short_asset] - recent_data[long_asset]).dropna()

    if len(spread_daily) < rolling_window:
        return pd.Series(dtype=float)

    rolling_3m = spread_daily.rolling(window=rolling_window).sum()
    return rolling_3m.dropna()


def calculate_pair_scenarios_3m(pair_3m_returns: pd.Series, position_size: float) -> Dict[str, float]:
    """3개월 rolling return 기준 시나리오별 기대수익률 계산"""
    if pair_3m_returns.empty or len(pair_3m_returns) < 20:
        return {}

    mean_return = pair_3m_returns.mean()
    std_return = pair_3m_returns.std()

    scenarios = {}
    std_levels = [-3, -2, -1, 1, 2, 3]

    for std_level in std_levels:
        spread_return_3m = mean_return + std_level * std_return
        portfolio_return = position_size * spread_return_3m
        scenarios[f"{std_level}std"] = portfolio_return * 10000  # bp 단위

    scenarios['mean_3m_bp'] = mean_return * position_size * 10000
    scenarios['std_3m_bp'] = std_return * position_size * 10000
    scenarios['sharpe_3m'] = mean_return / std_return if std_return > 0 else 0
    scenarios['position_bp'] = position_size * 10000

    # 연율화 근사
    scenarios['annualized_return_bp'] = scenarios['mean_3m_bp'] * 4
    scenarios['annualized_std_bp'] = scenarios['std_3m_bp'] * 2

    return scenarios


def calculate_common_positions(
        returns_by_asset: pd.DataFrame,
        views_df: pd.DataFrame,
        constraint_method: str,
        lookback_years: int = 3
) -> pd.DataFrame:
    """
    모든 탭에서 사용할 공통 포지션 계산
    ✅ Signal 방향 반영
    """
    if returns_by_asset.empty or views_df.empty:
        return pd.DataFrame()

    pairs = [(str(r['Long_Asset']), str(r['Short_Asset']))
             for _, r in views_df.iterrows()]
    signals = views_df['Signal'].astype(float).values
    pair_ids = views_df.get('Pair_ID', range(len(pairs))).values

    # RiskConstraintCalculator (Signal 전달)
    risk_calc = RiskConstraintCalculator(
        returns_by_asset,
        lookback_years=lookback_years,
        rolling_window_days=63,
        use_exponential_weighting=True,
        ewm_halflife_days=126,
        max_loss_bp_map={1: 0.10, 2: 0.15}
    )

    # ✅ Signal을 개별적으로 전달하여 방향 반영
    constraint_values, cap_arr = risk_calc.calculate_position_caps(
        pairs=pairs,
        signals=signals,
        constraint_method=constraint_method,
        asof_date=None,
        kappa_mode="cash-aware"
    )

    # Cash pair 감지
    def is_cash_name(name):
        if name is None:
            return False
        s = str(name).upper()
        cash_keywords = ("CASH", "T-BILL", "TBILL", "MMF", "CALL",
                         "KTB 3M", "UST 3M", "MONEY")
        return any(kw in s for kw in cash_keywords)

    is_cash_pair = [is_cash_name(la) or is_cash_name(sa) for la, sa in pairs]
    leg_factors = np.where(is_cash_pair, 1, 2)

    # ✅ Signal 방향 반영한 포지션
    # cap_arr는 항상 양수, Signal의 부호를 곱해서 방향 결정
    signed_caps = cap_arr * np.sign(signals)

    position_df = pd.DataFrame({
        'Pair_ID': pair_ids,
        'Pair': [f"{p[0]} vs {p[1]}" for p in pairs],
        'Long_Asset': [p[0] for p in pairs],
        'Short_Asset': [p[1] for p in pairs],
        'Signal': signals,
        'Is_Cash_Pair': is_cash_pair,
        'Leg_Factor': leg_factors,

        'Risk_Unit_3M_%': (constraint_values * 100).round(3),
        'Max_Loss_bp': [risk_calc.get_max_loss_bp(s) for s in signals],

        # ✅ Signal 방향이 반영된 포지션
        'Per_Leg_Position_bp': (signed_caps * 10000).round(3),
        'Total_Notional_bp': (signed_caps * leg_factors * 10000).round(3),

        'Constraint_Method': constraint_method
    })

    return position_df



# =============================================================================
# 리스크 제약 계산 클래스 (수정됨)
# =============================================================================
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


# =============================================================================
# 헬퍼 함수: Signal 기반 손실 허용치 계산
# =============================================================================
def _compute_loss_caps_bp_from_views(views_df: pd.DataFrame) -> np.ndarray:
    """
    Signal 강도(|S|)별 손실 허용치(bp) 벡터 생성.

    Rules:
    - |S| ≥ 2.0 → 0.15bp
    - |S| ≥ 1.0 → 0.10bp
    - 그 외 → 0.10 + 0.05 * |S| bp

    Args:
        views_df: Views DataFrame with 'Signal' column

    Returns:
        np.ndarray: 각 view에 대한 최대 손실 허용치 (bp)
    """
    s = pd.to_numeric(views_df.get("Signal", 0), errors="coerce").fillna(0.0).abs().values
    return np.where(s >= 2.0, 0.15,
                    np.where(s >= 1.0, 0.10, 0.10 + 0.05 * s))

class RiskConstraintCalculator:
    """
    리스크 제약 계산기 - 3개월 롤링 리턴 기반
    """

    def __init__(
        self,
        returns_by_asset: pd.DataFrame,
        lookback_years: int = 3,
        rolling_window_days: int = 63,
        z_default: float = 3.0,
        max_loss_bp_map: Optional[Dict[int, float]] = None,
        cash_keywords: Optional[List[str]] = None,
        min_sigma: float = 1e-6,
        use_exponential_weighting: bool = True,
        ewm_halflife_days: int = 126,
    ):
        """
        Parameters
        ----------
        returns_by_asset : pd.DataFrame
            일별 수익률(소수) 테이블. index=DatetimeIndex, columns=자산명
        lookback_years : int
            롤링 리턴 계산에 사용할 과거 기간 (년)
        rolling_window_days : int
            롤링 윈도우 크기 (영업일, 기본값: 63 = 약 3개월)
        z_default : float
            constraint_method 파싱 실패 시 사용할 기본 Z배수
        max_loss_bp_map : dict
            신호 강도 → 허용 최대 손실(bp) 매핑
            예) {1: 10, 2: 15}  ← 10bp, 15bp (소수 아님!)
        cash_keywords : list[str]
            캐시 자산 식별 키워드
        min_sigma : float
            수치 안정화를 위한 σ 하한
        use_exponential_weighting : bool
            True면 EWM std 사용 (최근 강조)
        ewm_halflife_days : int
            EWM 사용 시 halflife (영업일)
        """
        self.rets = returns_by_asset.sort_index()
        self.lookback_years = lookback_years
        self.rolling_window_days = int(rolling_window_days)
        self.z_default = float(z_default)
        self.min_sigma = float(min_sigma)
        self.use_ewm = use_exponential_weighting
        self.ewm_halflife = int(ewm_halflife_days)

        # bp 단위! (소수 아님)
        self.max_loss_bp_map = max_loss_bp_map or {1: 0.1, 2: 0.15}

        self.cash_keywords = [k.strip().lower() for k in (cash_keywords or [
            "cash", "money", "mm", "t-bill", "tbill", "bill", "bills",
            "ktb 3m", "ust 3m", "mmf", "call"
        ])]

        assert isinstance(self.rets.index, pd.DatetimeIndex), \
            "returns_by_asset.index는 DatetimeIndex여야 합니다."

    def get_max_loss_bp(self, signal_value: float) -> float:
        """
        |Signal|에 따른 허용 최대 손실(bp)을 반환
        """
        a = int(abs(round(float(signal_value))))
        if a in self.max_loss_bp_map:
            return float(self.max_loss_bp_map[a])
        if 1 in self.max_loss_bp_map:
            return float(self.max_loss_bp_map[1])
        k = sorted(self.max_loss_bp_map.keys())[0]
        return float(self.max_loss_bp_map[k])

    def calculate_position_caps(
            self,
            pairs: List[Tuple[str, str]],
            signals: List[float],
            constraint_method: str,
            asof_date: Optional[pd.Timestamp] = None,
            kappa_mode: str = "symmetric",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        각 페어에 대해 (risk_unit, cap) 계산 (Signal 방향 반영)
        """
        z = abs(self._parse_z(constraint_method))

        lookback_days = int(252 * self.lookback_years) + self.rolling_window_days
        end_idx = self._resolve_asof_index(asof_date)
        start_idx = max(0, end_idx - lookback_days + 1)
        window = self.rets.iloc[start_idx:end_idx + 1]

        risk_units = []
        caps = []

        for (la, sa), sig in zip(pairs, signals):
            la = str(la)
            sa = str(sa)

            # ✅ Signal 방향에 따라 스프레드 방향 결정
            if kappa_mode == "cash-aware" and self._is_cash_pair(la, sa):
                mu_3m, sigma_3m = self._single_leg_stats_3m_rolling(window, la, sa, sig)
                leg_factor = 1
            else:
                mu_3m, sigma_3m = self._pair_spread_stats_3m_rolling(window, la, sa, sig)
                leg_factor = 2

            # -zσ 시나리오
            scenario_loss = abs(mu_3m - z * sigma_3m)
            risk_unit = max(self.min_sigma, scenario_loss)

            # 허용 손실 (bp → 소수)
            max_loss_decimal = self.get_max_loss_bp(sig) / 10_000.0

            # Per-leg cap (항상 양수)
            cap = max_loss_decimal / (risk_unit * leg_factor) if risk_unit > 0 else np.inf

            risk_units.append(float(risk_unit))
            caps.append(float(cap))

        return np.array(risk_units, dtype=float), np.array(caps, dtype=float)

    def _pair_spread_stats_3m_rolling(
            self,
            window: pd.DataFrame,
            la: str,
            sa: str,
            signal: float  # ✅ Signal 추가
    ) -> Tuple[float, float]:
        """
        페어 스프레드의 3개월 롤링 리턴 평균과 표준편차
        ✅ Signal 방향에 따라 스프레드 방향 결정

        Signal > 0: spread = Long - Short (기존)
        Signal < 0: spread = Short - Long (반전)
        """
        if la not in window.columns and sa not in window.columns:
            return 0.0, 0.0

        a = window[la] if la in window.columns else pd.Series(0.0, index=window.index)
        b = window[sa] if sa in window.columns else pd.Series(0.0, index=window.index)

        # ✅ Signal 방향에 따라 스프레드 계산
        if signal >= 0:
            spread_daily = (a - b).dropna()  # Long - Short
        else:
            spread_daily = (b - a).dropna()  # Short - Long (반전)

        min_required = self.rolling_window_days + 20
        if len(spread_daily) < min_required:
            return 0.0, 0.0

        # 3개월 롤링 리턴
        rolling_3m = spread_daily.rolling(window=self.rolling_window_days).sum()
        rolling_3m_clean = rolling_3m.dropna()

        if len(rolling_3m_clean) < 2:
            return 0.0, 0.0

        # EWM 통계
        if self.use_ewm:
            ewm_mean = rolling_3m_clean.ewm(halflife=self.ewm_halflife).mean()
            ewm_std = rolling_3m_clean.ewm(halflife=self.ewm_halflife).std()
            mu = float(ewm_mean.iloc[-1]) if len(ewm_mean) > 0 else 0.0
            sigma = float(ewm_std.iloc[-1]) if len(ewm_std) > 0 else 0.0
        else:
            mu = float(rolling_3m_clean.mean())
            sigma = float(rolling_3m_clean.std(ddof=1))

        return mu, sigma


    def _single_leg_stats_3m_rolling(
            self,
            window: pd.DataFrame,
            la: str,
            sa: str,
            signal: float  # ✅ Signal 추가
    ) -> Tuple[float, float]:
        """
        Cash pair용: 비캐시 leg의 3개월 롤링 리턴
        ✅ Signal 방향에 따라 수익률 방향 결정
        """
        la_is_cash = self._is_cash_name(la)
        sa_is_cash = self._is_cash_name(sa)

        min_required = self.rolling_window_days + 20

        # 비캐시 자산 선택
        target_asset = None
        if la in window.columns and not la_is_cash:
            target_asset = la
        elif sa in window.columns and not sa_is_cash:
            target_asset = sa

        if target_asset is None:
            return 0.0, 0.0

        series = window[target_asset].dropna()

        if len(series) < min_required:
            return 0.0, 0.0

        # ✅ Signal < 0이면 수익률 반전
        if signal < 0:
            series = -series

        rolling_3m = series.rolling(window=self.rolling_window_days).sum()
        rolling_3m_clean = rolling_3m.dropna()

        if len(rolling_3m_clean) < 2:
            return 0.0, 0.0

        if self.use_ewm:
            ewm_mean = rolling_3m_clean.ewm(halflife=self.ewm_halflife).mean()
            ewm_std = rolling_3m_clean.ewm(halflife=self.ewm_halflife).std()
            mu = float(ewm_mean.iloc[-1]) if len(ewm_mean) > 0 else 0.0
            sigma = float(ewm_std.iloc[-1]) if len(ewm_std) > 0 else 0.0
        else:
            mu = float(rolling_3m_clean.mean())
            sigma = float(rolling_3m_clean.std(ddof=1))

        return mu, sigma


    def _resolve_asof_index(self, asof_date: Optional[pd.Timestamp]) -> int:
        """asof_date 인덱스 위치 반환"""
        if asof_date is None:
            return len(self.rets.index) - 1

        if asof_date in self.rets.index:
            return self.rets.index.get_loc(asof_date)

        pos = self.rets.index.searchsorted(asof_date, side="right") - 1
        if pos < 0:
            return 0
        return int(pos)

    def _parse_z(self, constraint_method: str) -> float:
        """
        문자열에서 Z배수 파싱
        """
        if constraint_method is None:
            return self.z_default

        s = str(constraint_method).lower().strip()

        # 숫자만
        try:
            return abs(float(s))
        except ValueError:
            pass

        # 'z=3'
        if "z" in s:
            try:
                return abs(float(s.split("=")[-1]))
            except Exception:
                pass

        # '-3std', '2std'
        for tok in ["std", "σ", "sigma"]:
            if tok in s:
                try:
                    num_str = s.replace(tok, "").replace("-", "").strip()
                    return abs(float(num_str))
                except Exception:
                    continue

        return self.z_default

    def _is_cash_name(self, name: str) -> bool:
        n = name.lower()
        return any(k in n for k in self.cash_keywords)

    def _is_cash_pair(self, la: str, sa: str) -> bool:
        return self._is_cash_name(la) or self._is_cash_name(sa)

    def explain_window(self, asof_date: Optional[pd.Timestamp] = None) -> Dict[str, object]:
        """윈도우 설명"""
        lookback_days = int(252 * self.lookback_years) + self.rolling_window_days
        end_idx = self._resolve_asof_index(asof_date)
        start_idx = max(0, end_idx - lookback_days + 1)
        idx = self.rets.index
        return {
            "lookback_years": self.lookback_years,
            "rolling_window_days": self.rolling_window_days,
            "use_exponential_weighting": self.use_ewm,
            "ewm_halflife_days": self.ewm_halflife if self.use_ewm else None,
            "start_date": idx[start_idx] if len(idx) else None,
            "end_date": idx[end_idx] if len(idx) else None,
            "z_default": self.z_default,
        }


# =============================================================================
# 리스크 제약 적용 일별 수익률 재계산 (cap=사이즈 직결 + 호환 컬럼 포함)
# =============================================================================
def calculate_daily_returns_with_constraint(
        returns_by_asset: pd.DataFrame,
        views_timeline: pd.DataFrame,
        w_bmk_daily: pd.DataFrame,
        w_opt_daily: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        constraint_method: str,
        lookback_years: int = 3,
        kappa_mode: str = "cash-aware",
        sizing_mode: str = "full_cap", # "cash-aware" | "symmetric"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    리스크 제약을 적용하여 일별 수익률 재계산 (패치 버전)
    - 원본 Optimal 구조 유지한 채 pair별 per-leg cap만 clip 방식으로 적용
    - 숏 포지션 유지 (음수 제거 안 함)
    - FX-vs-Cash 등 '한쪽만 위험'인 페어는 kappa_mode="cash-aware"로 처리

    Parameters
    ----------
    returns_by_asset : 일별 자산 수익률(소수)
    views_timeline   : 활성 pair/Signal/기간 정보
      필수 컬럼 예시: ["Long_Asset","Short_Asset","Signal","Start_Date","End_Date","Pair_ID"(선택)]
      선택 컬럼: ["Is_Cash_Pair"] (있으면 cash-aware 판정에 그대로 사용)
    w_bmk_daily      : 일별 벤치마크 가중치
    w_opt_daily      : 일별 최적 가중치(원본)
    start_date       : 시작일
    end_date         : 종료일
    constraint_method: 리스크 단위 계산 방법(예: "zscore_vol", "recent_vol" 등 클래스 내부 구현과 일치)
    lookback_years   : 리스크 단위 롤링 윈도우 길이(년)
    kappa_mode       : "cash-aware" 또는 "symmetric"

    Returns
    -------
    daily_returns_df : (Portfolio_Return, Benchmark_Return, Active_Return)
    position_changes_df : 페어별 일자/사이즈/손실 허용/실측 리스크 등 상세 로그
                          ※ KeyError 방지용 'Total_Active_bp' 포함
    """
    if returns_by_asset.empty or views_timeline.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 날짜 필터
    mask = (returns_by_asset.index >= start_date) & (returns_by_asset.index <= end_date)
    dates = returns_by_asset.index[mask]
    if len(dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    assets_list = returns_by_asset.columns.tolist()
    risk_calc = RiskConstraintCalculator(returns_by_asset, lookback_years=lookback_years)

    portfolio_returns: List[float] = []
    benchmark_returns: List[float] = []
    active_returns: List[float] = []
    used_dates: List[pd.Timestamp] = []
    position_records: List[Dict[str, Any]] = []

    # --- 간단한 현금/현금성 추정(폴백) ---
    def _is_cashish(name: str) -> bool:
        if name is None:
            return False
        s = str(name).upper()
        # 너무 공격적이지 않도록 'USDJPY' 같은 FX 종목을 오탐하지 않게 패턴 제한
        cash_tokens = (" CASH", "CASH ", "T-BILL", "TBILL", "MMF", "CALL", "KTB 3M", "UST 3M", "BILLS")
        return any(tok in s for tok in cash_tokens) or s.strip() in {"CASH", "USD CASH", "KRW CASH"}

    for date in dates:
        if date not in returns_by_asset.index:
            continue
        daily_returns = returns_by_asset.loc[date]

        # ── 벤치마크 가중치 구하기(해당일 없으면 직전일 사용) ──
        if not w_bmk_daily.empty:
            if date in w_bmk_daily.index:
                w_bmk = w_bmk_daily.loc[date].reindex(assets_list).fillna(0.0)
            else:
                prev_idx = w_bmk_daily.index[w_bmk_daily.index <= date]
                w_bmk = (w_bmk_daily.loc[prev_idx[-1]].reindex(assets_list).fillna(0.0)) if len(prev_idx) else pd.Series(0.0, index=assets_list)
        else:
            w_bmk = pd.Series(0.0, index=assets_list)

        # ── 원본 Optimal 가중치(해당일 없으면 직전일) ──
        if not w_opt_daily.empty:
            if date in w_opt_daily.index:
                w_opt_original = w_opt_daily.loc[date].reindex(assets_list).fillna(0.0)
            else:
                prev_idx = w_opt_daily.index[w_opt_daily.index <= date]
                w_opt_original = (w_opt_daily.loc[prev_idx[-1]].reindex(assets_list).fillna(0.0)) if len(prev_idx) else pd.Series(0.0, index=assets_list)
        else:
            w_opt_original = pd.Series(0.0, index=assets_list)

        # 합 0 또는 음수 방지용 정규화(벤치마크만)
        bmk_sum = w_bmk.sum()
        if bmk_sum > 0:
            w_bmk = w_bmk / bmk_sum

        # 최적 가중치는 Long/Short 합이 1이 아닐 수 있으므로 그대로 둠
        w_active_original = (w_opt_original - w_bmk).reindex(assets_list).fillna(0.0)

        bmk_return = float((w_bmk * daily_returns).sum())

        # ── 해당일 활성 뷰 추출 ──
        active_views = views_timeline.copy()
        if 'Start_Date' in active_views.columns:
            active_views = active_views[active_views['Start_Date'].fillna(pd.Timestamp.min) <= date]
        if 'End_Date' in active_views.columns:
            active_views = active_views[active_views['End_Date'].fillna(pd.Timestamp.max) >= date]
        active_views = active_views[active_views.get('Signal', 0) != 0]

        # 뷰가 없으면 제약 없이 원본으로 계산
        if active_views.empty:
            port_return = float((w_opt_original * daily_returns).sum())
            portfolio_returns.append(port_return)
            benchmark_returns.append(bmk_return)
            active_returns.append(port_return - bmk_return)
            used_dates.append(date)
            continue

        # ── Pair/Signal/ID 수집 ──
        pairs: List[Tuple[str, str]] = [(str(r['Long_Asset']), str(r['Short_Asset'])) for _, r in active_views.iterrows()]
        signals = active_views['Signal'].astype(float).values
        pair_ids = active_views['Pair_ID'].values if 'Pair_ID' in active_views.columns else np.arange(len(pairs))

        # cash-aware 판정 소스(명시 컬럼 우선, 없으면 휴리스틱)
        has_is_cash_pair_col = 'Is_Cash_Pair' in active_views.columns
        is_cash_pair_flags = []
        if has_is_cash_pair_col:
            is_cash_pair_flags = active_views['Is_Cash_Pair'].astype(bool).tolist()
        else:
            # 휴리스틱: 자산명 텍스트 기반
            is_cash_pair_flags = [_is_cashish(pa) or _is_cashish(pb) for (pa, pb) in pairs]

        # ── incidence 행렬 ──
        B = build_incidence_matrix(assets_list, pairs)   # shape (n_assets, n_pairs)
        if B.size == 0:
            # incidence 생성 실패 시 원본으로
            port_return = float((w_opt_original * daily_returns).sum())
            portfolio_returns.append(port_return)
            benchmark_returns.append(bmk_return)
            active_returns.append(port_return - bmk_return)
            used_dates.append(date)
            continue

        # ── 원본 active weights에서 pair per-leg 사이즈 역추정 ──
        #     (B @ x = w_active_original, x_i = per-leg size; long leg +x, short leg -x)
        x_original = reconstruct_pair_sizes(w_active_original.values, B, signals)

        # ── 리스크 단위와 per-leg cap 계산(★ 패치 핵심) ──
        #    날짜별 롤링 윈도우로 constraint_values(=risk unit) & cap_arr 반환
        constraint_values, cap_arr = risk_calc.calculate_position_caps(
            pairs=pairs,
            signals=signals,
            constraint_method=constraint_method,
            asof_date=date,
            kappa_mode=kappa_mode,  # "cash-aware"면 cash 페어는 leg_factor=1 기준으로 cap 산출
        )

        # ── Cap 적용(clip) ──
        if sizing_mode == "clip":
            x_constrained = np.sign(x_original) * np.minimum(
                np.abs(x_original), np.asarray(cap_arr, dtype=float)

            )
        else:
            x_constrained = np.sign(signals) * np.asarray(cap_arr, dtype=float)

                # ── 제약 적용된 active weights 복원 ──
        w_active_constrained = pd.Series(B @ x_constrained, index=assets_list)

        # ── 최종 포트폴리오 가중치(숏 허용, 합=1 강제 X) ──
        w_portfolio = (w_bmk + w_active_constrained).reindex(assets_list).fillna(0.0)

        # ── 일별 수익률 ──
        port_return = float((w_portfolio * daily_returns).sum())
        act_return = port_return - bmk_return

        portfolio_returns.append(port_return)
        benchmark_returns.append(bmk_return)
        active_returns.append(act_return)
        used_dates.append(date)

        # ── 로깅(진단/시각화용) ──
        for i, (pid, (la, sa)) in enumerate(zip(pair_ids, pairs)):
            abs_signal = int(abs(round(float(signals[i]))))
            # 허용손실 bp (클래스 내부 규칙을 그대로 사용)
            max_loss_bp = risk_calc.get_max_loss_bp(signals[i])

            # per-leg 포지션(소수)
            per_leg_orig = float(x_original[i])
            per_leg_cap  = float(cap_arr[i])
            per_leg_new  = float(x_constrained[i])

            # cash-aware면 한쪽 레그만 위험 → 총 익스포저 집계 시 leg_factor=1
            leg_factor = 1 if (kappa_mode == "cash-aware" and bool(is_cash_pair_flags[i])) else 2

            # 총 액티브(소수)와 리스크/손실
            total_active_abs = abs(per_leg_new) * leg_factor
            # constraint_values[i] = risk unit (예: z*σ) in decimal
            risk_unit = float(constraint_values[i])
            actual_loss = total_active_abs * abs(risk_unit)  # 소수 기준
            # bp 변환
            per_leg_orig_bp = per_leg_orig * 10_000.0
            per_leg_new_bp  = per_leg_new  * 10_000.0
            total_active_bp = total_active_abs * 10_000.0
            actual_loss_bp  = actual_loss * 10_000.0

            position_records.append({
                "Date": date,
                "Pair_ID": pid,
                "Pair": f"{la} vs {sa}",
                "Signal": float(signals[i]),
                "Signal_Abs": abs_signal,
                "Kappa_Mode": kappa_mode,
                "Is_Cash_Pair": bool(is_cash_pair_flags[i]),
                "Leg_Factor": leg_factor,  # 1 (cash-aware) or 2 (symmetric)
                "Risk_Unit": risk_unit,  # 소수 (예: 0.031 = 3.1%)
                "Constraint_Value_%": risk_unit * 100.0,  # % 표시용
                "Max_Loss_bp": float(max_loss_bp),  # 허용손실(bp)

                # --- per-leg 포지션(bp) 관련: 원본/캡/표준 이름 모두 기록 ---
                "Original_per_leg_bp": per_leg_orig_bp,
                "Capped_per_leg_bp": per_leg_new_bp,
                "Position_per_leg_bp": per_leg_new_bp,  # ← UI가 기대하는 표준 이름 (실제 적용치)

                # --- 총 익스포저/손실 (bp) ---
                "Total_Active_bp": total_active_bp,
                "Total_Notional_bp": total_active_bp,  # ← UI의 다른 표준 이름도 함께 기록
                "Actual_Loss_bp": actual_loss_bp,

                "Capped": bool(abs(per_leg_orig) > per_leg_cap),
                "Benchmark_Sum": float(w_bmk.sum()),
                "Portfolio_Sum": float(w_portfolio.sum()),
                "Active_Sum": float(w_active_constrained.sum()),
            })

    daily_returns_df = pd.DataFrame({
        "Portfolio_Return": portfolio_returns,
        "Benchmark_Return": benchmark_returns,
        "Active_Return": active_returns,
    }, index=pd.Index(used_dates, name="Date")).sort_index()

    position_changes_df = pd.DataFrame(position_records).sort_values(["Date", "Pair_ID"]) if position_records else pd.DataFrame()

    position_changes_df = pd.DataFrame(position_records).sort_values(
        ["Date", "Pair_ID"]) if position_records else pd.DataFrame()

    # === 컬럼 별칭/결측 보강 (과거 캐시/CSV 호환 목적) ===
    if not position_changes_df.empty:
        if "Position_per_leg_bp" not in position_changes_df.columns and "Capped_per_leg_bp" in position_changes_df.columns:
            position_changes_df["Position_per_leg_bp"] = position_changes_df["Capped_per_leg_bp"]

        if "Total_Notional_bp" not in position_changes_df.columns and "Total_Active_bp" in position_changes_df.columns:
            position_changes_df["Total_Notional_bp"] = position_changes_df["Total_Active_bp"]

    return daily_returns_df, position_changes_df

# =============================================================================
# TE/Vol 계산 함수 (리스크 제약 방법 통합)
# =============================================================================
def calculate_expected_tracking_error_with_constraint(
        views_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        returns_by_asset: pd.DataFrame,
        Wopt_last: pd.Series,
        Wbmk_last: pd.Series,
        constraint_method: str = "3Y_MDD",
        lookback_years: int = 3
) -> Tuple[float, pd.DataFrame]:
    """
    선택한 리스크 제약 방법에 따른 TE 계산

    Args:
        constraint_method: "3Y_MDD", "-3STD", "-2STD", "-1STD"

    Returns:
        (TE, debug_df): TE 값과 디버깅 정보 DataFrame
    """
    if cov_matrix.empty or weights_df.empty or returns_by_asset.empty:
        return 0.0, pd.DataFrame()

    if Wopt_last.empty or Wbmk_last.empty:
        return 0.0, pd.DataFrame()

    try:
        # 자산 리스트
        assets_list = [a for a in Wopt_last.index if a in returns_by_asset.columns]
        if len(assets_list) == 0:
            return 0.0, pd.DataFrame()

        Wact_current = (Wopt_last - Wbmk_last).reindex(assets_list).fillna(0.0)

        # Active views 필터링 (Signal != 0)
        active_views = views_df[views_df['Signal'] != 0].copy()

        if active_views.empty:
            # Signal이 없으면 현재 TE 반환
            cov = cov_matrix.reindex(index=assets_list, columns=assets_list).fillna(0.0).values
            w = Wact_current.values
            te_variance = float(w @ cov @ w)
            te = float(np.sqrt(max(0.0, te_variance)))
            return te, pd.DataFrame()

        # Pair 정보 추출
        pairs = [(str(row['Long_Asset']), str(row['Short_Asset'])) for _, row in active_views.iterrows()]
        signals = active_views['Signal'].astype(float).values
        pair_ids = active_views.get('Pair_ID', range(len(pairs))).values

        # Incidence matrix 생성
        B = build_incidence_matrix(assets_list, pairs)
        if B.size == 0:
            cov = cov_matrix.reindex(index=assets_list, columns=assets_list).fillna(0.0).values
            w = Wact_current.values
            te_variance = float(w @ cov @ w)
            te = float(np.sqrt(max(0.0, te_variance)))
            return te, pd.DataFrame()

        # 리스크 제약 계산
        risk_calc = RiskConstraintCalculator(returns_by_asset, lookback_years=lookback_years)
        constraint_values, cap_arr = risk_calc.calculate_position_caps(pairs, signals, constraint_method)

        # Signal 방향 적용
        x_pair = np.sign(signals) * cap_arr

        # Active weights 재구성
        Wact_new = pd.Series(B @ x_pair, index=assets_list)

        # TE 계산
        cov = cov_matrix.reindex(index=assets_list, columns=assets_list).fillna(0.0).values
        w = Wact_new.values
        te_variance = float(w @ cov @ w)
        te = float(np.sqrt(max(0.0, te_variance)))

        # Signal에 따른 최대 손실 허용치
        max_loss_per_pair = np.zeros(len(pairs))
        for i, signal in enumerate(signals):
            abs_signal = abs(signal)
            if abs_signal >= 2.0:
                max_loss_per_pair[i] = 0.15
            elif abs_signal >= 1.0:
                max_loss_per_pair[i] = 0.10
            else:
                max_loss_per_pair[i] = 0.10 + (abs_signal) * 0.05

        # 디버깅 정보 생성
        constraint_col_name = {
            "3Y_MDD": "MDD_%",
            "-3STD": "-3STD_%",
            "-2STD": "-2STD_%",
            "-1STD": "-1STD_%"
        }[constraint_method]

        debug_df = pd.DataFrame({
            'Pair_ID': pair_ids,
            'Pair': [f"{p[0]} vs {p[1]}" for p in pairs],
            'Signal': signals,
            constraint_col_name: (constraint_values * 100).round(2),
            'Max_Loss_bp': max_loss_per_pair.round(3),
            'Position_bp': (x_pair * 10000).round(3),
            'Actual_Loss_bp': (np.abs(x_pair) * np.abs(constraint_values) * 10000).round(3)
        })

        return te, debug_df

    except Exception as e:
        st.warning(f"TE 계산 오류: {e}")
        return 0.0, pd.DataFrame()


def calculate_expected_volatility_with_constraint(
        views_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        returns_by_asset: pd.DataFrame,
        Wopt_last: pd.Series,
        Wbmk_last: pd.Series,
        constraint_method: str = "3Y_MDD",
        lookback_years: int = 3
) -> float:
    """
    선택한 리스크 제약 방법에 따른 변동성 계산

    중요: Benchmark가 0이면 Portfolio Vol = TE와 같아야 함
    """
    if cov_matrix.empty or returns_by_asset.empty:
        return 0.0

    if Wopt_last.empty or Wbmk_last.empty:
        return 0.0

    try:
        assets_list = [a for a in Wopt_last.index if a in returns_by_asset.columns]
        if len(assets_list) == 0:
            return 0.0

        # TE 계산에서 Active weights를 가져옴
        te, debug_df = calculate_expected_tracking_error_with_constraint(
            views_df, weights_df, cov_matrix, returns_by_asset,
            Wopt_last, Wbmk_last, constraint_method, lookback_years
        )

        # Benchmark weights 확인
        Wbmk = Wbmk_last.reindex(assets_list).fillna(0.0)
        bm_sum = Wbmk.sum()

        # Benchmark가 거의 0이면 (100% Cash), Vol = TE
        if abs(bm_sum) < 0.001:
            return te

        # Benchmark가 있는 경우 Portfolio Vol 계산
        # Active weights 재구성
        Wact_new = pd.Series(0.0, index=assets_list)
        if not debug_df.empty:
            active_views = views_df[views_df['Signal'] != 0].copy()
            if not active_views.empty:
                pairs = [(str(row['Long_Asset']), str(row['Short_Asset']))
                         for _, row in active_views.iterrows()]
                B = build_incidence_matrix(assets_list, pairs)
                if B.size > 0:
                    x_pair = debug_df['Position_bp'].values / 10000
                    Wact_new = pd.Series(B @ x_pair, index=assets_list)

        # Portfolio = Benchmark + Active
        Wopt_new = Wbmk + Wact_new

        # 음수 제거 및 정규화
        Wopt_new = np.maximum(Wopt_new, 0)
        s = Wopt_new.sum()
        if s > 0:
            Wopt_new = Wopt_new / s
        else:
            return te

        # Portfolio Vol 계산
        cov = cov_matrix.reindex(index=assets_list, columns=assets_list).fillna(0.0).values
        portfolio_variance = float(Wopt_new.values @ cov @ Wopt_new.values)
        portfolio_vol = float(np.sqrt(max(0.0, portfolio_variance)))

        return portfolio_vol

    except Exception as e:
        st.warning(f"Vol 계산 오류: {e}")
        return 0.0


def calculate_expected_volatility(views_df: pd.DataFrame, weights_df: pd.DataFrame, cov_matrix: pd.DataFrame) -> float:
    """포트폴리오 변동성 계산 (연율, 소수) - Fallback용"""
    if cov_matrix.empty or weights_df.empty:
        return 0.0
    try:
        assets = weights_df["Asset"].astype(str).tolist()
        cov = cov_matrix.reindex(index=assets, columns=assets).fillna(0.0).values

        opt_weights = calculate_portfolio_weights(views_df, weights_df)
        if len(opt_weights) == 0:
            return 0.0

        portfolio_variance = float(opt_weights @ cov @ opt_weights)
        portfolio_vol = float(np.sqrt(max(0.0, portfolio_variance)))
        return portfolio_vol
    except Exception as e:
        st.warning(f"Vol 계산 오류: {e}")
        return 0.0


def calculate_expected_tracking_error(views_df: pd.DataFrame, weights_df: pd.DataFrame,
                                      cov_matrix: pd.DataFrame) -> float:
    """TE 계산 (Fallback용)"""
    if cov_matrix.empty or weights_df.empty:
        return 0.0

    try:
        assets = weights_df["Asset"].astype(str).tolist()
        n_assets = len(assets)

        # Benchmark weights
        if "Benchmark_Weight" not in weights_df.columns:
            return 0.0

        bm_weights = pd.to_numeric(weights_df["Benchmark_Weight"], errors="coerce").fillna(0.0).values

        # Optimal weights
        opt_weights = calculate_portfolio_weights(views_df, weights_df)
        if len(opt_weights) == 0:
            return 0.0

        # Active weights
        active_weights = opt_weights - bm_weights

        # TE 계산
        cov = cov_matrix.reindex(index=assets, columns=assets).fillna(0.0).values
        te_variance = float(active_weights @ cov @ active_weights)
        te = float(np.sqrt(max(0.0, te_variance)))

        return te
    except Exception as e:
        st.warning(f"TE 계산 오류: {e}")
        return 0.0


def compute_te_from_active_direct(active_weights: np.ndarray, cov_matrix: pd.DataFrame, assets: List[str]) -> float:
    """Active weight로부터 직접 TE 계산"""
    if cov_matrix.empty or len(active_weights) == 0:
        return 0.0
    try:
        cov = cov_matrix.reindex(index=assets, columns=assets).fillna(0.0).values
        te_variance = float(active_weights @ cov @ active_weights)
        te_annual = float(np.sqrt(max(0.0, te_variance)))
        return te_annual
    except Exception:
        return 0.0


# =============================================================================
# 시장 데이터 로더
# =============================================================================
def load_market_returns_csv(csv_path: str, asset_names: list, excel_path: str = None) -> pd.DataFrame:
    """시장 수익률 데이터 로드"""
    CASH_ANNUAL_RETURN = 0.025
    TRADING_DAYS = 252

    if not os.path.isfile(csv_path):
        st.warning(f"시장 데이터 CSV가 없습니다: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, encoding='utf-8')
    dcol = None
    for c in df.columns:
        if _norm(c) in {"date", "날짜"}:
            dcol = c
            break

    if dcol is None:
        st.warning("CSV에서 날짜 컬럼을 찾지 못했습니다.")
        return pd.DataFrame()

    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol).set_index(dcol)

    # Ticker 매핑
    ticker_mapping = {}
    if excel_path and os.path.isfile(excel_path):
        try:
            xls = pd.ExcelFile(excel_path, engine="openpyxl")
            if "Asset_Universe" in xls.sheet_names:
                au = pd.read_excel(xls, "Asset_Universe")
                au_cols = {c.lower().replace('_', '').replace(' ', ''): c for c in au.columns}
                col_name = None
                col_ticker = None
                for possible_name in ["assetname", "asset", "name"]:
                    if possible_name in au_cols:
                        col_name = au_cols[possible_name]
                        break
                for possible_ticker in ["bloombergticker", "ticker", "bloomberg", "symbol"]:
                    if possible_ticker in au_cols:
                        col_ticker = au_cols[possible_ticker]
                        break
                if col_name and col_ticker:
                    for _, row in au.iterrows():
                        asset_name = str(row[col_name]).strip()
                        ticker = str(row[col_ticker]).strip()
                        ticker_mapping[asset_name] = ticker
                    st.success(f"✅ Asset_Universe에서 {len(ticker_mapping)}개 Bloomberg Ticker 매핑 로드")
        except Exception as e:
            st.error(f"Asset_Universe 로드 실패: {e}")

    ret = pd.DataFrame(index=df.index)
    matched, not_matched, cash_assets = [], [], []

    for asset_name in asset_names:
        if is_cash_asset(asset_name):
            daily_return = CASH_ANNUAL_RETURN / TRADING_DAYS
            ret[asset_name] = daily_return
            cash_assets.append(asset_name)
            matched.append(f"✅ {asset_name} ← {CASH_ANNUAL_RETURN * 100:.2f}% 연간 수익률 (Cash)")
            continue

        ticker = ticker_mapping.get(asset_name)
        if ticker:
            found = False
            if ticker in df.columns:
                ret[asset_name] = df[ticker]
                matched.append(f"✅ {asset_name} ← {ticker}")
                found = True
            else:
                for col in df.columns:
                    if col.lower() == ticker.lower():
                        ret[asset_name] = df[col]
                        matched.append(f"✅ {asset_name} ← {col}")
                        found = True
                        break
                if not found:
                    ticker_base = ticker.replace(" Index", "").replace(" Comdty", "").replace(" Curncy", "")
                    for col in df.columns:
                        col_base = col.replace(" Index", "").replace(" Comdty", "").replace(" Curncy", "")
                        if col_base.lower() == ticker_base.lower():
                            ret[asset_name] = df[col]
                            matched.append(f"✅ {asset_name} ← {col}")
                            found = True
                            break
            if not found:
                not_matched.append(f"❌ {asset_name} (Ticker: {ticker})")
        else:
            if asset_name in df.columns:
                ret[asset_name] = df[asset_name]
                matched.append(f"✅ {asset_name} ← {asset_name} (직접 매칭)")
            else:
                not_matched.append(f"❌ {asset_name} (Ticker 없음)")

    if matched:
        with st.expander(f"✅ 매칭 성공 ({len(matched)}개)", expanded=False):
            for m in matched:
                st.text(m)
    if not_matched:
        with st.expander(f"❌ 매칭 실패 ({len(not_matched)}개)", expanded=False):
            for nm in not_matched:
                st.text(nm)

    if ret.empty:
        st.error("⚠️ 매칭된 자산이 없습니다.")
        return ret

    for col in ret.columns:
        if col not in cash_assets:
            ret[col] = ret[col].ffill().bfill().pct_change()

    ret = ret.dropna()

    if cash_assets:
        st.info(f"💵 Cash 자산 ({', '.join(cash_assets)}): 연간 {CASH_ANNUAL_RETURN * 100:.2f}% 수익률로 처리")

    st.success(f"📈 {len(ret.columns)}개 자산의 수익률 데이터 생성 완료")
    return ret


# =============================================================================
# Excel Views 로더
# =============================================================================
def load_views_from_excel(excel_path: str) -> pd.DataFrame:
    """Excel에서 Views 정보 로드"""
    if not os.path.isfile(excel_path):
        return pd.DataFrame()
    try:
        xls = pd.ExcelFile(excel_path, engine="openpyxl")
        vt = pd.read_excel(xls, "Views_Timeline")
        pddef = pd.read_excel(xls, "Pairs_Definition")
        au = pd.read_excel(xls, "Asset_Universe")
    except Exception as e:
        st.warning(f"엑셀 로딩 실패: {e}")
        return pd.DataFrame()

    au_cols = {c.lower(): c for c in au.columns}
    col_id = au_cols.get("asset_id") or au_cols.get("id")
    col_nm = au_cols.get("asset_name") or au_cols.get("name")
    if col_id is None or col_nm is None:
        st.warning("Asset_Universe 시트에 Asset_ID/Asset_Name 필수")
        return pd.DataFrame()
    id2name = dict(zip(au[col_id], au[col_nm].astype(str)))

    pd_cols = {c.lower(): c for c in pddef.columns}
    col_pair = pd_cols.get("pair_id") or pd_cols.get("pair")
    col_long = pd_cols.get("long_asset_id") or pd_cols.get("long_id")
    col_short = pd_cols.get("short_asset_id") or pd_cols.get("short_id")
    if col_pair is None or col_long is None or col_short is None:
        st.warning("Pairs_Definition 시트에 Pair_ID/Long_Asset_ID/Short_Asset_ID 필수")
        return pd.DataFrame()

    mp = pddef[[col_pair, col_long, col_short]].copy()
    mp["Long_Asset"] = mp[col_long].map(id2name).astype(str)
    mp["Short_Asset"] = mp[col_short].map(id2name).astype(str)
    mp = mp.rename(columns={col_pair: "Pair_ID"}).drop(columns=[col_long, col_short])

    vt_cols = {c.lower(): c for c in vt.columns}
    need = ["pair_id", "signal"]
    if not all(k in vt_cols for k in need):
        st.warning("Views_Timeline 시트에 Pair_ID/Signal 필수")
        return pd.DataFrame()

    v = vt.rename(columns={vt_cols["pair_id"]: "Pair_ID", vt_cols["signal"]: "Signal"}).copy()
    v["Signal"] = pd.to_numeric(v["Signal"], errors="coerce").fillna(0.0)

    if "start_date" in vt_cols:
        v["Start_Date"] = _safe_dt(v[vt_cols["start_date"]])
    else:
        v["Start_Date"] = pd.NaT

    if "end_date" in vt_cols:
        v["End_Date"] = _safe_dt(v[vt_cols["end_date"]])
    else:
        v["End_Date"] = pd.NaT

    if "status" in vt_cols:
        v["Status"] = v[vt_cols["status"]].astype(str)
    else:
        v["Status"] = "Active"

    out = v.merge(mp, on="Pair_ID", how="left", validate="many_to_one")
    out = out.dropna(subset=["Long_Asset", "Short_Asset"]).copy()
    return out[["Pair_ID", "Long_Asset", "Short_Asset", "Signal", "Start_Date", "End_Date", "Status"]]


def load_active_views_from_timeline(timeline_csv: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """타임라인에서 활성 뷰 추출"""
    if timeline_csv is None or timeline_csv.empty:
        return pd.DataFrame()

    v = timeline_csv.copy()
    for c in ["Start_Date", "End_Date"]:
        if c in v.columns:
            v[c] = _safe_dt(v[c])
        else:
            v[c] = pd.NaT

    v["Signal"] = pd.to_numeric(v.get("Signal", 0.0), errors="coerce").fillna(0.0)
    mask = (v["Start_Date"].fillna(pd.Timestamp.min) <= asof) & (v["End_Date"].fillna(pd.Timestamp.max) >= asof)
    v = v[mask].copy()

    keep_cols = [c for c in
                 ["Pair_ID", "Long_Asset", "Short_Asset", "Signal", "Start_Date", "End_Date", "Reason", "Status"] if
                 c in v.columns]
    return v[keep_cols]


# =============================================================================
# CSV 데이터 로더
# =============================================================================
@st.cache_data
def load_csv_data(data_dir: str):
    """CSV 파일들 로드"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {data_dir}")

    files = os.listdir(data_dir)
    stem_to_file = {}
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() in [".csv", ".xlsx", ".xls"]:
            stem_to_file[name.lower()] = f

    def _find(stem: str):
        cand = stem_to_file.get(stem.lower())
        return None if not cand else os.path.join(data_dir, cand)

    def _read_table(path: str, index_col=None) -> pd.DataFrame:
        _, ext = os.path.splitext(path)
        try:
            if ext.lower() == ".csv":
                df = pd.read_csv(path, index_col=index_col, encoding="utf-8-sig")
            else:
                df = pd.read_excel(path, index_col=index_col, engine="openpyxl")
            return df
        except Exception as e:
            st.warning(f"파일 읽기 실패: {path} - {e}")
            return pd.DataFrame()

    data = {}
    missing = []

    # 필수 파일
    required_stems = [
        "portfolio_weights",
        "performance_metrics",
        "expected_returns",
        "asset_rankings",
        "covariance_matrix",
    ]
    for stem in required_stems:
        p = _find(stem)
        if p:
            idx_col = 0 if stem == "covariance_matrix" else None
            df = _read_table(p, index_col=idx_col)
            data[stem] = df
        else:
            missing.append(stem)
            data[stem] = pd.DataFrame()

    # 선택 파일
    optional_stems = [
        "active_views",
        "attractiveness_scores",
        "daily_returns_series",
        "daily_weights_optimal",
        "daily_weights_benchmark",
        "weights_checkpoints",
        "rebalance_log",
        "rebalance_calendar",
        "weight_history",
        "view_timeline_history",
        "attribution_report",
        "attribution_summary",
        "pair_mdd_report",
        "pair_constraints",
    ]
    for opt_stem in optional_stems:
        p = _find(opt_stem)
        if p:
            if opt_stem in [
                "daily_returns_series", "daily_weights_optimal", "daily_weights_benchmark",
                "weights_checkpoints", "weight_history"
            ]:
                df = _read_table(p, index_col=0)
                if not df.empty:
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception:
                        pass
                data[opt_stem] = df
            else:
                data[opt_stem] = _read_table(p, index_col=None)
        else:
            data[opt_stem] = pd.DataFrame()

    # 날짜 변환
    if "rebalance_log" in data and not data["rebalance_log"].empty:
        if "Rebalance_Date" in data["rebalance_log"].columns:
            try:
                data["rebalance_log"]["Rebalance_Date"] = pd.to_datetime(data["rebalance_log"]["Rebalance_Date"],
                                                                         errors='coerce')
            except Exception:
                pass

    if "rebalance_calendar" in data and not data["rebalance_calendar"].empty:
        if "Rebalance_Date" in data["rebalance_calendar"].columns:
            try:
                data["rebalance_calendar"]["Rebalance_Date"] = pd.to_datetime(
                    data["rebalance_calendar"]["Rebalance_Date"], errors='coerce')
            except Exception:
                pass

    # 수치 변환
    if "performance_metrics" in data and not data["performance_metrics"].empty:
        try:
            pm = data["performance_metrics"].copy()
            pm.columns = [str(c).strip() for c in pm.columns]
            if "Value" in pm.columns:
                pm["Value"] = pm["Value"].apply(_to_float)
            data["performance_metrics"] = pm
        except Exception:
            pass

    if "portfolio_weights" in data and not data["portfolio_weights"].empty:
        try:
            pw = data["portfolio_weights"].copy()
            for col in ["Optimal_Weight", "Benchmark_Weight", "Active_Weight"]:
                if col in pw.columns:
                    pw[col] = pw[col].apply(_to_float)
            data["portfolio_weights"] = pw
        except Exception:
            pass

    # 로드 정보 표시
    loaded = [k for k in data.keys() if not data[k].empty]
    if loaded:
        st.sidebar.info(
            "📂 로드된 파일:\n" +
            "\n".join([f"✅ {k}" for k in loaded[:8]]) +
            (f"\n... 외 {len(loaded) - 8}개" if len(loaded) > 8 else "")
        )
    if missing:
        st.sidebar.warning("⚠️ 누락된 필수 파일:\n" + "\n".join([f"❌ {m}" for m in missing[:3]]))

    return data


# =============================================================================
# 리스크 제약 방법에 따른 일별 수익률 재계산
# =============================================================================


# =============================================================================
# 일별 성과 시각화 함수 (수정)
# =============================================================================
def display_daily_returns(
        daily_returns_original: pd.DataFrame,
        daily_returns_constrained: pd.DataFrame,
        position_changes_df: pd.DataFrame,
        constraint_method: str,
        inception_date: pd.Timestamp = None
):
    """
    일별 수익률 시각화 (원본 vs 제약 적용 비교)

    Args:
        daily_returns_original: 원본 일별 수익률
        daily_returns_constrained: 제약 적용 일별 수익률
        position_changes_df: 포지션 변화 추적
        constraint_method: 적용된 제약 방법
        inception_date: Inception Date
    """
    if daily_returns_original.empty and daily_returns_constrained.empty:
        st.info("일별 수익률 데이터가 없습니다.")
        return

    # Inception Date 필터링
    if inception_date is not None:
        if not daily_returns_original.empty:
            daily_returns_original = daily_returns_original[
                daily_returns_original.index >= inception_date
                ].copy()
        if not daily_returns_constrained.empty:
            daily_returns_constrained = daily_returns_constrained[
                daily_returns_constrained.index >= inception_date
                ].copy()

        if daily_returns_original.empty and daily_returns_constrained.empty:
            st.warning(f"⚠️ {inception_date.date()} 이후 데이터가 없습니다.")
            return

    try:
        # 제약 방법 표시
        constraint_display_map = {
            "3Y_MDD": "3년 MDD",
            "-3STD": "-3 표준편차 (3M)",
            "-2STD": "-2 표준편차 (3M)",
            "-1STD": "-1 표준편차 (3M)"
        }

        st.info(f"🎯 적용된 리스크 제약: **{CONSTRAINT_DISPLAY_MAP.get(constraint_method, constraint_method)}**")

        # 탭으로 구분
        tab1, tab2, tab3 = st.tabs(["📊 성과 비교", "📈 누적 수익률", "📋 포지션 변화"])

        # ===== Tab 1: 성과 비교 =====
        with tab1:
            st.subheader("📊 성과 지표 비교: 원본 vs 제약 적용")

            # 원본과 제약 적용 데이터가 모두 있는 경우
            if not daily_returns_original.empty and not daily_returns_constrained.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📉 원본 (CSV 데이터)")
                    metrics_cols_orig = st.columns(3)

                    for idx, col_name in enumerate(['Portfolio_Return', 'Benchmark_Return', 'Active_Return']):
                        if col_name in daily_returns_original.columns:
                            metrics = calculate_annualized_metrics(daily_returns_original[col_name])

                            display_name = {
                                'Portfolio_Return': '포트폴리오',
                                'Benchmark_Return': '벤치마크',
                                'Active_Return': '초과수익'
                            }[col_name]

                            with metrics_cols_orig[idx]:
                                st.markdown(f"**{display_name}**")
                                st.metric("누적", f"{metrics['cumulative_return'] * 10000:.3f}bp")
                                st.metric("연율화", f"{metrics['annualized_return'] * 10000:.3f}bp")
                                st.metric("변동성", f"{metrics['annualized_volatility'] * 10000:.3f}bp")

                with col2:
                    st.markdown(f"### 📈 제약 적용 ({constraint_display_map[constraint_method]})")
                    metrics_cols_const = st.columns(3)

                    for idx, col_name in enumerate(['Portfolio_Return', 'Benchmark_Return', 'Active_Return']):
                        if col_name in daily_returns_constrained.columns:
                            metrics = calculate_annualized_metrics(daily_returns_constrained[col_name])

                            # 원본과 비교
                            if col_name in daily_returns_original.columns:
                                metrics_orig = calculate_annualized_metrics(daily_returns_original[col_name])
                                delta_cum = (metrics['cumulative_return'] - metrics_orig['cumulative_return']) * 10000
                                delta_ann = (metrics['annualized_return'] - metrics_orig['annualized_return']) * 10000
                                delta_vol = (metrics['annualized_volatility'] - metrics_orig[
                                    'annualized_volatility']) * 10000
                            else:
                                delta_cum = delta_ann = delta_vol = 0.0

                            display_name = {
                                'Portfolio_Return': '포트폴리오',
                                'Benchmark_Return': '벤치마크',
                                'Active_Return': '초과수익'
                            }[col_name]

                            with metrics_cols_const[idx]:
                                st.markdown(f"**{display_name}**")
                                st.metric("누적", f"{metrics['cumulative_return'] * 10000:.3f}bp",
                                          delta=f"{delta_cum:+.3f}bp")
                                st.metric("연율화", f"{metrics['annualized_return'] * 10000:.3f}bp",
                                          delta=f"{delta_ann:+.3f}bp")
                                st.metric("변동성", f"{metrics['annualized_volatility'] * 10000:.3f}bp",
                                          delta=f"{delta_vol:+.3f}bp")

            # 제약 적용 데이터만 있는 경우
            elif not daily_returns_constrained.empty:
                st.markdown(f"### 📈 제약 적용 ({constraint_display_map[constraint_method]})")
                metrics_cols = st.columns(3)

                for idx, col_name in enumerate(['Portfolio_Return', 'Benchmark_Return', 'Active_Return']):
                    if col_name in daily_returns_constrained.columns:
                        metrics = calculate_annualized_metrics(daily_returns_constrained[col_name])

                        display_name = {
                            'Portfolio_Return': '포트폴리오',
                            'Benchmark_Return': '벤치마크',
                            'Active_Return': '초과수익'
                        }[col_name]

                        with metrics_cols[idx]:
                            st.markdown(f"**{display_name}**")
                            st.metric("누적 수익률", f"{metrics['cumulative_return'] * 10000:.3f}bp")
                            st.metric("연율화 수익률", f"{metrics['annualized_return'] * 10000:.3f}bp")
                            st.metric("연율화 변동성", f"{metrics['annualized_volatility'] * 10000:.3f}bp")
                            st.caption(f"거래일: {metrics['n_days']}일")

        # ===== Tab 2: 누적 수익률 =====
        with tab2:
            st.subheader("📈 누적 수익률 비교")

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("누적 수익률 비교", "일별 초과수익 비교"),
                vertical_spacing=0.12,
                row_heights=[0.6, 0.4]
            )

            # 원본 데이터
            if not daily_returns_original.empty:
                cum_returns_orig = (1 + daily_returns_original).cumprod() - 1

                for col in ["Portfolio_Return", "Benchmark_Return", "Active_Return"]:
                    if col in cum_returns_orig.columns:
                        name_map = {
                            "Portfolio_Return": "포트폴리오 (원본)",
                            "Benchmark_Return": "벤치마크 (원본)",
                            "Active_Return": "초과수익 (원본)"
                        }
                        color_map = {
                            "Portfolio_Return": "lightblue",
                            "Benchmark_Return": "lightgray",
                            "Active_Return": "lightgreen"
                        }
                        fig.add_trace(
                            go.Scatter(
                                x=cum_returns_orig.index,
                                y=cum_returns_orig[col] * 10000,
                                name=name_map[col],
                                line=dict(color=color_map[col], width=2, dash='dot'),
                                hovertemplate="%{y:.3f}bp<extra></extra>"
                            ),
                            row=1, col=1
                        )

                # 일별 초과수익 (원본)
                if "Active_Return" in daily_returns_original.columns:
                    colors = ["lightgreen" if r > 0 else "lightcoral"
                              for r in daily_returns_original["Active_Return"]]
                    fig.add_trace(
                        go.Bar(
                            x=daily_returns_original.index,
                            y=daily_returns_original["Active_Return"] * 10000,
                            name="초과수익 (원본)",
                            marker_color=colors,
                            opacity=0.5,
                            hovertemplate="%{y:.3f}bp<extra></extra>"
                        ),
                        row=2, col=1
                    )

            # 제약 적용 데이터
            if not daily_returns_constrained.empty:
                cum_returns_const = (1 + daily_returns_constrained).cumprod() - 1

                for col in ["Portfolio_Return", "Benchmark_Return", "Active_Return"]:
                    if col in cum_returns_const.columns:
                        name_map = {
                            "Portfolio_Return": "포트폴리오 (제약)",
                            "Benchmark_Return": "벤치마크 (제약)",
                            "Active_Return": "초과수익 (제약)"
                        }
                        color_map = {
                            "Portfolio_Return": "blue",
                            "Benchmark_Return": "gray",
                            "Active_Return": "green"
                        }
                        fig.add_trace(
                            go.Scatter(
                                x=cum_returns_const.index,
                                y=cum_returns_const[col] * 10000,
                                name=name_map[col],
                                line=dict(color=color_map[col], width=3),
                                hovertemplate="%{y:.3f}bp<extra></extra>"
                            ),
                            row=1, col=1
                        )

                # 일별 초과수익 (제약)
                if "Active_Return" in daily_returns_constrained.columns:
                    colors = ["green" if r > 0 else "red"
                              for r in daily_returns_constrained["Active_Return"]]
                    fig.add_trace(
                        go.Bar(
                            x=daily_returns_constrained.index,
                            y=daily_returns_constrained["Active_Return"] * 10000,
                            name="초과수익 (제약)",
                            marker_color=colors,
                            hovertemplate="%{y:.3f}bp<extra></extra>"
                        ),
                        row=2, col=1
                    )

            fig.update_xaxes(title_text="", row=2, col=1)
            fig.update_yaxes(title_text="수익률 (bp)", row=1, col=1, tickformat=".3f")
            fig.update_yaxes(title_text="초과수익 (bp)", row=2, col=1, tickformat=".3f")
            fig.update_layout(
                height=700,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # 폰트 크기 적용
            fig = apply_chart_font_settings(fig)

            st.plotly_chart(fig, use_container_width=True)

        # Tab 3: 포지션 변화
        with tab3:
            st.subheader("📋 Pair별 포지션 변화")

            if position_changes_df.empty:
                st.info("포지션 변화 데이터가 없습니다.")
            else:
                # 최근 날짜 선택
                unique_dates = sorted(position_changes_df['Date'].unique(), reverse=True)

                if len(unique_dates) > 0:
                    selected_date = st.selectbox(
                        "날짜 선택",
                        unique_dates,
                        format_func=lambda x: x.strftime('%Y-%m-%d')
                    )

                    # 선택된 날짜의 포지션
                    daily_positions = position_changes_df[
                        position_changes_df['Date'] == selected_date
                        ].copy()

                    if not daily_positions.empty:
                        # 포지션 크기 순으로 정렬
                        daily_positions['Abs_Position'] = daily_positions['Position_per_leg_bp'].abs()  # ★ 수정
                        daily_positions = daily_positions.sort_values('Abs_Position', ascending=False)

                        # 표시
                        st.markdown(f"#### {selected_date.strftime('%Y-%m-%d')} 포지션")

                        # 요약 통계
                        # 요약 통계
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Active Pairs", len(daily_positions))
                        with col2:
                            avg_pos = daily_positions['Position_per_leg_bp'].abs().mean()
                            st.metric("평균 Per-Leg 포지션", f"{avg_pos:.3f}bp")
                        with col3:
                            avg_notional = daily_positions['Total_Notional_bp'].mean()
                            st.metric("평균 Total Notional", f"{avg_notional:.3f}bp")
                        with col4:
                            avg_loss = daily_positions['Max_Loss_bp'].mean()
                            st.metric("평균 최대 손실", f"{avg_loss:.3f}bp")

                        # 상세 테이블
                        # 상세 테이블
                        display_cols = ['Pair_ID', 'Pair', 'Signal', 'Position_per_leg_bp',
                                        'Total_Notional_bp', 'Constraint_Value_%', 'Max_Loss_bp',
                                        'Actual_Loss_bp']
                        display_df = daily_positions[display_cols].copy()

                        # 포맷팅
                        display_df['Signal'] = display_df['Signal'].apply(lambda x: f"{x:.1f}")
                        display_df['Position_per_leg_bp'] = display_df['Position_per_leg_bp'].apply(
                            lambda x: f"{x:.3f}")
                        display_df['Total_Notional_bp'] = display_df['Total_Notional_bp'].apply(lambda x: f"{x:.3f}")
                        display_df['Constraint_Value_%'] = display_df['Constraint_Value_%'].apply(lambda x: f"{x:.2f}%")
                        display_df['Max_Loss_bp'] = display_df['Max_Loss_bp'].apply(lambda x: f"{x:.3f}")
                        display_df['Actual_Loss_bp'] = display_df['Actual_Loss_bp'].apply(lambda x: f"{x:.3f}")

                        st.dataframe(display_df, use_container_width=True)

                        # 차트
                        # 차트
                        fig_pos = go.Figure()

                        fig_pos.add_trace(go.Bar(
                            x=daily_positions['Pair'],
                            y=daily_positions['Position_per_leg_bp'],  # ⬅️ 변경
                            marker_color=['green' if p > 0 else 'red'
                                          for p in daily_positions['Position_per_leg_bp']],  # ⬅️ 변경
                        ))

                        fig_pos.update_layout(
                            title=f"{selected_date.strftime('%Y-%m-%d')} Pair별 Per-Leg 포지션",
                            xaxis_title="Pair",
                            yaxis_title="Per-Leg 포지션 (bp)",
                            yaxis_tickformat=".3f",
                            height=400
                        )

                        fig_pos = apply_chart_font_settings(fig_pos)

                        st.plotly_chart(fig_pos, use_container_width=True)

                # 시계열 포지션 변화
                st.markdown("---")
                st.subheader("📈 포지션 시계열 변화")

                # Pair 선택
                unique_pairs = sorted(position_changes_df['Pair'].unique())
                selected_pairs = st.multiselect(
                    "Pair 선택",
                    unique_pairs,
                    default=unique_pairs[:5] if len(unique_pairs) >= 5 else unique_pairs
                )

                if selected_pairs:
                    fig_ts = go.Figure()

                    for pair in selected_pairs:
                        pair_data = position_changes_df[position_changes_df['Pair'] == pair]

                        fig_ts.add_trace(go.Scatter(
                            x=pair_data['Date'],
                            y=pair_data['Position_per_leg_bp'],
                            mode='lines+markers',
                            name=pair,
                            line=dict(width=2),
                            marker=dict(size=6),
                            hovertemplate=f"{pair}<br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}bp<extra></extra>"
                        ))

                    fig_ts.update_layout(
                        title="Pair별 Per-Leg 포지션 시계열",
                        xaxis_title="날짜",
                        yaxis_title="Per-Leg 포지션 (bp)",
                        yaxis_tickformat=".3f",
                        height=500,
                        hovermode='x unified'
                    )

                    fig_ts.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                    fig_ts = apply_chart_font_settings(fig_ts)

                    st.plotly_chart(fig_ts, use_container_width=True)

        # Inception Date 정보 표시
        if inception_date is not None:
            if not daily_returns_constrained.empty:
                n_days = len(daily_returns_constrained)
                st.info(f"📅 Inception Date: {inception_date.date()} (이후 {n_days}일)")

    except Exception as e:
        st.warning(f"차트 생성 중 오류 발생: {str(e)}")


def display_rebalance_log(rebalance_log_df: pd.DataFrame):
    """리밸런싱 로그 표시"""
    if rebalance_log_df.empty:
        st.info("리밸런싱 로그가 없습니다.")
        return

    try:
        col1, col2, col3 = st.columns(3)
        with col1:
            total_rebal = len(rebalance_log_df)
            st.metric("총 리밸런싱 횟수", f"{total_rebal}회")
        with col2:
            if "Reason" in rebalance_log_df.columns:
                view_changes = rebalance_log_df["Reason"].str.contains("View", na=False).sum()
                st.metric("View 변경 리밸런싱", f"{view_changes}회")
        with col3:
            if "N_Views" in rebalance_log_df.columns:
                avg_views = rebalance_log_df["N_Views"].mean()
                st.metric("평균 Active Views", f"{avg_views:.1f}개")

        st.subheader("최근 리밸런싱 이벤트")
        for _, row in rebalance_log_df.tail(10).iterrows():
            reason_color = "#ff7f0e" if "View" in str(row.get("Reason", "")) else "#1f77b4"
            date_str = "N/A"
            if pd.notna(row.get('Rebalance_Date')):
                try:
                    date_str = row['Rebalance_Date'].strftime('%Y-%m-%d')
                except Exception:
                    date_str = str(row['Rebalance_Date'])

            te_bp = row.get('TE_ann', 0) * 10000
            st.markdown(f"""
            <div class="rebalance-log" style="border-left-color: {reason_color}">
                <b>{date_str}</b> - {row.get('Reason', 'N/A')}<br>
                <small>Views: {row.get('N_Views', 0):.0f} | TE: {te_bp:.3f}bp | Regime: {row.get('Regime_Score', 0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"리밸런싱 로그 표시 중 오류: {str(e)}")


def display_checkpoint_weights(checkpoints_df: pd.DataFrame, assets: list = None):
    """체크포인트 가중치 표시 - 폰트 크기 증가"""
    if checkpoints_df.empty:
        st.info("체크포인트 가중치 데이터가 없습니다.")
        return

    try:
        active_cols = [col for col in checkpoints_df.columns if col.endswith('_Active')]
        if not active_cols:
            st.warning("Active weight 데이터가 없습니다.")
            return

        if assets is None:
            last_checkpoint = checkpoints_df.iloc[-1]
            top_active = last_checkpoint[active_cols].abs().nlargest(10)
            assets = [col.replace('_Active', '') for col in top_active.index]

        fig = go.Figure()
        for asset in assets:
            col_name = f"{asset}_Active"
            if col_name in checkpoints_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=checkpoints_df.index,
                        y=checkpoints_df[col_name] * 10000,
                        mode='lines+markers',
                        name=asset,
                        line=dict(width=2),
                        marker=dict(size=8),
                        hovertemplate=f"{asset}<br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}bp<extra></extra>"
                    )
                )

        fig.update_layout(
            title="<b>체크포인트 Active Weight 변화</b>",
            xaxis_title="날짜",
            yaxis_title="Active Weight (bp)",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        fig.update_yaxes(tickformat=".3f")
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

        # 폰트 크기 적용
        fig = apply_chart_font_settings(fig)

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"체크포인트 가중치 표시 중 오류: {str(e)}")


# =============================================================================
# BL/Pair 유틸리티 함수들
# =============================================================================
def calculate_asset_rankings(weights_df: pd.DataFrame, views_df: pd.DataFrame, expected_returns_df: pd.DataFrame,
                             cov_matrix: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """자산 순위 계산"""
    if weights_df is None or weights_df.empty:
        return pd.DataFrame()

    try:
        asset_names_rank = weights_df["Asset"].astype(str).tolist()
        cash_mask = np.array([is_cash_asset(a) for a in asset_names_rank])

        # Pairwise scores (raw points from +/- signals; 2 -> 1pt, 1 -> 0.5pt)
        pairwise_raw = np.zeros(len(asset_names_rank))
        if views_df is not None and not views_df.empty:
            for _, view in views_df.iterrows():
                la = str(view.get("Long_Asset", ""))
                sa = str(view.get("Short_Asset", ""))
                try:
                    sig = float(view.get("Signal", 0.0)) or 0.0
                except Exception:
                    sig = 0.0
                # Each pair contributes half the signal magnitude per leg
                delta = 0.5 * sig
                if la in asset_names_rank:
                    pairwise_raw[asset_names_rank.index(la)] += delta
                if sa in asset_names_rank:
                    pairwise_raw[asset_names_rank.index(sa)] -= delta

        # Return scores
        return_scores = np.zeros(len(asset_names_rank))
        if expected_returns_df is not None and not expected_returns_df.empty and "Asset" in expected_returns_df.columns:
            ret_series = pd.to_numeric(
                expected_returns_df.set_index("Asset")["Expected_Return"],
                errors="coerce"
            ).reindex(asset_names_rank).fillna(0.0)
            return_scores = ret_series.values

        # Risk scores
        vols = np.ones(len(asset_names_rank))
        risk_scores = np.ones(len(asset_names_rank))
        if cov_matrix is not None and not cov_matrix.empty:
            cov = cov_matrix.reindex(index=asset_names_rank, columns=asset_names_rank).fillna(0.0)
            diag = np.diag(cov.values).astype(float)
            diag = np.clip(diag, 1e-8, None)
            vols = np.sqrt(diag)
            vols[cash_mask] = 1e-8
            risk_scores = 1.0 / vols
            risk_scores[cash_mask] = 0.0

        # Normalize
        pairwise_n = normalize_score(pairwise_raw, 0, 1)
        return_n = normalize_score(return_scores, 0, 1)
        risk_n = normalize_score(risk_scores, 0, 1)

        # Total score
        total = np.zeros(len(asset_names_rank))
        total[~cash_mask] = (
            0.4 * pairwise_n[~cash_mask]
            + 0.5 * return_n[~cash_mask]
            + 0.1 * risk_n[~cash_mask]
        )
        total[cash_mask] = 0.4 * pairwise_n[cash_mask] + 0.6 * return_n[cash_mask]

        ranks = rankdata(-total, method="average").astype(int)

        df = pd.DataFrame({
            "Asset": asset_names_rank,
            "Is_Cash": cash_mask,
            "Pairwise_Score": pairwise_raw,
            "Return_Score": return_n,
            "Risk_Score": risk_n,
            "Total_Score": total,
            "Rank": ranks,
            "Rank_Volatility": normalize_score(vols, 0, 3)
        }).sort_values("Rank")

        return df
    except Exception:
        return pd.DataFrame()

def ensure_decimal_returns(df_ret):
    """
    df_ret가 % 스케일(예: 1.2 = 1.2%)인지 소수(0.012)인지 감지해 소수로 통일.
    기준: |99% 분위수|가 0.5(=50%) 초과면 %로 간주하고 /100.
    """
    q99 = df_ret.abs().quantile(0.99, numeric_only=True).max()
    if q99 is not None and np.isfinite(q99) and q99 > 0.5:
        return df_ret / 100.0
    return df_ret


def build_recent_cov_constant_corr(df_ret_dec, window=63, rho=0.25):
    """
    최근 window(기본 63영업일) 일간 수익률(소수)로 공분산을 만들고,
    상수상관(Constant-Correlation) 타깃으로 ρ=0.25 수축(Convex combination)합니다.
    반환: cov_daily_dec (소수 스케일)
    """
    # 최근 구간 추출
    R = df_ret_dec.tail(window).dropna(how="all")
    R = R.dropna(axis=1, how="any")  # NaN 컬럼 제거
    if R.shape[1] == 0:
        raise ValueError("No valid columns for covariance after NaN filtering.")

    # 샘플 공분산 (소수 스케일)
    Sigma = R.cov(min_periods=max(10, window // 2)).values

    # 상수상관 타깃 구성
    std = R.std().values
    std = np.where(std <= 0, 1e-8, std)
    # 샘플 상관
    with np.errstate(divide='ignore', invalid='ignore'):
        Corr = Sigma / np.outer(std, std)
        Corr = np.nan_to_num(Corr, nan=0.0, posinf=0.0, neginf=0.0)
    # 평균 상관 (대각 제외)
    n = Corr.shape[0]
    if n > 1:
        r_bar = (Corr.sum() - np.trace(Corr)) / (n * (n - 1))
    else:
        r_bar = 0.0
    Corr_cc = np.full_like(Corr, r_bar, dtype=float)
    np.fill_diagonal(Corr_cc, 1.0)

    Sigma_cc = np.outer(std, std) * Corr_cc

    # 수축(ρ=0.25): Σ* = (1-ρ)Σ + ρΣ_cc
    Sigma_star = (1.0 - rho) * Sigma + rho * Sigma_cc
    return np.asarray(Sigma_star, dtype=float)


def build_incidence_matrix(assets: List[str], pairs: List[Tuple[str, str]]) -> np.ndarray:
    """Incidence 행렬 생성"""
    idx = {a: i for i, a in enumerate(assets)}
    N = len(assets)
    K = len(pairs)
    B = np.zeros((N, K))
    for j, (a_long, a_short) in enumerate(pairs):
        if a_long in idx and a_short in idx:
            B[idx[a_long], j] = +1.0
            B[idx[a_short], j] = -1.0
    return B


def reconstruct_pair_sizes(active_weights: np.ndarray, B: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """NNLS로 Pair 사이즈 추정"""
    if B.shape[1] == 0:
        return np.zeros(0)

    sgn = np.sign(signals)
    sgn[sgn == 0] = 1.0
    Bsig = B * sgn
    y, residual = nnls(Bsig, active_weights)
    x = sgn * y
    return x


def estimate_pair_contributions_nnls(Wact: pd.DataFrame, returns_df: pd.DataFrame, timeline_df: pd.DataFrame,
                                     start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Pair별 기여 추정"""
    if Wact.empty or returns_df.empty or timeline_df.empty:
        return pd.DataFrame()

    dates = returns_df.index[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    if len(dates) == 0:
        return pd.DataFrame()

    all_pairs = timeline_df[["Pair_ID", "Long_Asset", "Short_Asset"]].dropna().drop_duplicates()
    pair_key = list(map(tuple, all_pairs.values.tolist()))
    pair2idx = {k: i for i, k in enumerate(pair_key)}
    pair_contrib = np.zeros(len(pair_key))

    for d in dates:
        active = timeline_df[(timeline_df["Signal"] != 0)]
        if 'Start_Date' in active.columns:
            active = active[active['Start_Date'].fillna(pd.Timestamp.min) <= d]
        if 'End_Date' in active.columns:
            active = active[active['End_Date'].fillna(pd.Timestamp.max) >= d]
        if active.empty:
            continue

        common_assets = [c for c in Wact.columns if c in returns_df.columns]
        if not common_assets:
            break

        w = Wact.reindex(index=[d]).ffill().bfill()
        if w.empty:
            continue

        w_vec = w.iloc[0][common_assets].fillna(0.0).values
        r_vec = returns_df.loc[d, common_assets].fillna(0.0).values

        pairs_today = [(str(row['Long_Asset']), str(row['Short_Asset'])) for _, row in active.iterrows()]
        sig_today = active['Signal'].astype(float).values
        B = build_incidence_matrix(common_assets, pairs_today)
        if B.size == 0:
            continue

        x = reconstruct_pair_sizes(w_vec, B, sig_today)

        spreads = []
        for (la, sa) in pairs_today:
            if la in returns_df.columns and sa in returns_df.columns:
                spreads.append(float(returns_df.loc[d, la] - returns_df.loc[d, sa]))
            else:
                spreads.append(0.0)
        spreads = np.array(spreads)
        contrib_today = x * spreads

        for j, (la, sa) in enumerate(pairs_today):
            pid = active.iloc[j].get('Pair_ID', None)
            key = (pid, la, sa)
            if key in pair2idx:
                pair_contrib[pair2idx[key]] += contrib_today[j]

    out = all_pairs.copy()
    out['Contribution_%'] = (pair_contrib * 100.0)
    tot = float(np.sum(pair_contrib))
    out['Share_of_Active_%'] = (pair_contrib / tot * 100.0) if abs(tot) > 1e-12 else 0.0
    out['Contribution_bp'] = out['Contribution_%'] * 100.0
    out = out.sort_values('Contribution_%', ascending=False).reset_index(drop=True)
    return out


def compute_te_from_active(cov: pd.DataFrame, assets: List[str], w_act_series: pd.Series) -> float:
    """Active weight로부터 TE 계산"""
    if cov.empty or w_act_series.empty:
        return 0.0

    cov_use = cov.reindex(index=assets, columns=assets).fillna(0.0).values
    w = w_act_series.reindex(assets).fillna(0.0).values
    te2 = float(w @ cov_use @ w)
    return float(np.sqrt(max(te2, 0.0)))


def calculate_current_max_loss_bp(x_cur: np.ndarray, constraint_values: np.ndarray) -> float:
    """현재 포지션의 평균 최대 손실 계산 (bp)"""
    if len(x_cur) == 0 or len(constraint_values) == 0:
        return 0.1

    losses = np.abs(x_cur) * np.abs(constraint_values)
    if len(losses) == 0 or np.all(losses == 0):
        return 0.1

    avg_loss_bp = np.mean(losses[losses > 0]) * 10000 if np.any(losses > 0) else 0.1
    return max(0.05, float(avg_loss_bp))


def calculate_period_performance(daily_returns_df, periods={'1M': 21, '3M': 63, '6M': 126, '12M': 252}):
    """
    기간별 성과 계산 함수

    Args:
        daily_returns_df: 일별 수익률 DataFrame (Portfolio_Return, Benchmark_Return, Active_Return)
        periods: 기간별 영업일 수 딕셔너리

    Returns:
        DataFrame: 기간별 성과 지표
    """
    if daily_returns_df.empty:
        return pd.DataFrame()

    latest_date = daily_returns_df.index[-1]
    results = []

    for period_name, days in periods.items():
        # 해당 기간의 데이터 추출
        period_data = daily_returns_df.tail(days)

        if len(period_data) < days:
            # 데이터가 부족한 경우 사용 가능한 모든 데이터 사용
            period_data = daily_returns_df
            actual_days = len(period_data)
            note = f"(실제 {actual_days}일)"
        else:
            actual_days = days
            note = ""

        if len(period_data) == 0:
            continue

        # 각 컬럼별 성과 계산
        for col in ['Portfolio_Return', 'Benchmark_Return', 'Active_Return']:
            if col in period_data.columns:
                # 누적 수익률
                cum_ret = (1 + period_data[col]).prod() - 1

                # 연율화 수익률
                if actual_days > 0:
                    ann_ret = (1 + cum_ret) ** (252 / actual_days) - 1
                else:
                    ann_ret = 0.0

                # 연율화 변동성
                ann_vol = period_data[col].std() * np.sqrt(252)

                # Sharpe Ratio (Active Return의 경우)
                if col == 'Active_Return' and ann_vol > 0:
                    sharpe = ann_ret / ann_vol
                else:
                    sharpe = np.nan

                results.append({
                    'Period': f"{period_name}{note}",
                    'Type': col.replace('_Return', ''),
                    'Cumulative_bp': cum_ret * 10000,
                    'Annualized_bp': ann_ret * 10000,
                    'Volatility_bp': ann_vol * 10000,
                    'Sharpe': sharpe,
                    'Start_Date': period_data.index[0].strftime('%Y-%m-%d'),
                    'End_Date': period_data.index[-1].strftime('%Y-%m-%d'),
                    'Days': actual_days
                })

    return pd.DataFrame(results)


# =============================================================================
# 메인 대시보드
# =============================================================================
def main():
    st.title("📊 ITAA Black-Litterman Portfolio Tracker")
    st.markdown("### 일별 백테스트 결과 및 Pairwise View 분석 (bp 단위)")
    st.markdown("---")

    # Session state 초기화 (기본값 설정)
    if 'constraint_method' not in st.session_state:
        st.session_state.constraint_method = "-3STD"  # 기본값
    if 'lookback_years' not in st.session_state:
        st.session_state.lookback_years = 3  # 기본값
    if 'adjusted_views' not in st.session_state:
        st.session_state.adjusted_views = None
    if 'inception_date' not in st.session_state:
        st.session_state.inception_date = None
    if 'common_positions' not in st.session_state:
        st.session_state.common_positions = None

    # Sidebar 설정 (간소화)
    # Streamlit Cloud용 상대 경로 설정
    from pathlib import Path
    BASE_DIR = Path(__file__).parent
    DEFAULT_DATA_DIR = str(BASE_DIR / "iTAA")
    DEFAULT_MARKET_CSV = str(BASE_DIR / "data" / "pr_res_bd.csv")
    DEFAULT_EXCEL_PATH = str(BASE_DIR / "data" / "itaa_Master.xlsx")

    with st.sidebar:
        st.header("⚙️ 설정")
        data_dir = st.text_input(
            "결과 CSV 디렉토리",
            value=DEFAULT_DATA_DIR,
            help="portfolio_weights 등 CSV가 저장된 폴더",
        )
        market_csv = st.text_input(
            "시장 데이터 CSV (pr_res_bd.csv)",
            value=DEFAULT_MARKET_CSV
        )
        excel_path = st.text_input(
            "itaa_Master.xlsx 경로",
            value=DEFAULT_EXCEL_PATH,
            help="Asset_Universe/Pairs_Definition/Views_Timeline/Benchmark_Weights 매핑"
        )

        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.session_state.adjusted_views = None
            st.rerun()

    # 기본 분석 기간 설정 (사이드바 없이 자동 설정)
    default_end = pd.Timestamp.now().normalize()
    default_start = default_end - pd.Timedelta(days=90)
    start_date = default_start
    end_date = default_end

    # 기본 constraint_method와 lookback_years 사용
    constraint_method = st.session_state.constraint_method
    lookback_years = st.session_state.lookback_years


    # 데이터 로드
    try:
        data = load_csv_data(data_dir)
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        st.stop()


    # CSV 데이터 추출
    daily_returns_port = data.get("daily_returns_series", pd.DataFrame())
    views_df_csv = data.get("active_views", pd.DataFrame())
    weights_df = data.get("portfolio_weights", pd.DataFrame())
    expected_returns_df = data.get("expected_returns", pd.DataFrame())
    cov_matrix = data.get("covariance_matrix", pd.DataFrame())
    rebalance_log_df = data.get("rebalance_log", pd.DataFrame())
    rebalance_calendar_df = data.get("rebalance_calendar", pd.DataFrame())
    w_opt_daily = data.get("daily_weights_optimal", pd.DataFrame())
    w_bmk_daily = data.get("daily_weights_benchmark", pd.DataFrame())
    weight_history = data.get("weight_history", pd.DataFrame())
    timeline_history = data.get("view_timeline_history", pd.DataFrame())
    attrib_report = data.get("attribution_report", pd.DataFrame())
    attrib_summary = data.get("attribution_summary", pd.DataFrame())
    pair_mdd_report_file = data.get("pair_mdd_report", pd.DataFrame())
    pair_constraints_file = data.get("pair_constraints", pd.DataFrame())
    performance_metrics = data.get("performance_metrics", pd.DataFrame())

    # 자산 리스트 구성
    if not weights_df.empty:
        asset_names = weights_df["Asset"].astype(str).tolist()
    elif not w_opt_daily.empty:
        asset_names = list(map(str, w_opt_daily.columns))
    elif not weight_history.empty:
        tmp_cols = [c.replace('_Optimal', '').replace('_Benchmark', '').replace('_Active', '') for c in
                    weight_history.columns]
        asset_names = sorted(list(pd.Index(tmp_cols).unique()))
    else:
        asset_names = []

    # 시장 수익률 로드
    returns_by_asset = load_market_returns_csv(market_csv, asset_names, excel_path) if asset_names else pd.DataFrame()

    # Views 로드
    views_from_excel = load_views_from_excel(excel_path)
    asof_for_views = end_date
    views_from_timeline = load_active_views_from_timeline(timeline_history, asof_for_views)
    views_source = views_from_excel if not views_from_excel.empty else views_from_timeline

    # 일별 수익률 합성 (필요시)
    if daily_returns_port.empty and not weight_history.empty and not returns_by_asset.empty:
        st.info("🧮 weight_history 기반 daily_returns_series 합성 생성")
        weight_history = weight_history.sort_index()
        assets_from_wh = sorted({c.replace('_Optimal', '').replace('_Benchmark', '').replace('_Active', '') for c in
                                 weight_history.columns})
        common_assets = [a for a in assets_from_wh if a in returns_by_asset.columns]

        def get_weights_for_day(day):
            idx = weight_history.index.searchsorted(day, side='right') - 1
            if idx < 0:
                return None
            row = weight_history.iloc[idx]
            w_opt = np.array([row.get(f"{a}_Optimal", 0.0) for a in common_assets])
            w_bmk = np.array([row.get(f"{a}_Benchmark", 0.0) for a in common_assets])
            return w_opt, w_bmk

        port_ret_series = []
        bmk_ret_series = []
        act_ret_series = []
        idx_used = []

        for day, r in returns_by_asset[common_assets].iterrows():
            w = get_weights_for_day(day)
            if w is None:
                continue
            w_opt, w_bmk = w
            pr = float(np.dot(r.values, w_opt))
            br = float(np.dot(r.values, w_bmk))
            port_ret_series.append(pr)
            bmk_ret_series.append(br)
            act_ret_series.append(pr - br)
            idx_used.append(day)

        daily_returns_port = pd.DataFrame({
            'Portfolio_Return': port_ret_series,
            'Benchmark_Return': bmk_ret_series,
            'Active_Return': act_ret_series
        }, index=pd.to_datetime(idx_used))

        daily_returns_port = daily_returns_port.loc[
            (daily_returns_port.index >= start_date) & (daily_returns_port.index <= end_date)]
        st.success(f"✅ 합성 일별 수익률 생성: {len(daily_returns_port)} 영업일")

    # 리스크 제약 적용 일별 수익률 재계산
    # 리스크 제약 적용 일별 수익률 재계산
    daily_returns_constrained = pd.DataFrame()
    position_changes_df = pd.DataFrame()

    if not returns_by_asset.empty and not views_source.empty and not w_bmk_daily.empty:
        with st.spinner(f"🔄 {constraint_method} 제약 적용 일별 수익률 계산 중..."):
            daily_returns_constrained, position_changes_df = calculate_daily_returns_with_constraint(
                returns_by_asset=returns_by_asset,
                views_timeline=views_source,
                w_bmk_daily=w_bmk_daily,
                w_opt_daily=w_opt_daily,  # ★ 추가
                start_date=start_date,
                end_date=end_date,
                constraint_method=constraint_method,
                lookback_years=3,
                sizing_mode="full_cap",
            )

        if not daily_returns_constrained.empty:
            st.success(f"✅ 제약 적용 일별 수익률 계산 완료: {len(daily_returns_constrained)} 영업일")

            # 진단 정보 표시
            if not position_changes_df.empty:
                avg_position = position_changes_df['Position_per_leg_bp'].abs().mean()
                max_position = position_changes_df['Position_per_leg_bp'].abs().max()
                avg_total_active = position_changes_df['Total_Active_bp'].mean()
                avg_loss = position_changes_df['Actual_Loss_bp'].mean()

                st.info(f"""
                📊 포지션 통계:
                - 평균 Per-leg 포지션: {avg_position:.3f}bp
                - 최대 Per-leg 포지션: {max_position:.3f}bp
                - 평균 Total Active: {avg_total_active:.3f}bp
                - 평균 실제 손실 (제약 발생 시): {avg_loss:.3f}bp
                """)
    # Active PnL 계산
    def _calc_active_pnl_simple():
        if returns_by_asset.empty or w_opt_daily.empty or w_bmk_daily.empty:
            return pd.DataFrame(), [], 0.0, pd.DataFrame()

        s = start_date
        e = end_date
        r = returns_by_asset.loc[(returns_by_asset.index >= s) & (returns_by_asset.index <= e)].copy()
        Wopt = w_opt_daily.reindex(r.index).ffill().bfill()
        Wbmk = w_bmk_daily.reindex(r.index).ffill().bfill()
        common = [c for c in r.columns if c in Wopt.columns and c in Wbmk.columns]
        if not common:
            return pd.DataFrame(), [], 0.0, pd.DataFrame()

        r = r[common]
        Wopt = Wopt[common]
        Wbmk = Wbmk[common]
        Wact = Wopt - Wbmk
        pnl_ai = Wact * r
        total_active_return = pnl_ai.sum(axis=1).sum() * 100
        return pnl_ai, common, total_active_return, Wact

    pnl_ai, common_assets, total_active_return, Wact_period = _calc_active_pnl_simple()

    # 탭 구성
    tabs = st.tabs([
        "🎯 Active View 순위",
        "📈 수익 기여도",
        "⚖️ 액티브 포지션",
        "📊 일별 성과",
        "🔄 리밸런싱 분석",
        "⚡ 가중치 추적",
        "📋 포트폴리오 개요",
        "⚠️ 리스크 분석",
        "🔗 상관관계 분석",
        "🎲 Pair 기대수익률 (3M Rolling)",
        "🛑 리스크 제약 감도분석",
        "📊 실제 포트폴리오 성과"  # ← 새로 추가

    ])

    # =========================================================================
    # Tab 0: Active View 순위
    # =========================================================================
    with tabs[0]:
        st.header("🎯 Active Pairwise View에 따른 자산 순위 변화 (Signal 전용)")

        if (views_source is None or views_source.empty) or weights_df.empty:
            st.warning("필요한 데이터가 없습니다. Views 또는 Weights 데이터를 확인하세요.")
        else:
            # ========== 1. 포지션 크기 결정 기준 선택 (최상단) ==========
            st.subheader("⚙️ 포지션 크기 결정 기준")
            st.markdown("""
            모든 탭에서 공통으로 사용할 포지션 계산 방법을 선택하세요.  
            💡 **3개월 롤링 리턴** 기반 (EWM halflife=126일)
            """)

            col_config1, col_config2 = st.columns(2)

            with col_config1:
                constraint_method = st.selectbox(
                    "리스크 제약 방법",
                    ["3Y_MDD", "-3STD", "-2STD", "-1STD"],
                    index=1,  # 기본값: -3STD
                    key="global_constraint_method",
                    help="페어별 포지션 크기를 결정하는 리스크 제약 방법"
                )

            with col_config2:
                lookback_years = st.slider(
                    "룩백 기간 (년)",
                    1, 5, 3,
                    key="global_lookback_years",
                    help="3개월 롤링 리턴 계산에 사용할 과거 데이터 기간"
                )

            # Constraint 방법 표시 맵
            CONSTRAINT_DISPLAY_MAP = {
                "3Y_MDD": "3년 최대손실(MDD)",
                "-3STD": "-3 표준편차 (3M 롤링)",
                "-2STD": "-2 표준편차 (3M 롤링)",
                "-1STD": "-1 표준편차 (3M 롤링)"
            }

            constraint_display = CONSTRAINT_DISPLAY_MAP.get(constraint_method, constraint_method)
            st.info(f"🎯 선택된 제약: **{constraint_display}**, 룩백: **{lookback_years}년**")

            # Session state에 설정 저장
            st.session_state.constraint_method = constraint_method
            st.session_state.lookback_years = lookback_years

            st.markdown("---")

            # ========== 2. As-of 가중치 준비 ==========
            if not w_opt_daily.empty and not w_bmk_daily.empty:
                asof = min(w_opt_daily.index.max(), w_bmk_daily.index.max())
                Wopt_last = w_opt_daily.loc[asof].fillna(0.0)
                Wbmk_last = w_bmk_daily.loc[asof].fillna(0.0)
            elif not weight_history.empty:
                asof = weight_history.index.max()
                row = weight_history.loc[asof]
                assets_wh = sorted({c.replace('_Optimal', '').replace('_Benchmark', '').replace('_Active', '')
                                    for c in weight_history.columns})
                Wopt_last = pd.Series({a: row.get(f"{a}_Optimal", 0.0) for a in assets_wh})
                Wbmk_last = pd.Series({a: row.get(f"{a}_Benchmark", 0.0) for a in assets_wh})
            else:
                Wopt_last = pd.Series(dtype=float)
                Wbmk_last = pd.Series(dtype=float)

            # ========== 3. 레이아웃: View 조정 + 순위 변화 ==========
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("📝 View 조정")
                adjusted_views = views_source.copy().reset_index(drop=True)

                for idx in range(len(adjusted_views)):
                    row = adjusted_views.iloc[idx]
                    pair_id = row.get('Pair_ID', idx + 1)
                    pair_name = f"Pair {pair_id}: {row.get('Long_Asset', '')} vs {row.get('Short_Asset', '')}"

                    with st.expander(f"**{pair_name}**", expanded=(idx < 3)):
                        signal = st.slider(
                            "Signal",
                            -2, 2,
                            int(_to_float(row.get('Signal', 0)) or 0),
                            key=f"signal_{idx}"
                        )
                        adjusted_views.loc[idx, 'Signal'] = float(signal)

                        # Signal에 따른 손실 허용치 표시
                        abs_sig = abs(signal)
                        if abs_sig >= 2.0:
                            max_loss = 0.15
                        elif abs_sig >= 1.0:
                            max_loss = 0.10
                        else:
                            max_loss = 0.10 + abs_sig * 0.5

                        st.caption(f"💡 최대 손실 허용: {max_loss:.2f}bp (Signal {abs_sig:.0f} 기준)")

                # 조정된 views를 session_state에 저장
                st.session_state.adjusted_views = adjusted_views

                # ========== 4. 공통 포지션 계산 및 저장 ==========
                st.markdown("---")
                st.subheader("📊 Pair별 포지션 크기")

                if not returns_by_asset.empty and not adjusted_views.empty:
                    with st.spinner("포지션 계산 중... (EWM 방식의 3개월 롤링 리턴)"):
                        common_positions = calculate_common_positions(
                            returns_by_asset,
                            adjusted_views,
                            constraint_method,
                            lookback_years
                        )

                    if not common_positions.empty:
                        # ⚠️ Session state에 저장 (모든 탭에서 사용)
                        st.session_state.common_positions = common_positions

                        st.success("✅ 포지션이 계산되었습니다. (모든 탭에 적용)")

                        # 요약 메트릭
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            total_notional = common_positions['Total_Notional_bp'].abs().sum()
                            st.metric("총 Notional", f"{total_notional:.1f}bp")
                        with col_m2:
                            avg_position = common_positions['Per_Leg_Position_bp'].abs().mean()
                            st.metric("평균 Per-Leg", f"{avg_position:.2f}bp")
                        with col_m3:
                            n_pairs = len(common_positions)
                            n_cash = common_positions['Is_Cash_Pair'].sum()
                            st.metric("Pair 수", f"{n_pairs} ({n_cash}개 Cash)")

                        # 상세 테이블
                        with st.expander("📋 포지션 상세 정보", expanded=True):
                            display_cols = [
                                'Pair_ID', 'Pair', 'Signal', 'Leg_Factor',
                                'Risk_Unit_3M_%', 'Max_Loss_bp',
                                'Per_Leg_Position_bp', 'Total_Notional_bp'
                            ]
                            position_display = common_positions[display_cols].copy()

                            # 포맷팅
                            position_display['Risk_Unit_3M_%'] = position_display['Risk_Unit_3M_%'].apply(
                                lambda x: f"{x:.3f}%"
                            )
                            position_display['Max_Loss_bp'] = position_display['Max_Loss_bp'].apply(
                                lambda x: f"{x:.2f}"
                            )
                            position_display['Per_Leg_Position_bp'] = position_display['Per_Leg_Position_bp'].apply(
                                lambda x: f"{x:.3f}"
                            )
                            position_display['Total_Notional_bp'] = position_display['Total_Notional_bp'].apply(
                                lambda x: f"{x:.3f}"
                            )

                            st.dataframe(position_display, use_container_width=True)

                            st.caption(f"✅ {constraint_display} 기준으로 계산")

                        # 포지션 분포 차트
                        fig_pos = go.Figure()

                        colors = ['#2ca02c' if s > 0 else '#d62728'
                                  for s in common_positions['Signal']]

                        fig_pos.add_trace(go.Bar(
                            x=common_positions['Pair'],
                            y=common_positions['Total_Notional_bp'],
                            marker_color=colors,
                            text=common_positions['Total_Notional_bp'].apply(lambda x: f"{x:.1f}"),
                            textposition='outside',
                            hovertemplate="<b>%{x}</b><br>Notional: %{y:.2f}bp<extra></extra>"
                        ))

                        fig_pos.update_layout(
                            title=f"Pair별 Total Notional (bp)",
                            xaxis_title="Pair",
                            yaxis_title="Total Notional (bp)",
                            height=400,
                            showlegend=False
                        )
                        fig_pos.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                        fig_pos = apply_chart_font_settings(fig_pos)
                        st.plotly_chart(fig_pos, use_container_width=True)

                    else:
                        st.warning("포지션 계산 결과가 없습니다.")
                else:
                    st.info("포지션 계산에 필요한 데이터가 없습니다.")

                # ========== 5. 예상 리스크 지표 ==========
                st.markdown("---")
                st.subheader("📊 예상 리스크 지표")

                # Benchmark 상태 확인
                if not Wbmk_last.empty:
                    bm_sum = Wbmk_last.sum()
                    if abs(bm_sum) < 0.001:
                        st.info(f"💡 Benchmark = 0 (100% Cash) → **TE = Vol**")

                # 리스크 지표 계산
                if not returns_by_asset.empty and not Wopt_last.empty:
                    # 공통 컬럼 정렬
                    cols = [c for c in returns_by_asset.columns if c in Wopt_last.index]
                    R = returns_by_asset[cols]
                    w_p = Wopt_last.reindex(cols).fillna(0.0)

                    if not Wbmk_last.empty:
                        w_b = Wbmk_last.reindex(cols).fillna(0.0)
                    else:
                        w_b = pd.Series(0.0, index=cols)

                    # 공분산 (63일, 상수상관 ρ=0.25)
                    R_dec = _pc_ensure_decimal_returns(R)
                    C = _pc_build_recent_cov_constant_corr(R_dec, window=63, rho=0.25)

                    # 현재 포트폴리오 리스크
                    cur_te_bp = _pc_te_bp_from_cov((w_p - w_b).values, C, 252)
                    cur_vol_bp = _pc_te_bp_from_cov(w_p.values, C, 252)

                    # 조정 후 포트폴리오 가중치 계산
                    if 'common_positions' in st.session_state and st.session_state.common_positions is not None:
                        common_pos_df = st.session_state.common_positions

                        pairs_adj = [(str(r['Long_Asset']), str(r['Short_Asset']))
                                     for _, r in adjusted_views.iterrows()]
                        signals_adj = adjusted_views['Signal'].astype(float).values

                        # Incidence matrix
                        B = build_incidence_matrix(cols, pairs_adj)

                        if B.size > 0 and len(pairs_adj) > 0:
                            # Per-leg 포지션 (Signal 방향 반영, 소수 변환)
                            x_adj = np.zeros(len(pairs_adj))

                            for i, pid in enumerate(adjusted_views.get('Pair_ID', range(len(pairs_adj)))):
                                pos_row = common_pos_df[common_pos_df['Pair_ID'] == pid]
                                if not pos_row.empty:
                                    # ✅ bp → 소수 올바른 변환 (1bp = 0.0001 = 0.01%)
                                    per_leg_bp = float(pos_row.iloc[0]['Per_Leg_Position_bp'])
                                    per_leg_decimal = per_leg_bp / 10000.0

                                    # Signal 방향 반영
                                    x_adj[i] = float(np.sign(signals_adj[i]) * per_leg_decimal)

                            # Active weights
                            w_active_adj = pd.Series(B @ x_adj, index=cols)

                            # 이상치 감지
                            max_active = w_active_adj.abs().max()
                            sum_active = w_active_adj.abs().sum()

                            if max_active > 0.5 or sum_active > 2.0:
                                st.error(f"⚠️ 조정 후 가중치 이상 감지!")
                                st.error(f"Max: {max_active * 100:.1f}%, Sum: {sum_active * 100:.1f}%")
                                w_pa = w_p.copy()
                            else:
                                # Portfolio weights = Benchmark + Active
                                w_portfolio_raw = w_b + w_active_adj

                                bm_sum = float(w_b.sum())

                                if abs(bm_sum) < 1e-6:
                                    # BM=0 (현금) → Active 그대로 (숏 허용)
                                    w_pa = w_portfolio_raw
                                else:
                                    # BM 있을 때: 롱온리 + 정규화
                                    w_pa = w_portfolio_raw.clip(lower=0)
                                    s = w_pa.sum()
                                    w_pa = (w_pa / s) if s > 0 else w_p.copy()

                            # 조정 후 리스크
                            adj_te_bp = _pc_te_bp_from_cov((w_pa - w_b).values, C, 252)
                            adj_vol_bp = _pc_te_bp_from_cov(w_pa.values, C, 252)

                            # 이상치 재검증
                            if adj_te_bp > 1000:
                                st.error(f"🚨 조정 후 TE 비정상: {adj_te_bp:.1f}bp")
                                adj_te_bp = cur_te_bp
                                adj_vol_bp = cur_vol_bp
                        else:
                            adj_te_bp = cur_te_bp
                            adj_vol_bp = cur_vol_bp
                    else:
                        adj_te_bp = cur_te_bp
                        adj_vol_bp = cur_vol_bp

                    # 화면 출력
                    col_r1, col_r2 = st.columns(2)

                    with col_r1:
                        st.markdown("#### 📉 현재 예상 리스크")
                        st.metric("예상 TE", f"{cur_te_bp:,.2f}bp")
                        st.metric("예상 Vol", f"{cur_vol_bp:,.2f}bp")

                    with col_r2:
                        st.markdown("#### 📈 조정 후 예상 리스크")
                        delta_te = adj_te_bp - cur_te_bp
                        delta_vol = adj_vol_bp - cur_vol_bp

                        st.metric("예상 TE", f"{adj_te_bp:,.2f}bp", delta=f"{delta_te:+.2f}bp")
                        st.metric("예상 Vol", f"{adj_vol_bp:,.2f}bp", delta=f"{delta_vol:+.2f}bp")

                    # ========== 6. 손실 한도 점검 ==========
                    st.markdown("---")
                    st.subheader("⚠️ 손실 한도 점검")



                    if 'common_positions' in st.session_state and st.session_state.common_positions is not None:
                        check_df = common_positions.copy()

                        # ✅ 수정: get_max_loss_for_signal 함수
                        def get_max_loss_for_signal(signal):
                            abs_sig = abs(signal)
                            if abs_sig >= 2.0:
                                return 0.15  # bp
                            elif abs_sig >= 1.0:
                                return 0.1  # bp
                            else:
                                return 0.1 + abs_sig * 0.05

                        check_df['Max_Loss_Allowed_bp'] = check_df['Signal'].apply(
                            get_max_loss_for_signal
                        )



                        # 실제 손실 (bp): Risk_Unit × Position × Leg_Factor
                        check_df['Expected_Loss_bp'] = (
                                check_df['Risk_Unit_3M_%'] / 100.0 *  # % → 소수
                                check_df['Per_Leg_Position_bp'].abs() / 10000.0 *  # bp → 소수
                                check_df['Leg_Factor'] * 10000.0  # bp로 변환
                        )

                        # Max Loss는 이미 올바름
                        check_df['Utilization_%'] = (
                                check_df['Expected_Loss_bp'] / check_df['Max_Loss_bp'] * 100
                        )
                        # 위반 여부 (1% 여유)
                        check_df['Violation'] = (check_df['Utilization_%'] > 101)

                        # 요약
                        n_violations = check_df['Violation'].sum()
                        avg_util = check_df['Utilization_%'].mean()
                        max_util = check_df['Utilization_%'].max()

                        col_c1, col_c2, col_c3 = st.columns(3)

                        with col_c1:
                            if n_violations > 0:
                                st.error(f"⚠️ 위반: {n_violations}개")
                            else:
                                st.success("✅ 모두 한도 내")

                        with col_c2:
                            st.metric("평균 활용률", f"{avg_util:.1f}%")

                        with col_c3:
                            color = "🔴" if max_util > 101 else "🟢"
                            st.metric("최대 활용률", f"{color} {max_util:.1f}%")

                        # 위반 항목 표시
                        if n_violations > 0:
                            st.warning(f"⚠️ {n_violations}개 Pair 손실 한도 초과:")

                            violation_df = check_df[check_df['Violation']].copy()
                            display_cols = [
                                'Pair_ID', 'Pair', 'Signal',
                                'Max_Loss_bp', 'Expected_Loss_bp', 'Utilization_%'
                            ]

                            viol_display = violation_df[display_cols].copy()
                            viol_display = viol_display.style.apply(
                                lambda x: ['background-color: #ffe6e6'] * len(x), axis=1
                            ).format({
                                'Max_Loss_bp': '{:.2f}',
                                'Expected_Loss_bp': '{:.2f}',
                                'Utilization_%': '{:.1f}'
                            })

                            st.dataframe(viol_display, use_container_width=True)
                        else:
                            with st.expander("✅ 손실 한도 상세", expanded=False):
                                detail_cols = [
                                    'Pair_ID', 'Pair', 'Signal',
                                    'Max_Loss_bp', 'Expected_Loss_bp', 'Utilization_%'
                                ]
                                detail_df = check_df[detail_cols].copy()
                                st.dataframe(
                                    detail_df.style.format({
                                        'Max_Loss_bp': '{:.2f}',
                                        'Expected_Loss_bp': '{:.2f}',
                                        'Utilization_%': '{:.1f}'
                                    }),
                                    use_container_width=True
                                )
                    else:
                        st.info("포지션 계산 후 손실 한도를 점검할 수 있습니다.")

                    # 리스크 경고
                    if adj_te_bp > 50.0:
                        st.warning("⚠️ **높은 리스크**: TE > 50bp. View 강도 조정 권장")
                    elif adj_te_bp > 30.0:
                        st.info("ℹ️ 중간 리스크 수준. View 설정 검토 필요")

                else:
                    st.warning("⚠️ 시장 데이터 또는 포지션 정보가 없어 리스크 계산 불가")

            # ========== 7. 오른쪽 컬럼: 실시간 순위 변화 ==========
            with col2:
                st.subheader("📊 실시간 순위 변화")

                expected_returns_use = expected_returns_df
                if (expected_returns_use is None or expected_returns_use.empty) and not returns_by_asset.empty:
                    ret_series = returns_by_asset.mean() * 252.0
                    expected_returns_use = pd.DataFrame({
                        "Asset": ret_series.index,
                        "Expected_Return": ret_series.values
                    })

                cov_matrix_use = cov_matrix
                if (cov_matrix_use is None or cov_matrix_use.empty) and not returns_by_asset.empty:
                    cov_matrix_use = returns_by_asset.cov() * 252.0

                current_ranking = calculate_asset_rankings(
                    weights_df, views_source, expected_returns_use, cov_matrix_use
                )
                adjusted_ranking = calculate_asset_rankings(
                    weights_df, adjusted_views, expected_returns_use, cov_matrix_use
                )

                if not current_ranking.empty and not adjusted_ranking.empty:
                    cash_assets_list = current_ranking[current_ranking['Is_Cash']]['Asset'].tolist()
                    if cash_assets_list:
                        st.info(f"💵 Cash 자산: {', '.join(cash_assets_list)}")

                    fig = go.Figure()

                    # 현재 순위
                    fig.add_trace(go.Scatter(
                        x=current_ranking['Rank_Volatility'],
                        y=current_ranking['Rank'].max() - current_ranking['Rank'] + 1,
                        mode='markers',
                        name='현재 순위',
                        marker=dict(
                            size=15,
                            color='lightgray',
                            symbol='circle-open',
                            line=dict(width=2)
                        ),
                        text=current_ranking['Asset'],
                        hovertemplate="<b>%{text}</b><br>현재: %{customdata}<extra></extra>",
                        customdata=current_ranking['Rank']
                    ))

                    # 조정 순위
                    fig.add_trace(go.Scatter(
                        x=adjusted_ranking['Rank_Volatility'],
                        y=adjusted_ranking['Rank'].max() - adjusted_ranking['Rank'] + 1,
                        mode='markers+text',
                        name='조정 순위',
                        marker=dict(
                            size=adjusted_ranking['Total_Score'] * 30 + 10,
                            color=adjusted_ranking['Total_Score'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="종합점수")
                        ),
                        text=adjusted_ranking['Asset'],
                        textposition="top center",
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "조정 순위: %{customdata[0]}<br>"
                            "Pairwise: %{customdata[1]:.3f}<br>"
                            "Return: %{customdata[2]:.3f}<br>"
                            "Risk: %{customdata[3]:.3f}<br>"
                            "<extra></extra>"
                        ),
                        customdata=adjusted_ranking[[
                            'Rank', 'Pairwise_Score', 'Return_Score', 'Risk_Score'
                        ]].values
                    ))

                    # 순위 변화 화살표
                    for _, row in current_ranking.iterrows():
                        asset = row['Asset']
                        current_rank = row['Rank']
                        adj_row = adjusted_ranking[adjusted_ranking['Asset'] == asset]

                        if not adj_row.empty:
                            adjusted_rank = adj_row.iloc[0]['Rank']
                            if current_rank != adjusted_rank:
                                fig.add_annotation(
                                    x=adj_row.iloc[0]['Rank_Volatility'],
                                    y=adjusted_ranking['Rank'].max() - adjusted_rank + 1,
                                    ax=row['Rank_Volatility'],
                                    ay=current_ranking['Rank'].max() - current_rank + 1,
                                    xref="x", yref="y",
                                    axref="x", ayref="y",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor="red" if adjusted_rank > current_rank else "green",
                                    opacity=0.5
                                )

                    fig.update_layout(
                        title="자산 순위 변화 (Y: 순위, X: Rank 변동성)",
                        xaxis_title="Rank 변동성",
                        yaxis_title="순위 (높을수록 좋음)",
                        height=600,
                        hovermode='closest',
                        showlegend=True
                    )

                    fig = apply_chart_font_settings(fig)
                    st.plotly_chart(fig, use_container_width=True)

                    # 상세 순위 테이블
                    st.subheader("📋 상세 순위 정보")

                    display_df = adjusted_ranking[[
                        'Rank', 'Asset', 'Is_Cash', 'Total_Score',
                        'Pairwise_Score', 'Return_Score', 'Risk_Score'
                    ]].copy()

                    display_df['Is_Cash'] = display_df['Is_Cash'].map({True: '💵', False: ''})

                    max_assets = len(display_df)
                    if max_assets > 1:
                        n_display = st.slider(
                            "표시 자산 수",
                            min_value=min(5, max_assets),
                            max_value=max_assets,
                            value=min(10, max_assets),
                            key="asset_ranking_display"
                        )
                        display_subset = display_df.head(n_display).copy()
                    else:
                        display_subset = display_df.copy()

                    st.dataframe(display_subset, use_container_width=True)

                    fig_pairwise = go.Figure(
                        data=[go.Bar(
                            x=display_subset['Asset'],
                            y=display_subset['Pairwise_Score'],
                            marker=dict(color=display_subset['Pairwise_Score'], colorscale='RdBu')
                        )]
                    )
                    fig_pairwise.update_layout(
                        title="자산별 Pairwise Score",
                        xaxis_title="Asset",
                        yaxis_title="Pairwise Score (pts)",
                        yaxis=dict(range=[-1.2, 1.2]),
                        height=400
                    )
                    fig_pairwise = apply_chart_font_settings(fig_pairwise)
                    st.plotly_chart(fig_pairwise, use_container_width=True)


    # =========================================================================
    # Tab 1: 수익 기여도
    # =========================================================================
    with tabs[1]:
        st.header("📈 Pairwise View별 수익 기여도 (bp 단위)")

        with st.expander("🔍 데이터 로드 상태 확인", expanded=False):
            st.write(
                f"returns_by_asset: {len(returns_by_asset)} rows, {len(returns_by_asset.columns) if not returns_by_asset.empty else 0} cols")
            st.write(
                f"w_opt_daily: {len(w_opt_daily)} rows, {len(w_opt_daily.columns) if not w_opt_daily.empty else 0} cols")
            st.write(
                f"w_bmk_daily: {len(w_bmk_daily)} rows, {len(w_bmk_daily.columns) if not w_bmk_daily.empty else 0} cols")
            st.write(f"선택 기간: {start_date.date()} ~ {end_date.date()}")
            st.write(f"pnl_ai: {len(pnl_ai)} rows, {len(pnl_ai.columns) if not pnl_ai.empty else 0} cols")
            st.write(f"common_assets: {len(common_assets) if common_assets else 0}개")

        if returns_by_asset.empty:
            st.error("❌ 시장 수익률 데이터가 없습니다. pr_res_bd.csv 파일을 확인하세요.")
        elif pnl_ai.empty and weight_history.empty:
            st.error(f"❌ 선택한 기간에 Active PnL 데이터가 없습니다. 날짜 범위를 조정하세요.")
            if not w_opt_daily.empty:
                st.info(f"📅 사용 가능한 데이터 범위: {w_opt_daily.index.min().date()} ~ {w_opt_daily.index.max().date()}")
        else:
            st.success(f"✅ Active PnL 계산 완료")

            pair_contrib = pd.DataFrame()
            if not attrib_report.empty:
                pair_contrib = attrib_report.copy()
                if 'Contribution_bps' in pair_contrib.columns and 'Contribution_%' not in pair_contrib.columns:
                    pair_contrib['Contribution_%'] = pair_contrib['Contribution_bps'] / 100.0
                st.info(f"✅ attribution_report.csv 사용 ({len(pair_contrib)}개 페어)")

            if pair_contrib.empty and not timeline_history.empty:
                st.info("🧮 타임라인 + NNLS 기반 추정 기여 계산")
                if Wact_period is None or Wact_period.empty:
                    if not weight_history.empty:
                        wh = weight_history.sort_index()
                        wh_assets = sorted(
                            {c.replace('_Optimal', '').replace('_Benchmark', '').replace('_Active', '') for c in
                             wh.columns}
                        )
                        common_wh = [a for a in wh_assets if a in returns_by_asset.columns]
                        rows = []
                        for d in returns_by_asset.index[
                            (returns_by_asset.index >= start_date) & (returns_by_asset.index <= end_date)]:
                            idx = wh.index.searchsorted(d, side='right') - 1
                            if idx < 0:
                                continue
                            row = wh.iloc[idx]
                            wopt = np.array([row.get(f"{a}_Optimal", 0.0) for a in common_wh])
                            wbmk = np.array([row.get(f"{a}_Benchmark", 0.0) for a in common_wh])
                            rows.append(pd.Series(wopt - wbmk, index=common_wh, name=d))
                        Wact_period = pd.DataFrame(rows).sort_index()

                if not Wact_period.empty:
                    pair_contrib = estimate_pair_contributions_nnls(Wact_period, returns_by_asset, timeline_history,
                                                                    start_date, end_date)

            if pair_contrib.empty:
                st.warning("저장된 Attribution/추정 불가 → 자산별 기여만 표시")
                if not pnl_ai.empty:
                    asset_contrib = pnl_ai.sum(axis=0).sort_values(ascending=False) * 10000
                    fig = go.Figure(data=[go.Bar(x=asset_contrib.index, y=asset_contrib.values)])
                    fig.update_layout(yaxis_tickformat=".3f", yaxis_title="기여도 (bp)")
                    fig = apply_chart_font_settings(fig)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                show_cols = [c for c in ['Pair_ID', 'Long_Asset', 'Short_Asset', 'Contribution_bp', 'Share_of_Active_%']
                             if c in pair_contrib.columns]
                pair_display = pair_contrib[show_cols].head(20).copy()
                if 'Contribution_bp' in pair_display.columns:
                    pair_display['Contribution_bp'] = pair_display['Contribution_bp'].apply(lambda x: f"{x:.3f}")
                st.dataframe(pair_display, use_container_width=True)

    # =========================================================================
    # Tab 2: 액티브 포지션
    # =========================================================================
    # =========================================================================
    # Tab 2: 액티브 포지션 (Signal 조정 반영)
    # =========================================================================
    with tabs[2]:
        st.header("⚖️ 현재 액티브 포지션 (as-of, bp 단위)")

        # Signal 조정 확인
        views_for_position = st.session_state.get('adjusted_views')
        if views_for_position is not None:
            st.success("✅ Asset View 탭에서 조정한 Signal이 반영됩니다")
        else:
            views_for_position = views_source
            st.info("ℹ️ 원본 Signal을 사용합니다")

        # 공통 포지션 확인
        if 'common_positions' not in st.session_state or st.session_state.common_positions is None:
            st.warning("⚠️ **포지션이 계산되지 않았습니다.**")
            st.info("Asset View 탭에서 포지션을 먼저 계산하세요.")
            st.stop()

        common_positions = st.session_state.common_positions
        constraint_method = st.session_state.get('constraint_method', '-3STD')

        # As-of 날짜
        if not w_opt_daily.empty and not w_bmk_daily.empty:
            asof = min(w_opt_daily.index.max(), w_bmk_daily.index.max())
            st.caption(f"📅 As-of: {asof.date()}")
        elif not weight_history.empty:
            asof = weight_history.index.max()
            st.caption(f"📅 As-of: {asof.date()} (from weight_history)")
        else:
            asof = None
            st.warning("As-of 날짜를 확인할 수 없습니다.")
            st.stop()

        # Pair별 포지션 구성
        st.subheader("📊 Pair별 Active 포지션")

        # 포지션 요약
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_notional = common_positions['Total_Notional_bp'].abs().sum()
            st.metric("총 Notional", f"{total_notional:.1f}bp")
        with col2:
            avg_position = common_positions['Per_Leg_Position_bp'].abs().mean()
            st.metric("평균 Per-Leg", f"{avg_position:.2f}bp")
        with col3:
            n_pairs = len(common_positions)
            st.metric("활성 Pair", f"{n_pairs}개")
        with col4:
            st.metric("제약 방법", CONSTRAINT_DISPLAY_MAP.get(constraint_method, constraint_method))

        # 상세 테이블
        st.markdown("---")
        display_cols = [
            'Pair_ID', 'Pair', 'Long_Asset', 'Short_Asset', 'Signal',
            'Per_Leg_Position_bp', 'Total_Notional_bp', 'Leg_Factor',
            'Risk_Unit_3M_%', 'Max_Loss_bp'
        ]

        position_display = common_positions[display_cols].copy()

        # 포맷팅
        position_display['Signal'] = position_display['Signal'].apply(lambda x: f"{x:.1f}")
        position_display['Per_Leg_Position_bp'] = position_display['Per_Leg_Position_bp'].apply(lambda x: f"{x:.3f}")
        position_display['Total_Notional_bp'] = position_display['Total_Notional_bp'].apply(lambda x: f"{x:.3f}")
        position_display['Risk_Unit_3M_%'] = position_display['Risk_Unit_3M_%'].apply(lambda x: f"{x:.3f}")
        position_display['Max_Loss_bp'] = position_display['Max_Loss_bp'].apply(lambda x: f"{x:.3f}")

        st.dataframe(position_display, use_container_width=True)

        # 포지션 차트
        st.markdown("---")
        st.subheader("📊 Pair별 포지션 크기")

        fig = go.Figure()

        # Signal에 따른 색상
        colors = ['#2ca02c' if s > 0 else '#d62728' for s in common_positions['Signal']]

        fig.add_trace(go.Bar(
            x=common_positions['Pair'],
            y=common_positions['Total_Notional_bp'],
            marker_color=colors,
            text=common_positions['Total_Notional_bp'].apply(lambda x: f"{x:.1f}"),
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Total Notional: %{y:.3f}bp<extra></extra>"
        ))

        fig.update_layout(
            title="Pair별 Total Notional (bp)",
            xaxis_title="Pair",
            yaxis_title="Total Notional (bp)",
            height=450,
            showlegend=False
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig = apply_chart_font_settings(fig)
        st.plotly_chart(fig, use_container_width=True)

        # 자산별 집계된 Active Weight
        st.markdown("---")
        st.subheader("📊 자산별 Active Weight")

        # Incidence matrix로 자산별 집계
        pairs = [(str(r['Long_Asset']), str(r['Short_Asset'])) for _, r in common_positions.iterrows()]
        signals = common_positions['Signal'].values

        if not returns_by_asset.empty:
            assets_list = returns_by_asset.columns.tolist()
            B = build_incidence_matrix(assets_list, pairs)

            if B.size > 0:
                # Per-leg 포지션 (소수)
                x_pair = common_positions['Per_Leg_Position_bp'].values / 10000.0

                # 자산별 Active weight
                asset_active = pd.Series(B @ x_pair, index=assets_list)
                asset_active = asset_active[asset_active != 0].sort_values(ascending=False)

                # ✅ 수정: 슬라이더 조건부 표시
                max_assets = len(asset_active)

                if max_assets == 0:
                    st.info("활성 자산이 없습니다.")
                elif max_assets == 1:
                    # 자산이 1개면 슬라이더 없이 바로 표시
                    n_display = 1
                    st.caption("표시할 자산: 1개")
                elif max_assets == 2:
                    # 자산이 2개면 슬라이더 없이 바로 표시
                    n_display = 2
                    st.caption("표시할 자산: 2개")
                else:
                    # 자산이 3개 이상일 때만 슬라이더 표시
                    min_display = min(3, max_assets)
                    max_display = min(30, max_assets)
                    default_display = min(15, max_assets)

                    n_display = st.slider(
                        "표시할 자산 수",
                        min_value=min_display,
                        max_value=max_display,
                        value=default_display,
                        key="asset_position_display"
                    )

                # 테이블
                asset_display = pd.DataFrame({
                    'Asset': asset_active.head(n_display).index,
                    'Active_Weight_bp': (asset_active.head(n_display) * 10000).values
                })
                asset_display['Active_Weight_bp'] = asset_display['Active_Weight_bp'].apply(lambda x: f"{x:.3f}")
                st.dataframe(asset_display, use_container_width=True)

                # 차트
                fig_asset = go.Figure()

                fig_asset.add_trace(go.Bar(
                    x=asset_active.head(n_display).index,
                    y=asset_active.head(n_display).values * 10000,
                    marker_color=['#2ca02c' if v > 0 else '#d62728' for v in asset_active.head(n_display)],
                    hovertemplate="%{x}: %{y:.3f}bp<extra></extra>"
                ))

                fig_asset.update_layout(
                    title="자산별 Active Weight (bp)",
                    xaxis_title="Asset",
                    yaxis_title="Active Weight (bp)",
                    height=400,
                    showlegend=False
                )
                fig_asset.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                fig_asset = apply_chart_font_settings(fig_asset)
                st.plotly_chart(fig_asset, use_container_width=True)
            else:
                st.info("Incidence matrix 생성 실패")
        else:
            st.info("시장 수익률 데이터가 필요합니다")



    # =========================================================================
    # Tab 3: 일별 성과 (Signal 조정 반영 + 진단 기능)
    # =========================================================================
    with tabs[3]:
        st.header("📊 일별 성과 (bp 단위)")

        # ===== Inception Date 설정 =====
        st.subheader("📅 Inception Date 설정")

        col_inc1, col_inc2 = st.columns([1, 3])

        with col_inc1:
            use_inception = st.checkbox(
                "Inception Date 사용",
                value=True,
                key="use_inception_tab3",
                help="특정 날짜부터의 성과를 계산합니다"
            )

        with col_inc2:
            if use_inception:
                default_inception = pd.Timestamp.now() - pd.Timedelta(days=90)

                if not returns_by_asset.empty:
                    min_date = returns_by_asset.index.min()
                    max_date = returns_by_asset.index.max()
                else:
                    min_date = default_inception
                    max_date = pd.Timestamp.now()

                inception_date = st.date_input(
                    "Inception Date",
                    value=max(default_inception.date(), min_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="inception_date_tab3",
                    help="이 날짜부터의 수익률을 계산합니다"
                )
                inception_date = pd.Timestamp(inception_date)
                st.success(f"✅ Inception: {inception_date.strftime('%Y-%m-%d')}")
            else:
                inception_date = None
                st.info("전체 기간의 성과를 표시합니다")

        st.markdown("---")

        # ===== Signal 조정 여부 확인 =====
        views_to_use = st.session_state.get('adjusted_views')
        signal_adjusted = (views_to_use is not None)

        if signal_adjusted:
            st.success("✅ **Asset View 탭에서 조정한 Signal이 반영됩니다**")

            with st.expander("📝 조정된 Signal 확인", expanded=False):
                if not views_to_use.empty:
                    signal_summary = views_to_use[['Pair_ID', 'Long_Asset', 'Short_Asset', 'Signal']].copy()
                    signal_summary['Signal'] = signal_summary['Signal'].apply(lambda x: f"{x:.1f}")
                    st.dataframe(signal_summary, use_container_width=True)
        else:
            views_to_use = views_source
            st.info("ℹ️ 원본 Signal을 사용합니다. Asset View 탭에서 Signal을 조정할 수 있습니다.")

        # ===== 포지션 및 설정 확인 =====
        if 'common_positions' not in st.session_state or st.session_state.common_positions is None:
            st.warning("⚠️ **포지션이 계산되지 않았습니다.**")
            st.info("""
            💡 **다음 단계를 진행하세요:**
            1. **Asset View 탭**으로 이동
            2. 포지션 크기 결정 기준 선택
            3. Signal 조정 (필요시)
            4. 포지션 계산 후 이 탭으로 돌아오세요
            """)
            st.stop()

        common_positions = st.session_state.common_positions
        constraint_method = st.session_state.get('constraint_method', '-3STD')
        lookback_years = st.session_state.get('lookback_years', 3)

        # ===== 일별 수익률 계산 =====
        st.subheader("🔄 일별 수익률 계산")

        if returns_by_asset.empty:
            st.error("시장 수익률 데이터가 필요합니다.")
            st.stop()

        if views_to_use.empty:
            st.warning("활성 Pair 정보가 필요합니다.")
            st.stop()

        active_views = views_to_use[views_to_use['Signal'] != 0].copy()

        if active_views.empty:
            st.info("현재 활성 Pair가 없습니다.")
            st.stop()

        # Pairs, signals, IDs
        pairs = [(str(r['Long_Asset']), str(r['Short_Asset']))
                 for _, r in active_views.iterrows()]
        signals = active_views['Signal'].astype(float).values
        pair_ids = active_views.get('Pair_ID', range(len(pairs))).values

        # 포지션 맵 생성
        position_map = dict(zip(
            common_positions['Pair_ID'],
            common_positions['Per_Leg_Position_bp'] / 10000  # bp → 소수
        ))
        leg_factor_map = dict(zip(
            common_positions['Pair_ID'],
            common_positions['Leg_Factor']
        ))
        signal_map = dict(zip(
            common_positions['Pair_ID'],
            common_positions['Signal']
        ))

        # ===== 일별 P&L 계산 =====
        with st.spinner("일별 성과 계산 중..."):
            pair_daily_pnl = {}

            for pid, (la, sa) in zip(pair_ids, pairs):
                if la not in returns_by_asset.columns or sa not in returns_by_asset.columns:
                    continue

                # Signal 가져오기
                signal = signal_map.get(pid, 1.0)

                # ✅ 스프레드 수익률 계산 (항상 Long - Short)
                spread_ret = returns_by_asset[la] - returns_by_asset[sa]

                # ✅ 포지션 크기 (부호 포함 - 이미 Signal 방향 반영됨)
                signed_pos = position_map.get(pid, 0.0)
                legs = leg_factor_map.get(pid, 2)

                # ✅ 일별 P&L 계산
                # Signal > 0: (Long - Short) * (+pos) * legs = 올바름
                # Signal < 0: (Long - Short) * (-pos) * legs = 올바름 (부호 반전)
                daily_pnl = spread_ret * signed_pos * legs

                pair_daily_pnl[pid] = daily_pnl

            # 전체 포트폴리오 일별 수익률
            if pair_daily_pnl:
                portfolio_daily_returns = sum(pair_daily_pnl.values())

                # DataFrame 생성
                daily_returns = pd.DataFrame({
                    'Portfolio_Return': portfolio_daily_returns
                })

                # Benchmark 추가
                if not w_bmk_daily.empty:
                    common_assets = [a for a in returns_by_asset.columns if a in w_bmk_daily.columns]
                    if common_assets:
                        bmk_returns = []
                        for date in portfolio_daily_returns.index:
                            if date in w_bmk_daily.index:
                                w_bmk = w_bmk_daily.loc[date, common_assets].fillna(0.0)
                            else:
                                prev_dates = w_bmk_daily.index[w_bmk_daily.index <= date]
                                if len(prev_dates) > 0:
                                    w_bmk = w_bmk_daily.loc[prev_dates[-1], common_assets].fillna(0.0)
                                else:
                                    w_bmk = pd.Series(0.0, index=common_assets)

                            bmk_ret = (w_bmk * returns_by_asset.loc[date, common_assets]).sum()
                            bmk_returns.append(bmk_ret)

                        daily_returns['Benchmark_Return'] = bmk_returns
                        daily_returns['Active_Return'] = daily_returns['Portfolio_Return'] - daily_returns[
                            'Benchmark_Return']
                else:
                    daily_returns['Benchmark_Return'] = 0.0
                    daily_returns['Active_Return'] = daily_returns['Portfolio_Return']

                # Session state 저장
                st.session_state['daily_returns_recalculated'] = daily_returns
                st.session_state['pair_daily_pnl'] = pair_daily_pnl

                st.success(f"✅ 일별 수익률 계산 완료: {len(daily_returns)}일")

                if signal_adjusted:
                    st.info(f"💡 조정된 Signal이 반영되었습니다 ({constraint_method}, {lookback_years}년)")
            else:
                st.error("일별 수익률 계산에 실패했습니다.")
                st.stop()

        # Inception Date 필터링
        if inception_date is not None:
            daily_returns = daily_returns[daily_returns.index >= inception_date].copy()

            if daily_returns.empty:
                st.error(f"⚠️ {inception_date.date()} 이후 데이터가 없습니다.")
                st.stop()

        # ===== 포지션 요약 =====
        with st.expander("📋 현재 적용된 포지션 요약", expanded=False):
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)

            with col_s1:
                total_notional = common_positions['Total_Notional_bp'].abs().sum()
                st.metric("총 Notional", f"{total_notional:.1f}bp")
            with col_s2:
                avg_position = common_positions['Per_Leg_Position_bp'].abs().mean()
                st.metric("평균 Per-Leg", f"{avg_position:.2f}bp")
            with col_s3:
                n_pairs = len(common_positions)
                st.metric("활성 Pair", f"{n_pairs}개")
            with col_s4:
                avg_risk = common_positions['Risk_Unit_3M_%'].mean()
                st.metric("평균 리스크", f"{avg_risk:.2f}%")

            st.dataframe(
                common_positions[[
                    'Pair_ID', 'Pair', 'Signal', 'Per_Leg_Position_bp',
                    'Total_Notional_bp', 'Risk_Unit_3M_%'
                ]].style.format({
                    'Signal': '{:.1f}',
                    'Per_Leg_Position_bp': '{:.3f}',
                    'Total_Notional_bp': '{:.3f}',
                    'Risk_Unit_3M_%': '{:.3f}'
                }),
                use_container_width=True
            )

        # ===== 진단: Pair별 최근 성과 =====
        st.markdown("---")
        if st.checkbox("🔍 Pair별 최근 성과 진단", value=False, key="show_pair_diagnosis"):
            st.subheader("🔍 Pair별 최근 5일 성과 진단")

            if 'pair_daily_pnl' in st.session_state:
                pair_pnl_dict = st.session_state['pair_daily_pnl']

                # 최근 5일 데이터
                n_days_to_show = min(5, len(daily_returns))
                recent_dates = daily_returns.index[-n_days_to_show:]

                for pid in pair_ids:
                    if pid not in pair_pnl_dict:
                        continue

                    pair_info = common_positions[common_positions['Pair_ID'] == pid]
                    if pair_info.empty:
                        continue

                    pair_row = pair_info.iloc[0]
                    pair_name = pair_row['Pair']
                    signal = pair_row['Signal']
                    position_bp = pair_row['Per_Leg_Position_bp']
                    legs = pair_row['Leg_Factor']

                    with st.expander(f"**{pair_name}** (Signal: {signal:.1f}, Position: {position_bp:.3f}bp)",
                                     expanded=False):
                        # 최근 5일 P&L
                        pnl_series = pair_pnl_dict[pid]
                        recent_pnl = pnl_series.loc[recent_dates]

                        # 자산 수익률
                        la, sa = pair_name.split(' vs ')

                        if la in returns_by_asset.columns and sa in returns_by_asset.columns:
                            asset_rets = returns_by_asset.loc[recent_dates, [la, sa]] * 100  # % 변환

                            # 테이블 생성
                            diag_df = pd.DataFrame({
                                'Date': [d.strftime('%Y-%m-%d') for d in recent_dates],
                                f'{la} (%)': asset_rets[la].values,
                                f'{sa} (%)': asset_rets[sa].values,
                                'Spread (%)': (asset_rets[la] - asset_rets[sa]).values,
                                'Position': [f"{position_bp:.3f}bp"] * len(recent_dates),
                                'Legs': [legs] * len(recent_dates),
                                'Pair PnL (bp)': (recent_pnl * 10000).values
                            })

                            # 포맷팅
                            for col in [f'{la} (%)', f'{sa} (%)', 'Spread (%)']:
                                diag_df[col] = diag_df[col].apply(lambda x: f"{x:.2f}")
                            diag_df['Pair PnL (bp)'] = diag_df['Pair PnL (bp)'].apply(lambda x: f"{x:.3f}")

                            st.dataframe(diag_df, use_container_width=True)

                            # 최근 날짜 검증
                            last_date = recent_dates[-1]
                            last_la_ret = asset_rets[la].iloc[-1]
                            last_sa_ret = asset_rets[sa].iloc[-1]
                            last_spread = last_la_ret - last_sa_ret
                            last_pnl = recent_pnl.iloc[-1] * 10000

                            # 방향 검증
                            if signal > 0:
                                expected_sign = "양수" if last_spread > 0 else "음수"
                            else:
                                expected_sign = "양수" if last_spread < 0 else "음수"

                            actual_sign = "양수" if last_pnl > 0 else "음수"

                            st.markdown(f"""
                            **{last_date.strftime('%Y-%m-%d')} 검증:**
                            - {la}: {last_la_ret:.2f}% | {sa}: {last_sa_ret:.2f}% 
                            - Spread: {last_spread:.2f}% 
                            - Signal: {signal:.1f} ({"Long" if signal > 0 else "Short"})
                            - Position: {position_bp:.3f}bp × {legs}legs
                            - P&L: {last_pnl:.3f}bp
                            """)

                            if expected_sign == actual_sign:
                                st.success(f"✅ 방향 일치: 예상 {expected_sign}, 실제 {actual_sign}")
                            else:
                                st.error(f"⚠️ 방향 불일치! 예상 {expected_sign}, 실제 {actual_sign}")
                        else:
                            st.warning(f"자산 수익률 데이터를 찾을 수 없습니다: {la}, {sa}")

        # ===== 성과 지표 계산 =====
        st.markdown("---")
        st.subheader("📊 성과 지표")

        metrics_data = {}

        for col in ['Portfolio_Return', 'Benchmark_Return', 'Active_Return']:
            if col in daily_returns.columns:
                series = daily_returns[col].dropna()

                if len(series) == 0:
                    continue

                # 누적 수익률
                cum_ret = (1 + series).prod() - 1

                # 거래일 수
                n_days = len(series)

                # 연율화 수익률
                if n_days > 0:
                    ann_ret = (1 + cum_ret) ** (252 / n_days) - 1
                else:
                    ann_ret = 0.0

                # 연율화 변동성
                ann_vol = series.std() * np.sqrt(252)

                # Sharpe Ratio
                if col == 'Active_Return' and ann_vol > 0:
                    sharpe = ann_ret / ann_vol
                else:
                    sharpe = np.nan

                # MDD 계산
                cum_series = (1 + series).cumprod()
                running_max = cum_series.expanding().max()
                drawdown = (cum_series - running_max) / running_max
                mdd = drawdown.min()

                metrics_data[col] = {
                    'cumulative': cum_ret * 10000,
                    'annualized': ann_ret * 10000,
                    'volatility': ann_vol * 10000,
                    'sharpe': sharpe,
                    'mdd': mdd * 100,
                    'n_days': n_days
                }

        # 3단 레이아웃
        if metrics_data:
            cols = st.columns(3)

            col_names = {
                'Portfolio_Return': '📈 포트폴리오',
                'Benchmark_Return': '📉 벤치마크',
                'Active_Return': '⚡ 초과수익'
            }

            for idx, (col_key, col_label) in enumerate(col_names.items()):
                if col_key in metrics_data:
                    with cols[idx]:
                        st.markdown(f"### {col_label}")
                        metrics = metrics_data[col_key]

                        st.metric("누적 수익률", f"{metrics['cumulative']:.3f}bp")
                        st.metric("연율화 수익률", f"{metrics['annualized']:.3f}bp")
                        st.metric("연율화 변동성", f"{metrics['volatility']:.3f}bp")

                        if col_key == 'Active_Return' and not np.isnan(metrics['sharpe']):
                            st.metric("Sharpe Ratio", f"{metrics['sharpe']:.3f}")

                        st.metric("MDD", f"{metrics['mdd']:.2f}%")
                        st.caption(f"거래일: {metrics['n_days']}일")

        # ===== 누적 수익률 그래프 =====
        st.markdown("---")
        st.subheader("📈 누적 수익률 추이")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("누적 수익률 (bp)", "일별 초과수익 (bp)"),
            vertical_spacing=0.12,
            row_heights=[0.65, 0.35]
        )

        cum_returns = (1 + daily_returns).cumprod() - 1

        colors = {
            'Portfolio_Return': '#1f77b4',
            'Benchmark_Return': '#7f7f7f',
            'Active_Return': '#2ca02c'
        }

        names = {
            'Portfolio_Return': '포트폴리오',
            'Benchmark_Return': '벤치마크',
            'Active_Return': '초과수익'
        }

        for col in ['Portfolio_Return', 'Benchmark_Return', 'Active_Return']:
            if col in cum_returns.columns:
                fig.add_trace(
                    go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns[col] * 10000,
                        name=names[col],
                        line=dict(color=colors[col], width=2.5),
                        hovertemplate=f"{names[col]}: %{{y:.3f}}bp<extra></extra>"
                    ),
                    row=1, col=1
                )

        if 'Active_Return' in daily_returns.columns:
            bar_colors = ['#2ca02c' if r > 0 else '#d62728'
                          for r in daily_returns['Active_Return']]

            fig.add_trace(
                go.Bar(
                    x=daily_returns.index,
                    y=daily_returns['Active_Return'] * 10000,
                    marker_color=bar_colors,
                    showlegend=False,
                    hovertemplate="초과수익: %{y:.3f}bp<extra></extra>"
                ),
                row=2, col=1
            )

        fig.update_xaxes(title_text="날짜", row=2, col=1)
        fig.update_yaxes(title_text="누적 수익률 (bp)", row=1, col=1, tickformat=".3f")
        fig.update_yaxes(title_text="일별 초과수익 (bp)", row=2, col=1, tickformat=".3f")

        fig.update_layout(
            height=700,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

        fig = apply_chart_font_settings(fig)
        st.plotly_chart(fig, use_container_width=True)

        # ===== Drawdown 차트 =====
        st.markdown("---")
        st.subheader("📉 Drawdown 추이")

        fig_dd = go.Figure()

        for col in ['Portfolio_Return', 'Active_Return']:
            if col in daily_returns.columns:
                cum_series = (1 + daily_returns[col]).cumprod()
                running_max = cum_series.expanding().max()
                drawdown = (cum_series - running_max) / running_max

                fig_dd.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown * 100,
                        name=names[col],
                        line=dict(color=colors[col], width=2),
                        fill='tozeroy',
                        hovertemplate=f"{names[col]}: %{{y:.2f}}%<extra></extra>"
                    )
                )

        fig_dd.update_layout(
            title="Drawdown (%)",
            xaxis_title="날짜",
            yaxis_title="Drawdown (%)",
            height=400,
            hovermode='x unified'
        )

        fig_dd.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_dd = apply_chart_font_settings(fig_dd)
        st.plotly_chart(fig_dd, use_container_width=True)

        # ===== 기간별 성과 =====
        st.markdown("---")
        st.subheader("📊 기간별 성과")

        periods = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252}
        period_results = []

        for period_name, days in periods.items():
            if len(daily_returns) < days:
                continue

            period_data = daily_returns.tail(days)

            for col in ['Portfolio_Return', 'Benchmark_Return', 'Active_Return']:
                if col not in period_data.columns:
                    continue

                series = period_data[col].dropna()
                if len(series) == 0:
                    continue

                cum_ret = (1 + series).prod() - 1
                ann_ret = (1 + cum_ret) ** (252 / len(series)) - 1
                ann_vol = series.std() * np.sqrt(252)

                period_results.append({
                    '기간': period_name,
                    '유형': names[col],
                    '누적 (bp)': f"{cum_ret * 10000:.3f}",
                    '연율화 (bp)': f"{ann_ret * 10000:.3f}",
                    '변동성 (bp)': f"{ann_vol * 10000:.3f}"
                })

        if period_results:
            period_df = pd.DataFrame(period_results)

            for metric in ['누적 (bp)', '연율화 (bp)', '변동성 (bp)']:
                st.markdown(f"**{metric}**")
                pivot = period_df.pivot(index='기간', columns='유형', values=metric)
                st.dataframe(pivot, use_container_width=True)
                st.markdown("")

        # ===== 데이터 다운로드 =====
        st.markdown("---")
        st.subheader("📥 데이터 다운로드")

        download_df = daily_returns.copy()
        for col in download_df.columns:
            download_df[col] = download_df[col] * 10000

        csv_data = download_df.to_csv().encode('utf-8-sig')

        filename_suffix = f"from_{inception_date.strftime('%Y%m%d')}" if inception_date else "full_period"
        signal_suffix = "_adjusted" if signal_adjusted else "_original"

        st.download_button(
            label="📥 일별 수익률 다운로드 (CSV)",
            data=csv_data,
            file_name=f"daily_returns_{filename_suffix}{signal_suffix}_{constraint_method}.csv",
            mime="text/csv",
            key="download_daily_returns_tab3"
        )

        # 정보 표시
        info_parts = []
        if inception_date:
            info_parts.append(f"📅 Inception: {inception_date.strftime('%Y-%m-%d')}")
        info_parts.append(f"📊 거래일: {len(daily_returns)}일")
        info_parts.append(f"🎯 제약: {CONSTRAINT_DISPLAY_MAP.get(constraint_method, constraint_method)}")
        if signal_adjusted:
            info_parts.append("✅ Signal 조정 반영")

        st.info(" | ".join(info_parts))


    # =========================================================================
    # Tab 4: 리밸런싱 분석
    # =========================================================================
    with tabs[4]:
        st.header("🔄 리밸런싱 분석")
        display_rebalance_log(rebalance_log_df)
        if not rebalance_calendar_df.empty:
            st.subheader("📅 리밸런싱 캘린더")
            st.dataframe(rebalance_calendar_df, use_container_width=True)

    # =========================================================================
    # Tab 5: 가중치 추적
    # =========================================================================
    with tabs[5]:
        st.header("⚡ 가중치 추적 (bp 단위)")
        checkpoints = data.get("weights_checkpoints", pd.DataFrame())
        if checkpoints.empty and not weight_history.empty:
            df = weight_history.copy()
            has_active = any(c.endswith("_Active") for c in df.columns)
            if not has_active:
                base_assets = sorted(
                    {c.replace('_Optimal', '').replace('_Benchmark', '').replace('_Active', '') for c in df.columns}
                )
                for a in base_assets:
                    df[f"{a}_Active"] = df.get(f"{a}_Optimal", 0.0) - df.get(f"{a}_Benchmark", 0.0)
            checkpoints = df
        display_checkpoint_weights(checkpoints)

    # =========================================================================
    # Tab 6: 포트폴리오 개요
    # =========================================================================
    with tabs[6]:
        st.header("📋 포트폴리오 개요")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Performance Metrics (bp 단위)")
            pm = data.get("performance_metrics", pd.DataFrame())
            if not pm.empty:
                pm_display = pm.copy()
                if "Metric" in pm_display.columns and "Value" in pm_display.columns:
                    for idx, row in pm_display.iterrows():
                        metric_name = str(row["Metric"]).lower()
                        if any(k in metric_name for k in ["tracking_error", "volatility", "active_return", "return"]):
                            pm_display.at[idx, "Value"] = f"{row['Value'] * 10000:.3f}"
                st.dataframe(pm_display, use_container_width=True)
            else:
                st.info("Performance Metrics 파일이 없습니다.")
        with c2:
            st.subheader("Portfolio Weights (Latest, bp 단위)")
            if not weights_df.empty:
                weights_display = weights_df.copy()
                for col in ["Optimal_Weight", "Benchmark_Weight", "Active_Weight"]:
                    if col in weights_display.columns:
                        weights_display[col] = weights_display[col].apply(lambda x: f"{x * 10000:.3f}")
                st.dataframe(weights_display, use_container_width=True)
            else:
                st.info("Portfolio Weights 파일이 없습니다.")

    # =========================================================================
    # Tab 7: 리스크 분석
    # =========================================================================
    with tabs[7]:
        st.header("⚠️ 리스크 분석 (bp 단위)")

        if cov_matrix.empty:
            st.info("공분산 행렬이 없습니다.")
        else:
            if not w_opt_daily.empty and not w_bmk_daily.empty:
                asof = min(w_opt_daily.index.max(), w_bmk_daily.index.max())
                Wopt_last = w_opt_daily.loc[asof].fillna(0.0)
                Wbmk_last = w_bmk_daily.loc[asof].fillna(0.0)
                st.caption(f"As-of: {asof.date()}")
            elif not weight_history.empty:
                asof = weight_history.index.max()
                row = weight_history.loc[asof]
                assets_wh = sorted({c.replace('_Optimal', '').replace('_Benchmark', '').replace('_Active', '') for c in
                                    weight_history.columns})
                Wopt_last = pd.Series({a: row.get(f"{a}_Optimal", 0.0) for a in assets_wh})
                Wbmk_last = pd.Series({a: row.get(f"{a}_Benchmark", 0.0) for a in assets_wh})
                st.caption(f"As-of: {asof.date()} (from weight_history)")
            else:
                Wopt_last = pd.Series(dtype=float)
                Wbmk_last = pd.Series(dtype=float)

            if Wopt_last.empty or Wbmk_last.empty:
                st.info("현재 포지션 정보를 찾을 수 없어 TE 계산을 생략합니다.")
            else:
                Wact_last = (Wopt_last - Wbmk_last).fillna(0.0)
                assets_list = [a for a in Wact_last.index if a in cov_matrix.index]
                te = compute_te_from_active(cov_matrix, assets_list, Wact_last.reindex(assets_list))
                st.metric("Tracking Error (연율)", f"{te * 10000:.3f}bp")

                cov_use = cov_matrix.reindex(index=assets_list, columns=assets_list).fillna(0.0).values
                w = Wact_last.reindex(assets_list).fillna(0.0).values
                if np.any(np.isfinite(cov_use)) and np.any(np.isfinite(w)):
                    mct = cov_use @ w
                    cont = w * mct
                    cont_series = pd.Series(cont, index=assets_list).sort_values(ascending=False)

                    # BP 단위로 변환
                    cont_series_bp = cont_series * 10000

                    st.subheader("TE 기여(근사) Top 15 (bp 단위)")
                    fig = go.Figure(data=[go.Bar(
                        x=cont_series_bp.head(15).index,
                        y=cont_series_bp.head(15).values,
                        hovertemplate="%{y:.3f}bp<extra></extra>"
                    )])
                    fig.update_layout(
                        yaxis_tickformat=".3f",
                        yaxis_title="TE 기여 (bp)",
                        height=500
                    )
                    fig = apply_chart_font_settings(fig)
                    st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # Tab 8: 상관관계 분석
    # =========================================================================
    with tabs[8]:
        st.header("🔗 상관관계 분석")

        if returns_by_asset.empty:
            st.info("상관관계를 계산할 수익률 데이터가 없습니다.")
        else:
            # 데이터 품질 확인
            st.subheader("📊 데이터 품질 확인")
            data_quality = pd.DataFrame({
                'Asset': returns_by_asset.columns,
                'Total_Rows': len(returns_by_asset),
                'Valid_Rows': returns_by_asset.notna().sum().values,
                'Valid_Ratio': (returns_by_asset.notna().sum() / len(returns_by_asset)).values,
                'Mean': returns_by_asset.mean().values,
                'Std': returns_by_asset.std().values
            })
            data_quality['Valid_Ratio_%'] = (data_quality['Valid_Ratio'] * 100).round(2)

            # 데이터가 부족한 자산 표시
            low_quality = data_quality[data_quality['Valid_Ratio'] < 0.8]
            if not low_quality.empty:
                st.warning(f"⚠️ {len(low_quality)}개 자산의 데이터 비율이 80% 미만입니다:")
                st.dataframe(low_quality[['Asset', 'Valid_Rows', 'Valid_Ratio_%']])

            with st.expander("전체 데이터 품질 보기"):
                st.dataframe(data_quality)

            # Rolling correlation
            st.markdown("---")
            st.subheader("📈 Rolling Correlation")
            window = st.slider("롤링 상관 윈도우(일)", 20, 252, 60, step=5)
            rc = calculate_rolling_correlation(returns_by_asset, window=window)

            if rc:
                pair = st.selectbox("페어 선택", list(rc.keys()))
                series = rc[pair]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=pair))
                fig.update_layout(height=400, title=f"{pair} 롤링 상관({window}D)")
                fig = apply_chart_font_settings(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("계산 가능한 rolling correlation이 없습니다. 데이터 품질을 확인하세요.")

            # Correlation matrix
            st.markdown("---")
            st.subheader("🗺️ 상관관계 히트맵")

            lookback_days = st.slider("사용 데이터 기간(일)", 60, 756, 252, step=30)
            recent_returns = returns_by_asset.tail(lookback_days)

            valid_ratio = recent_returns.notna().sum() / len(recent_returns)
            assets_to_use = valid_ratio[valid_ratio >= 0.7].index.tolist()

            if len(assets_to_use) < 2:
                st.error("유효한 데이터가 있는 자산이 부족합니다 (최소 2개 필요).")
            else:
                st.info(f"✅ {len(assets_to_use)}개 자산으로 상관관계 계산 (70% 이상 유효 데이터)")

                corr_matrix_viz = recent_returns[assets_to_use].corr()

                fig = px.imshow(
                    corr_matrix_viz,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    zmin=-1, zmax=1
                )
                fig.update_layout(height=700, title=f"상관관계 히트맵 ({lookback_days}일 기준)")
                fig = apply_chart_font_settings(fig)
                st.plotly_chart(fig, use_container_width=True)

            # Stability
            st.markdown("---")
            st.subheader("📊 상관 안정성(윈도우별 표준편차) Heatmap")
            stab = calculate_correlation_stability(returns_by_asset, window=window)

            if not stab.empty:
                fig = px.imshow(stab, text_auto='.3f', aspect='auto', color_continuous_scale='RdBu_r', origin='lower')
                fig.update_layout(height=600, title=f"상관관계 안정성 ({window}일 윈도우)")
                fig = apply_chart_font_settings(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("안정성 히트맵을 계산할 구간이 부족합니다.")

    # =========================================================================
    # Tab 9: Pair 기대수익률 (3M Rolling) - 조정된 Signal 반영
    # =========================================================================
    with tabs[9]:
        st.header("🎲 Pair 전략 기대수익률 분석 (3개월 Rolling Return 기준)")
        st.markdown("""
        이 탭에서는 선정된 Pair 전략의 **3개월 rolling return** 분포를 기반으로  
        **-3σ, -2σ, -1σ, +1σ, +2σ, +3σ** 수준에서의 기대수익률을 계산합니다.

        💡 **3개월 Rolling Return**: 일별 수익률을 63영업일(약 3개월) 단위로 누적한 수익률  
        💡 **EWM 통계**: 최근 데이터에 더 높은 가중치 (halflife=126일)

        ✅ **Asset View 탭에서 조정한 Signal과 포지션이 자동으로 반영됩니다.**
        """)

        # ===== 공통 포지션 확인 =====
        if 'common_positions' not in st.session_state or st.session_state.common_positions is None:
            st.warning("⚠️ **포지션이 계산되지 않았습니다.**")
            st.info("""
            💡 **다음 단계를 진행하세요:**
            1. **Asset View 탭**으로 이동
            2. 포지션 크기 결정 기준 선택 (예: -3STD)
            3. Signal 조정 (필요시)
            4. 포지션이 자동으로 계산되고 이 탭에 반영됩니다
            """)
            st.stop()

        if returns_by_asset.empty:
            st.warning("시장 수익률 데이터가 필요합니다.")
            st.stop()

        # Session state에서 설정 가져오기
        common_positions = st.session_state.common_positions
        constraint_method = st.session_state.get('constraint_method', '-3STD')
        lookback_years = st.session_state.get('lookback_years', 3)

        st.success(f"✅ Asset View의 포지션 사용 중 ({constraint_method}, {lookback_years}년)")

        # Views 가져오기
        views_to_use = (
            st.session_state.adjusted_views
            if st.session_state.get("adjusted_views") is not None
            else views_source
        )

        if views_to_use.empty:
            st.warning("활성 Pair 정보가 필요합니다.")
            st.stop()

        # 활성 Pair
        active_views = views_to_use[views_to_use['Signal'] != 0].copy()

        if st.session_state.get("adjusted_views") is not None:
            st.info("ℹ️ Asset View에서 조정한 Signal이 반영되었습니다.")

        if active_views.empty:
            st.info("현재 활성 Pair가 없습니다.")
            st.stop()

        # ✅ pairs, signals, pair_ids 정의
        pairs = [(str(r['Long_Asset']), str(r['Short_Asset']))
                 for _, r in active_views.iterrows()]
        signals = active_views['Signal'].astype(float).values
        pair_ids = active_views.get('Pair_ID', range(len(pairs))).values

        position_map = dict(zip(
            common_positions['Pair_ID'],
            common_positions['Per_Leg_Position_bp'] / 10000  # bp → 소수
        ))
        leg_factor_map = dict(zip(
            common_positions['Pair_ID'],
            common_positions['Leg_Factor']
        ))

        # ===== 3개월 롤링 리턴 계산 =====
        scenarios_data = []

        with st.spinner("3개월 rolling return 계산 중... (EWM 방식)"):
            for idx, (pid, (la, sa)) in enumerate(zip(pair_ids, pairs)):
                pair_name = f"{la} vs {sa}"
                signal = signals[idx]  # ✅ Signal 가져오기

                # ✅ Signal 전달하여 방향 반영
                r3 = calculate_pair_3m_rolling_returns(
                    returns_by_asset, la, sa, signal, lookback_years
                )

                if r3.empty or len(r3) < 2:
                    continue

                # ✅ EWM 통계 (RiskConstraintCalculator와 동일한 방식)
                if len(r3) >= 126:
                    ewm_mean = r3.ewm(halflife=126).mean().iloc[-1]
                    ewm_std = r3.ewm(halflife=126).std().iloc[-1]
                    mu = float(ewm_mean)
                    sd = float(ewm_std)
                else:
                    mu = float(r3.mean())
                    sd = float(r3.std(ddof=1))

                # 시나리오 (소수)
                scenarios = {
                    '-4std': mu - 4.0 * sd,
                    '-3std': mu - 3.0 * sd,
                    '-2std': mu - 2.0 * sd,
                    '-1std': mu - 1.0 * sd,
                    'Mean': mu,
                    '+1std': mu + 1.0 * sd,
                    '+2std': mu + 2.0 * sd,
                    '+3std': mu + 3.0 * sd,
                }

                # 포지션 (공통 포지션 사용)
                signed_pos_per_leg_dec = position_map.get(pid, 0.0)  # 소수
                abs_pos_per_leg_dec = abs(signed_pos_per_leg_dec)
                abs_pos_per_leg_bp = abs_pos_per_leg_dec * 10000
                legs = leg_factor_map.get(pid, 2)

                # ✅ 간단한 Expected Loss 계산
                pos_row = common_positions[common_positions['Pair_ID'] == pid]

                if not pos_row.empty:
                    # Tab 0과 동일한 Risk_Unit 사용
                    risk_unit_decimal = pos_row.iloc[0]['Risk_Unit_3M_%'] / 100.0
                    per_leg_decimal = abs_pos_per_leg_dec
                    legs = int(pos_row.iloc[0]['Leg_Factor'])

                    expected_loss_bp = risk_unit_decimal * per_leg_decimal * legs * 10000
                else:
                    # Fallback (이 경우는 거의 없어야 함)
                    loss_scenario_decimal = abs(scenarios[scenario_key])
                    expected_loss_bp = loss_scenario_decimal * abs_pos_per_leg_dec * legs * 10000

                # Signal별 최대 손실 허용치
                abs_signal = abs(signals[idx])
                if abs_signal >= 2.0:
                    max_loss_bp = 0.15
                elif abs_signal >= 1.0:
                    max_loss_bp = 0.1
                else:
                    max_loss_bp = 0.1 + abs_signal * 0.05

                # ✅ Utilization
                util_pct = (expected_loss_bp / max_loss_bp) * 100.0

                # 포트폴리오 기여도 (Signal 방향 반영)
                scenarios_bp = {
                    k: v * signed_pos_per_leg_dec * legs * 10000
                    for k, v in scenarios.items()
                }

                # 연율화
                mu_ann = mu * 4.0
                sd_ann = sd * np.sqrt(4.0)
                sharpe_3m = (mu / sd) if sd > 0 else np.nan

                scenarios_data.append({
                    'Pair_ID': pid,
                    'Pair': pair_name,
                    'Long': la,
                    'Short': sa,
                    'Signal': float(signals[idx]),
                    'Position_bp': abs_pos_per_leg_bp,
                    'Total_Notional_bp': abs_pos_per_leg_bp * legs,
                    'Legs': int(legs),

                    # 페어 통계 (%)
                    'Pair_Mean_3M_%': mu * 100,
                    'Pair_Std_3M_%': sd * 100,
                    'Sharpe_3M': sharpe_3m,
                    'Pair_Annual_Return_%': mu_ann * 100,
                    'Pair_Annual_Std_%': sd_ann * 100,

                    # 포트폴리오 기여도 (bp)
                    'Portfolio_Mean_3M_bp': scenarios_bp['Mean'],
                    'Portfolio_Std_3M_bp': sd * abs_pos_per_leg_dec * legs * 10000,
                    'Portfolio_Annual_Return_bp': mu_ann * signed_pos_per_leg_dec * legs * 10000,
                    'Portfolio_Annual_Std_bp': sd_ann * abs_pos_per_leg_dec * legs * 10000,

                    # 시나리오 (%)
                    '-3std_%': scenarios['-3std'] * 100,
                    '-2std_%': scenarios['-2std'] * 100,
                    '-1std_%': scenarios['-1std'] * 100,
                    'Mean_%': scenarios['Mean'] * 100,
                    '+1std_%': scenarios['+1std'] * 100,
                    '+2std_%': scenarios['+2std'] * 100,
                    '+3std_%': scenarios['+3std'] * 100,

                    # 시나리오 (bp)
                    '-3std_bp': scenarios_bp['-3std'],
                    '-2std_bp': scenarios_bp['-2std'],
                    '-1std_bp': scenarios_bp['-1std'],
                    '+1std_bp': scenarios_bp['+1std'],
                    '+2std_bp': scenarios_bp['+2std'],
                    '+3std_bp': scenarios_bp['+3std'],

                    # ✅ 손실 점검 (간소화)
                    'Expected_Loss_3M_bp': expected_loss_bp,
                    'Max_Loss_bp': max_loss_bp,
                    'Utilization_%': util_pct,

                    'N_Observations': int(len(r3))
                })
        if not scenarios_data:
            st.warning("계산 가능한 Pair가 없습니다.")
            st.stop()

        scenarios_df = pd.DataFrame(scenarios_data)

        # ===== 포트폴리오 전체 시나리오 =====
        st.subheader("📈 포트폴리오 전체 시나리오 (3M Rolling Return 기준)")

        portfolio_scenarios = {
            c: float(scenarios_df[c].sum())
            for c in ['-3std_bp', '-2std_bp', '-1std_bp', '+1std_bp', '+2std_bp', '+3std_bp']
        }
        mean_return_3m = float(scenarios_df['Portfolio_Mean_3M_bp'].sum())
        total_std_3m = float(np.sqrt((scenarios_df['Portfolio_Std_3M_bp'] ** 2).sum()))
        annual_return = float(scenarios_df['Portfolio_Annual_Return_bp'].sum())
        annual_std = float(np.sqrt((scenarios_df['Portfolio_Annual_Std_bp'] ** 2).sum()))

        # 손실 한도 점검
        total_expected_loss = float(scenarios_df['Expected_Loss_3M_bp'].sum())
        total_max_loss = float(scenarios_df['Max_Loss_bp'].sum())
        total_util = (total_expected_loss / total_max_loss * 100) if total_max_loss > 0 else np.nan

        # 3열 레이아웃
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="scenario-box">
                <h4>📉 하방 시나리오 (3M)</h4>
                <p><b>-3σ:</b> {portfolio_scenarios['-3std_bp']:.2f}bp</p>
                <p><b>-2σ:</b> {portfolio_scenarios['-2std_bp']:.2f}bp</p>
                <p><b>-1σ:</b> {portfolio_scenarios['-1std_bp']:.2f}bp</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="scenario-box" style="border-color:#28a745;">
                <h4>📊 중심 경향</h4>
                <p><b>평균 (3M):</b> {mean_return_3m:.2f}bp</p>
                <p><b>표준편차 (3M):</b> {total_std_3m:.2f}bp</p>
                <p><b>연율화 수익률:</b> {annual_return:.2f}bp</p>
                <p><b>연율화 Std:</b> {annual_std:.2f}bp</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="scenario-box">
                <h4>📈 상방 시나리오 (3M)</h4>
                <p><b>+1σ:</b> {portfolio_scenarios['+1std_bp']:.2f}bp</p>
                <p><b>+2σ:</b> {portfolio_scenarios['+2std_bp']:.2f}bp</p>
                <p><b>+3σ:</b> {portfolio_scenarios['+3std_bp']:.2f}bp</p>
            </div>
            """, unsafe_allow_html=True)

        # 손실 한도 요약
        st.markdown("---")
        st.subheader("⚠️ 손실 한도 점검")

        col_l1, col_l2, col_l3 = st.columns(3)

        with col_l1:
            st.metric("예상 최대 손실 (-3σ)", f"{total_expected_loss:.2f}bp")
        with col_l2:
            st.metric("허용 최대 손실", f"{total_max_loss:.2f}bp")
        with col_l3:
            util_color = "🟢" if total_util < 80 else "🟡" if total_util < 100 else "🔴"
            st.metric("Utilization", f"{util_color} {total_util:.1f}%")

        if total_util > 100:
            st.error("⚠️ **손실 한도 초과!** Asset View에서 Signal을 조정하거나 제약 방법을 변경하세요.")
        elif total_util > 90:
            st.warning("⚠️ 손실 한도에 근접했습니다. 주의가 필요합니다.")

        # ===== 막대 차트 =====
        st.markdown("---")
        st.subheader("📊 기대수익률 분포 (3M Rolling Return)")

        scenario_names = ['-3σ', '-2σ', '-1σ', 'Mean', '+1σ', '+2σ', '+3σ']
        scenario_values = [
            portfolio_scenarios['-3std_bp'],
            portfolio_scenarios['-2std_bp'],
            portfolio_scenarios['-1std_bp'],
            mean_return_3m,
            portfolio_scenarios['+1std_bp'],
            portfolio_scenarios['+2std_bp'],
            portfolio_scenarios['+3std_bp'],
        ]

        fig = go.Figure()

        colors = ['#d62728' if v < 0 else '#2ca02c' for v in scenario_values]

        fig.add_trace(go.Bar(
            x=scenario_names,
            y=scenario_values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in scenario_values],
            textposition='outside',
            hovertemplate="%{x}: %{y:.2f}bp<extra></extra>"
        ))

        fig.update_layout(
            title="포트폴리오 전체 기대수익률 시나리오 (3M, bp)",
            xaxis_title="시나리오",
            yaxis_title="기대수익률 (bp)",
            height=500,
            showlegend=False
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig = apply_chart_font_settings(fig)
        st.plotly_chart(fig, use_container_width=True)

        # ===== 상세 테이블 =====
        st.markdown("---")
        st.subheader("📋 Pair별 상세 시나리오")

        display_cols = [
            'Pair_ID', 'Pair', 'Signal', 'Position_bp', 'Total_Notional_bp', 'Legs',

            # 페어 통계 (%)
            'Pair_Mean_3M_%', 'Pair_Std_3M_%', 'Sharpe_3M',
            'Pair_Annual_Return_%', 'Pair_Annual_Std_%',

            # 시나리오 (%)
            '-3std_%', '-2std_%', '-1std_%', 'Mean_%',
            '+1std_%', '+2std_%', '+3std_%',

            # 포트폴리오 기여도 (bp)
            'Portfolio_Mean_3M_bp', 'Portfolio_Std_3M_bp',
            '-3std_bp', '-2std_bp', '-1std_bp',
            '+1std_bp', '+2std_bp', '+3std_bp',

            # 손실 점검
            'Expected_Loss_3M_bp', 'Max_Loss_bp', 'Utilization_%',
            'N_Observations'
        ]

        scenarios_display = scenarios_df[display_cols].copy()

        # 포맷팅
        format_dict = {}
        for c in scenarios_display.columns:
            if c.endswith('_bp'):
                format_dict[c] = '{:.3f}'
            elif c.endswith('_%'):
                format_dict[c] = '{:.2f}'
            elif c in ['Sharpe_3M']:
                format_dict[c] = '{:.3f}'
            elif c in ['Legs', 'N_Observations']:
                scenarios_display[c] = scenarios_display[c].astype(int)

        # Utilization 색상
        def highlight_util(row):
            util = row['Utilization_%']
            if util > 100:
                color = '#ffe6e6'  # 빨강
            elif util > 90:
                color = '#fff4e6'  # 노랑
            else:
                color = 'white'
            return ['background-color: {}'.format(color)] * len(row)

        styled_df = scenarios_display.style.apply(highlight_util, axis=1).format(format_dict)
        st.dataframe(styled_df, use_container_width=True)

        # ===== 3M 분포 히스토그램 =====
        st.markdown("---")
        st.subheader("📊 3M Rolling Return 분포")

        selected_pair_hist = st.selectbox(
            "분포를 확인할 Pair 선택",
            scenarios_df['Pair'].tolist(),
            key="hist_pair_select"
        )

        if selected_pair_hist:
            info = scenarios_df[scenarios_df['Pair'] == selected_pair_hist].iloc[0]
            la, sa = selected_pair_hist.split(' vs ')

            r3_hist = calculate_pair_3m_rolling_returns(
                returns_by_asset, la, sa, lookback_years
            )

            if not r3_hist.empty:
                fig_hist = go.Figure()

                # 히스토그램
                fig_hist.add_trace(go.Histogram(
                    x=r3_hist * 100,  # % 단위
                    nbinsx=50,
                    name='3M Returns',
                    opacity=0.7,
                    marker_color='#1f77b4'
                ))

                # 평균 및 표준편차 선
                mean_pct = float(info['Pair_Mean_3M_%'])
                std_pct = float(info['Pair_Std_3M_%'])

                fig_hist.add_vline(
                    x=mean_pct,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"평균: {mean_pct:.2f}%",
                    annotation_position="top"
                )

                for i in [-3, -2, -1, 1, 2, 3]:
                    x_val = mean_pct + i * std_pct
                    fig_hist.add_vline(
                        x=x_val,
                        line_dash="dot",
                        line_color="orange" if abs(i) <= 1 else "gray",
                        line_width=1,
                        annotation_text=f"{i:+d}σ"
                    )

                fig_hist.update_layout(
                    title=f"{selected_pair_hist} - 3개월 Rolling Return 분포 (EWM)",
                    xaxis_title="3M Return (%)",
                    yaxis_title="빈도",
                    height=400,
                    showlegend=False
                )
                fig_hist = apply_chart_font_settings(fig_hist)
                st.plotly_chart(fig_hist, use_container_width=True)

                # 통계 요약
                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                with col_h1:
                    st.metric("평균", f"{mean_pct:.2f}%")
                with col_h2:
                    st.metric("표준편차", f"{std_pct:.2f}%")
                with col_h3:
                    st.metric("-3σ", f"{mean_pct - 3 * std_pct:.2f}%")
                with col_h4:
                    st.metric("+3σ", f"{mean_pct + 3 * std_pct:.2f}%")

        # ===== 다운로드 =====
        st.markdown("---")
        st.subheader("📥 데이터 다운로드")

        csv_scenarios = scenarios_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Pair 시나리오 분석 다운로드 (CSV)",
            data=csv_scenarios,
            file_name=f"pair_scenarios_3m_rolling_{constraint_method}_{lookback_years}y.csv",
            mime="text/csv",
            key="download_scenarios_3m"
        )
    # =========================================================================
    # Tab 10: 리스크 제약 감도분석
    # =========================================================================
    with tabs[10]:
        st.header("🛑 리스크 제약 감도분석")
        st.markdown(f"""
        현재 선택된 제약 방법: **{CONSTRAINT_DISPLAY_MAP.get(constraint_method, constraint_method)}**


        이 탭에서는 각 Pair의 제약 값을 기반으로 최대 허용 손실을 설정하고, 
        이에 따라 포지션 사이즈를 조정합니다.
        """)

        if returns_by_asset.empty:
            st.info("시장 수익률 데이터가 필요합니다.")
        else:
            # as-of 가중치
            if not w_opt_daily.empty and not w_bmk_daily.empty:
                asof = min(w_opt_daily.index.max(), w_bmk_daily.index.max())
                Wopt_last = w_opt_daily.loc[asof]
                Wbmk_last = w_bmk_daily.loc[asof]
            elif not weight_history.empty:
                asof = weight_history.index.max()
                row = weight_history.loc[asof]
                assets_wh = sorted({c.replace('_Optimal', '').replace('_Benchmark', '').replace('_Active', '') for c in
                                    weight_history.columns})
                Wopt_last = pd.Series({a: row.get(f"{a}_Optimal", 0.0) for a in assets_wh})
                Wbmk_last = pd.Series({a: row.get(f"{a}_Benchmark", 0.0) for a in assets_wh})
            else:
                asof = None
                Wopt_last = pd.Series(dtype=float)
                Wbmk_last = pd.Series(dtype=float)

            if Wopt_last.empty or Wbmk_last.empty:
                st.warning("현재 포지션이 없어 감도 분석을 수행할 수 없습니다.")
            else:
                Wact_last = (Wopt_last - Wbmk_last).fillna(0.0)
                assets_list = [a for a in Wact_last.index if a in returns_by_asset.columns]
                Wact_last = Wact_last.reindex(assets_list).fillna(0.0)

                tl = timeline_history.copy() if not timeline_history.empty else views_source.copy()
                if tl.empty or 'Long_Asset' not in tl.columns or 'Short_Asset' not in tl.columns:
                    st.warning("페어 타임라인/뷰 데이터가 없어 분석을 건너뜁니다.")
                else:
                    for c in ["Start_Date", "End_Date"]:
                        if c in tl.columns:
                            tl[c] = pd.to_datetime(tl[c], errors="coerce")

                    active_rows = tl.copy()
                    now_ref = asof or pd.Timestamp.today()
                    if 'Start_Date' in active_rows.columns:
                        active_rows = active_rows[active_rows['Start_Date'].fillna(pd.Timestamp.min) <= now_ref]
                    if 'End_Date' in active_rows.columns:
                        active_rows = active_rows[active_rows['End_Date'].fillna(pd.Timestamp.max) >= now_ref]

                    if active_rows.empty:
                        st.warning("현재 활성 Pair가 없습니다.")
                    else:
                        pairs = active_rows[['Long_Asset', 'Short_Asset']].dropna().astype(str).values.tolist()
                        signals = pd.to_numeric(active_rows.get('Signal', 0.0), errors='coerce').fillna(0.0).values
                        pair_ids = active_rows.get('Pair_ID', range(len(pairs))).values

                        B = build_incidence_matrix(assets_list, pairs)
                        if B.size == 0:
                            st.warning("Incidence 행렬 생성 실패(자산-페어 매핑 확인).")
                        else:
                            # 기본 설정
                            st.subheader("🎛️ 기본 설정")
                            col1, col2 = st.columns(2)

                            with col1:
                                lookback_years = st.slider(
                                    "제약 계산 룩백 기간 (년)",
                                    1, 5, 3,
                                    key="constraint_lookback",
                                    help="Historical 제약 값 계산에 사용할 과거 기간"
                                )

                            # 제약 값 계산
                            risk_calc = RiskConstraintCalculator(returns_by_asset, lookback_years=lookback_years)

                            # 현재 페어 사이즈 추정
                            x_cur = reconstruct_pair_sizes(Wact_last.values, B, signals)

                            # 각 Pair의 제약 값 계산
                            constraint_values, cap_arr_default = risk_calc.calculate_position_caps(
                                pairs, signals, constraint_method
                            )

                            # 현재 적용된 최대 손실 계산
                            current_max_loss = calculate_current_max_loss_bp(x_cur, constraint_values)

                            with col2:
                                # 손실 한도 범위: 0.01bp ~ 0.25bp
                                default_max_loss_bp = st.slider(
                                    "기본 최대 허용 손실 (bp)",
                                    min_value=0.01,
                                    max_value=0.25,
                                    value=min(max(0.1, current_max_loss), 0.25),
                                    step=0.01,
                                    format="%.3f",
                                    key="default_constraint_loss",
                                    help=f"현재 포트폴리오 적용 손실: {current_max_loss:.3f}bp"
                                )

                            st.info(
                                f"💡 현재 포트폴리오에 적용된 평균 손실 한도: **{current_max_loss:.3f}bp** | 설정값: **{default_max_loss_bp:.3f}bp**"
                            )

                            # Pair별 개별 제약 설정
                            st.markdown("---")
                            st.subheader("⚙️ Pair별 최대 손실 허용치 설정 (0.01bp~0.25bp)")

                            # Session state 초기화
                            if 'pair_max_loss_constraint' not in st.session_state:
                                st.session_state.pair_max_loss_constraint = {}

                            max_loss_per_pair = np.full(len(pairs), default_max_loss_bp)

                            col_count = 2
                            cols = st.columns(col_count)

                            for idx in range(len(pairs)):
                                pair_id = pair_ids[idx]
                                la, sa = pairs[idx]

                                col_idx = idx % col_count
                                with cols[col_idx]:
                                    with st.expander(f"**Pair {pair_id}: {la} vs {sa}**", expanded=False):
                                        constraint_pct = constraint_values[idx] * 100
                                        cur_pos = x_cur[idx] * 10000

                                        constraint_label = {
                                            "3Y_MDD": "Historical MDD",
                                            "-3STD": "-3 표준편차",
                                            "-2STD": "-2 표준편차",
                                            "-1STD": "-1 표준편차"
                                        }[constraint_method]

                                        st.metric(constraint_label, f"{constraint_pct:.2f}%")
                                        st.metric("현재 포지션", f"{cur_pos:.3f}bp")

                                        # 현재 pair의 실제 손실 계산
                                        pair_current_loss = abs(x_cur[idx]) * abs(constraint_values[idx]) * 10000
                                        st.caption(f"현재 적용 손실: {pair_current_loss:.3f}bp")

                                        loss_bp = st.slider(
                                            "최대 허용 손실 (bp)",
                                            min_value=0.01,
                                            max_value=0.25,
                                            value=float(st.session_state.pair_max_loss_constraint.get(idx,
                                                                                                      default_max_loss_bp)),
                                            step=0.01,
                                            format="%.3f",
                                            key=f"pair_constraint_loss_{idx}",
                                            help=f"이 Pair의 최대 손실 허용치 (0.01bp~0.25bp)"
                                        )
                                        st.session_state.pair_max_loss_constraint[idx] = loss_bp
                                        max_loss_per_pair[idx] = loss_bp

                            # 각 Pair별 최대 포지션 계산
                            _views_for_caps = st.session_state.get("adjusted_views", views_source).reset_index(
                                drop=True)
                            _loss_caps_bp = _compute_loss_caps_bp_from_views(_views_for_caps)

                            # 기존 cap_arr 생성부 교체
                            cap_arr = np.array([
                                float(max_loss_per_pair[i] / 10000.0) / (abs(constraint_values[i]) * 2.0)
                                if abs(constraint_values[i]) > 1e-8
                                else 1.0
                                for i in range(len(constraint_values))
                            ])

                            # 캡 적용한 대안 포지션
                            x_cap = np.clip(x_cur, -cap_arr, cap_arr)
                            Wact_alt = pd.Series(B @ x_cap, index=assets_list)

                            # 포지션 변경 감지
                            position_changes = np.abs((x_cap - x_cur) * 10000)
                            has_changes = position_changes > 1e-9
                            n_changed = has_changes.sum()

                            # 진단 정보
                            st.markdown("---")
                            st.subheader("🔬 포지션 변경 진단")

                            col_diag1, col_diag2, col_diag3 = st.columns(3)
                            with col_diag1:
                                st.metric("총 Pair 수", len(pairs))
                            with col_diag2:
                                st.metric(
                                    "포지션 변경 Pair",
                                    f"{n_changed}개",
                                    help="제약 변경으로 포지션이 변경된 Pair"
                                )
                            with col_diag3:
                                total_change = position_changes.sum()
                                st.metric("총 변경량", f"{total_change:.3f}bp")

                            if n_changed > 0:
                                st.success(f"✅ {n_changed}개 Pair에서 포지션 변경이 발생했습니다.")
                            else:
                                st.info("ℹ️ 모든 Pair의 현재 포지션이 허용 캡 이내입니다. 더 작은 손실 한도를 설정하면 제약이 적용됩니다.")

                            # TE 및 변동성 계산
                            if cov_matrix.empty:
                                te_cur = te_alt = vol_cur = vol_alt = np.nan
                            else:
                                cov_use_assets = [a for a in assets_list if a in cov_matrix.index]
                                te_cur = compute_te_from_active(cov_matrix, cov_use_assets,
                                                                Wact_last.reindex(cov_use_assets))
                                te_alt = compute_te_from_active(cov_matrix, cov_use_assets,
                                                                Wact_alt.reindex(cov_use_assets))

                                cov_np = cov_matrix.reindex(index=cov_use_assets, columns=cov_use_assets).fillna(
                                    0.0).values
                                w_cur = (Wbmk_last.reindex(cov_use_assets).fillna(0.0) + Wact_last.reindex(
                                    cov_use_assets).fillna(0.0)).values
                                w_alt = (Wbmk_last.reindex(cov_use_assets).fillna(0.0) + Wact_alt.reindex(
                                    cov_use_assets).fillna(0.0)).values
                                vol_cur = float(np.sqrt(w_cur @ cov_np @ w_cur))
                                vol_alt = float(np.sqrt(w_alt @ cov_np @ w_alt))

                            # 메트릭 표시
                            st.markdown("---")
                            st.subheader("📊 리스크 지표 변화")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("현재 TE(연율)", f"{(te_cur * 10000 if pd.notna(te_cur) else 0):.3f}bp")
                            with col2:
                                te_delta = ((te_alt - te_cur) * 10000 if (pd.notna(te_alt) and pd.notna(te_cur)) else 0)
                                st.metric("대안 TE(연율)", f"{(te_alt * 10000 if pd.notna(te_alt) else 0):.3f}bp",
                                          delta=f"{te_delta:+.3f}bp")
                            with col3:
                                vol_delta = (
                                    (vol_alt - vol_cur) * 10000 if (pd.notna(vol_alt) and pd.notna(vol_cur)) else 0)
                                st.metric("포트폴리오 변동성 변화", f"{(vol_alt * 10000 if pd.notna(vol_alt) else 0):.3f}bp",
                                          delta=f"{vol_delta:+.3f}bp")

                            st.markdown("---")

                            # Active Weight 변화 시각화
                            st.subheader("📊 자산별 Active Weight 변화")
                            diff = (Wact_alt - Wact_last).sort_values(key=lambda s: s.abs(), ascending=False)

                            # 변경이 있는 자산만 필터링
                            diff_changed = diff[np.abs(diff * 10000) > 1e-6]

                            if len(diff_changed) == 0:
                                st.info("자산별 Active Weight 변경이 없습니다.")
                            else:
                                # Slider 오류 완전 해결
                                max_assets_to_display = len(diff_changed)

                                if max_assets_to_display <= 1:
                                    n_display = max_assets_to_display
                                else:
                                    min_display_assets = min(5, max_assets_to_display)
                                    max_display_assets = min(30, max_assets_to_display)

                                    if min_display_assets >= max_display_assets:
                                        n_display = max_display_assets
                                    else:
                                        default_display_assets = min(15, max_assets_to_display)
                                        n_display = st.slider(
                                            "표시 자산 수",
                                            min_value=min_display_assets,
                                            max_value=max_display_assets,
                                            value=default_display_assets,
                                            key="n_display_constraint"
                                        )

                                show_assets = diff_changed.head(n_display).index.tolist()

                                fig = make_subplots(
                                    rows=2, cols=1,
                                    subplot_titles=("Active Weight 비교 (bp)", "Active Weight 변화량 (bp)"),
                                    vertical_spacing=0.15,
                                    row_heights=[0.6, 0.4]
                                )

                                fig.add_trace(
                                    go.Bar(
                                        name="현재",
                                        x=show_assets,
                                        y=(Wact_last.reindex(show_assets) * 10000).values,
                                        marker_color='lightblue',
                                        hovertemplate="%{y:.3f}bp<extra></extra>"
                                    ),
                                    row=1, col=1
                                )
                                fig.add_trace(
                                    go.Bar(
                                        name="대안(제약 Cap)",
                                        x=show_assets,
                                        y=(Wact_alt.reindex(show_assets) * 10000).values,
                                        marker_color='lightcoral',
                                        hovertemplate="%{y:.3f}bp<extra></extra>"
                                    ),
                                    row=1, col=1
                                )

                                changes = (diff.reindex(show_assets) * 10000).values
                                colors = ['green' if c > 0 else 'red' for c in changes]
                                fig.add_trace(
                                    go.Bar(
                                        name="변화량",
                                        x=show_assets,
                                        y=changes,
                                        marker_color=colors,
                                        showlegend=False,
                                        hovertemplate="%{y:.3f}bp<extra></extra>"
                                    ),
                                    row=2, col=1
                                )

                                fig.update_xaxes(title_text="", row=1, col=1)
                                fig.update_xaxes(title_text="자산", row=2, col=1)
                                fig.update_yaxes(title_text="Active Weight (bp)", row=1, col=1, tickformat=".3f")
                                fig.update_yaxes(title_text="변화량 (bp)", row=2, col=1, tickformat=".3f")
                                fig.update_layout(barmode='group', height=700, hovermode='x unified')

                                # 폰트 크기 적용
                                fig = apply_chart_font_settings(fig)

                                st.plotly_chart(fig, use_container_width=True)

                            st.markdown("---")

                            # Pair별 제약 상세 테이블
                            st.subheader("🎯 Pair별 리스크 제약 상세")

                            bind = (np.abs(x_cur) > np.abs(x_cap) + 1e-9)

                            constraint_col_name = {
                                "3Y_MDD": "MDD_%",
                                "-3STD": "-3STD_%",
                                "-2STD": "-2STD_%",
                                "-1STD": "-1STD_%"
                            }[constraint_method]

                            bind_df = pd.DataFrame({
                                "Pair_ID": pair_ids,
                                "Pair": [f"{p[0]} vs {p[1]}" for p in pairs],
                                "Long_Asset": [p[0] for p in pairs],
                                "Short_Asset": [p[1] for p in pairs],
                                "Signal": signals,
                                constraint_col_name: (constraint_values * 100).round(2),
                                "Max_Loss_bp": max_loss_per_pair.round(3),
                                "Per_Leg_Cap_bp": (cap_arr * 10000).round(3),
                                "Current_Position_bp": (x_cur * 10000).round(3),
                                "Capped_Position_bp": (x_cap * 10000).round(3),
                                "Position_Change_bp": ((x_cap - x_cur) * 10000).round(3),
                                "Actual_Loss_bp": (np.abs(x_cap) * np.abs(constraint_values) * 10000).round(3),
                                "Binding": bind,
                                "Status": ["⚠️ 제약 적용" if b else "✅ OK" for b in bind]
                            })

                            def highlight_binding(row):
                                if row['Binding']:
                                    return ['background-color: #ffe6e6'] * len(row)
                                return [''] * len(row)

                            styled_df = bind_df.style.apply(highlight_binding, axis=1).format({
                                constraint_col_name: '{:.2f}',
                                'Max_Loss_bp': '{:.3f}',
                                'Per_Leg_Cap_bp': '{:.3f}',
                                'Current_Position_bp': '{:.3f}',
                                'Capped_Position_bp': '{:.3f}',
                                'Position_Change_bp': '{:+.3f}',
                                'Actual_Loss_bp': '{:.3f}'
                            })

                            st.dataframe(styled_df, use_container_width=True)

                            # 요약 통계
                            st.markdown("---")
                            st.subheader("📊 요약 통계")

                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

                            with summary_col1:
                                st.metric("총 Pair 수", len(pairs))
                            with summary_col2:
                                n_binding = bind.sum()
                                st.metric(
                                    "제약 적용",
                                    f"{n_binding}개",
                                    delta=f"{n_binding / len(pairs) * 100:.1f}%" if len(pairs) > 0 else "0%"
                                )
                            with summary_col3:
                                avg_loss = bind_df['Actual_Loss_bp'].mean()
                                max_loss_avg = max_loss_per_pair.mean()
                                st.metric(
                                    "평균 실제 손실",
                                    f"{avg_loss:.3f}bp",
                                    delta=f"허용치: {max_loss_avg:.3f}bp"
                                )
                            with summary_col4:
                                avg_constraint = np.abs(constraint_values).mean() * 100
                                st.metric(f"평균 {constraint_col_name.replace('_%', '')}", f"{avg_constraint:.2f}%")

                            # Position Cap 비교 차트
                            st.markdown("---")
                            st.subheader("🎯 포지션 크기 비교: Current vs Capped")

                            fig_pos = go.Figure()

                            fig_pos.add_trace(go.Bar(
                                name='현재 포지션',
                                x=bind_df['Pair'],
                                y=bind_df['Current_Position_bp'],
                                marker_color='lightblue',
                                hovertemplate='현재: %{y:.3f}bp<extra></extra>'
                            ))

                            fig_pos.add_trace(go.Bar(
                                name='Capped 포지션',
                                x=bind_df['Pair'],
                                y=bind_df['Capped_Position_bp'],
                                marker_color='orange',
                                hovertemplate='Capped: %{y:.3f}bp<extra></extra>'
                            ))

                            fig_pos.update_layout(
                                title='포지션 크기 비교 (Per Leg, bp)',
                                xaxis_title='Pair',
                                yaxis_title='포지션 크기 (bp)',
                                yaxis_tickformat=".3f",
                                height=450,
                                barmode='group',
                                hovermode='x unified'
                            )

                            # 폰트 크기 적용
                            fig_pos = apply_chart_font_settings(fig_pos)

                            st.plotly_chart(fig_pos, use_container_width=True)

                            # 최대 손실 vs 실제 손실 비교
                            st.subheader("💰 최대 허용 손실 vs 실제 손실")

                            fig_loss = go.Figure()

                            fig_loss.add_trace(go.Bar(
                                name='최대 허용 손실',
                                x=bind_df['Pair'],
                                y=bind_df['Max_Loss_bp'],
                                marker_color='lightgreen',
                                hovertemplate='허용: %{y:.3f}bp<extra></extra>'
                            ))

                            fig_loss.add_trace(go.Bar(
                                name='실제 손실 (Capped)',
                                x=bind_df['Pair'],
                                y=bind_df['Actual_Loss_bp'],
                                marker_color='lightcoral',
                                hovertemplate='실제: %{y:.3f}bp<extra></extra>'
                            ))

                            fig_loss.update_layout(
                                title='Pair별 최대 허용 손실 vs 실제 손실 (bp)',
                                xaxis_title='Pair',
                                yaxis_title='손실 (bp)',
                                yaxis_tickformat=".3f",
                                height=400,
                                barmode='group',
                                hovermode='x unified'
                            )

                            # 폰트 크기 적용
                            fig_loss = apply_chart_font_settings(fig_loss)

                            st.plotly_chart(fig_loss, use_container_width=True)

                            # 제약 방법 비교
                            st.markdown("---")
                            st.subheader("🔄 다른 제약 방법과 비교")

                            comparison_methods = ["3Y_MDD", "-3STD", "-2STD", "-1STD"]
                            comparison_data = []

                            with st.spinner("다른 제약 방법 계산 중..."):
                                for method in comparison_methods:
                                    if method == constraint_method:
                                        # 현재 방법은 이미 계산됨
                                        comparison_data.append({
                                            'Method': method,
                                            'Avg_Position_bp': (x_cap * 10000).mean(),
                                            'Total_TE_bp': te_alt * 10000 if pd.notna(te_alt) else 0,
                                            'Avg_Loss_bp': bind_df['Actual_Loss_bp'].mean()
                                        })
                                    else:
                                        # 다른 방법 계산
                                        constraint_vals_comp, cap_arr_comp = risk_calc.calculate_position_caps(
                                            pairs, signals, method
                                        )
                                        x_comp = np.sign(signals) * cap_arr_comp
                                        Wact_comp = pd.Series(B @ x_comp, index=assets_list)

                                        if not cov_matrix.empty:
                                            te_comp = compute_te_from_active(cov_matrix, cov_use_assets,
                                                                             Wact_comp.reindex(cov_use_assets))
                                        else:
                                            te_comp = np.nan

                                        avg_loss_comp = (np.abs(x_comp) * np.abs(constraint_vals_comp) * 10000).mean()

                                        comparison_data.append({
                                            'Method': method,
                                            'Avg_Position_bp': (x_comp * 10000).mean(),
                                            'Total_TE_bp': te_comp * 10000 if pd.notna(te_comp) else 0,
                                            'Avg_Loss_bp': avg_loss_comp
                                        })

                            comparison_df = pd.DataFrame(comparison_data)

                            # 비교 차트
                            fig_comp = make_subplots(
                                rows=1, cols=3,
                                subplot_titles=("평균 포지션", "예상 TE", "평균 손실"),
                                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
                            )

                            colors_comp = ['green' if m == constraint_method else 'lightgray' for m in
                                           comparison_df['Method']]

                            fig_comp.add_trace(
                                go.Bar(
                                    x=comparison_df['Method'],
                                    y=comparison_df['Avg_Position_bp'],
                                    marker_color=colors_comp,
                                    showlegend=False,
                                    hovertemplate="%{y:.3f}bp<extra></extra>"
                                ),
                                row=1, col=1
                            )

                            fig_comp.add_trace(
                                go.Bar(
                                    x=comparison_df['Method'],
                                    y=comparison_df['Total_TE_bp'],
                                    marker_color=colors_comp,
                                    showlegend=False,
                                    hovertemplate="%{y:.3f}bp<extra></extra>"
                                ),
                                row=1, col=2
                            )

                            fig_comp.add_trace(
                                go.Bar(
                                    x=comparison_df['Method'],
                                    y=comparison_df['Avg_Loss_bp'],
                                    marker_color=colors_comp,
                                    showlegend=False,
                                    hovertemplate="%{y:.3f}bp<extra></extra>"
                                ),
                                row=1, col=3
                            )

                            fig_comp.update_yaxes(title_text="bp", tickformat=".3f", row=1, col=1)
                            fig_comp.update_yaxes(title_text="bp", tickformat=".3f", row=1, col=2)
                            fig_comp.update_yaxes(title_text="bp", tickformat=".3f", row=1, col=3)

                            fig_comp.update_layout(height=400, title_text="리스크 제약 방법 비교")

                            # 폰트 크기 적용
                            fig_comp = apply_chart_font_settings(fig_comp)

                            st.plotly_chart(fig_comp, use_container_width=True)

                            # 비교 테이블
                            comparison_display = comparison_df.copy()
                            comparison_display['Avg_Position_bp'] = comparison_display['Avg_Position_bp'].apply(
                                lambda x: f"{x:.3f}")
                            comparison_display['Total_TE_bp'] = comparison_display['Total_TE_bp'].apply(
                                lambda x: f"{x:.3f}")
                            comparison_display['Avg_Loss_bp'] = comparison_display['Avg_Loss_bp'].apply(
                                lambda x: f"{x:.3f}")

                            # 현재 방법 하이라이트
                            def highlight_current_method(row):
                                if row['Method'] == constraint_method:
                                    return ['background-color: #e7f3ff'] * len(row)
                                return [''] * len(row)

                            styled_comp_df = comparison_display.style.apply(highlight_current_method, axis=1)
                            st.dataframe(styled_comp_df, use_container_width=True)

                            # CSV 다운로드
                            st.markdown("---")
                            st.subheader("📥 데이터 다운로드")

                            col_d1, col_d2, col_d3 = st.columns(3)

                            with col_d1:
                                csv_pair = bind_df.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label="📥 Pair 제약 상세 다운로드",
                                    data=csv_pair,
                                    file_name=f"constraint_analysis_{constraint_method}_{default_max_loss_bp:.3f}bp.csv",
                                    mime="text/csv",
                                    key="download_pair_constraints_detail"
                                )

                            with col_d2:
                                weight_change_df = pd.DataFrame({
                                    'Asset': assets_list,
                                    'Current_Active_bp': (Wact_last.reindex(assets_list) * 10000).values,
                                    'Alternative_Active_bp': (Wact_alt.reindex(assets_list) * 10000).values,
                                    'Change_bp': ((Wact_alt.reindex(assets_list) - Wact_last.reindex(
                                        assets_list)) * 10000).values
                                }).sort_values('Change_bp', key=lambda x: x.abs(), ascending=False)

                                # 소수점 3자리 포맷
                                for col in ['Current_Active_bp', 'Alternative_Active_bp', 'Change_bp']:
                                    weight_change_df[col] = weight_change_df[col].apply(lambda x: f"{x:.3f}")

                                csv_weight = weight_change_df.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label="📥 Active Weight 변화 다운로드",
                                    data=csv_weight,
                                    file_name=f"active_weight_changes_{constraint_method}_{default_max_loss_bp:.3f}bp.csv",
                                    mime="text/csv",
                                    key="download_weight_changes_constraint"
                                )

                            with col_d3:
                                csv_comp = comparison_df.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label="📥 제약 방법 비교 다운로드",
                                    data=csv_comp,
                                    file_name=f"constraint_method_comparison_{lookback_years}y.csv",
                                    mime="text/csv",
                                    key="download_comparison"
                                )

    # =========================================================================
    # Tab 11: 실제 포트폴리오 성과 (Enhanced with Risk Simulation)
    # =========================================================================
    with tabs[11]:
        st.header("📊 실제 포트폴리오 성과 분석 & 리스크 시뮬레이션")
        st.markdown("""
        업로드된 실제 포트폴리오 데이터를 기반으로 성과를 분석하고, 포지션 크기 변경에 따른 리스크 변화를 시뮬레이션합니다.

        **새로운 기능:**
        1. 📊 **포지션 타입 및 크기 조정 → 리스크 변화**: 각 페어의 포지션 타입과 크기를 변경했을 때 포트폴리오 전체 리스크가 얼마나 변하는지 확인
        2. 🎯 **목표 리스크 → 포지션 역산**: 원하는 리스크 변화량을 입력하면 해당 페어의 포지션 크기를 자동 계산
        3. 📈 **손익 분포 그래프**: -3σ ~ +3σ 시나리오별 P&L 분포 시각화 (현재 + 조정 포지션 비교)

        **포지션 타입 (각 자산별 Long/Short 선택 가능):**
        - ⚖️ **Pair (L/S)**: Long 자산 매수 + Short 자산 매도 동시 포지션
        - 📈 **Long: [자산명]**: 해당 자산만 단독 매수 (Long/Short 양쪽 자산 모두 선택 가능)
        - 📉 **Short: [자산명]**: 해당 자산만 단독 매도 (Long/Short 양쪽 자산 모두 선택 가능)
        """)

        # ===== 파일 로드 =====
        try:
            import os

            # 기본 디렉토리
            base_dir = data_dir  # Streamlit Cloud compatible

            # 수익률 데이터
            actual_returns_df = pd.read_csv(
                os.path.join(base_dir, 'actual_portfolio_returns.csv'),
                parse_dates=['Date'],
                index_col='Date'
            )

            # 포지션 데이터
            actual_positions_df = pd.read_csv(
                os.path.join(base_dir, 'actual_portfolio_positions.csv'),
                parse_dates=['Date']
            )

            # 요약 데이터
            actual_summary_df = pd.read_csv(
                os.path.join(base_dir, 'actual_portfolio_summary.csv')
            )

            # ===== Inception 날짜 결정 (포지션 진입일 기준) =====
            # ENTRY 이벤트가 있는 날짜들 찾기
            entry_dates = actual_positions_df[
                actual_positions_df['Event'].str.contains('ENTRY', case=False, na=False)
            ]['Date']

            if not entry_dates.empty:
                inception_date = entry_dates.min()
                st.info(f"📅 **Inception Date**: {inception_date.strftime('%Y-%m-%d')} (첫 포지션 진입일)")
            else:
                # ENTRY 이벤트가 없으면 포지션 데이터의 가장 빠른 날짜 사용
                inception_date = actual_positions_df['Date'].min()
                st.info(f"📅 **Inception Date**: {inception_date.strftime('%Y-%m-%d')} (포지션 데이터 시작일)")

            # Inception 날짜 이후 데이터만 필터링
            actual_returns_df = actual_returns_df[actual_returns_df.index >= inception_date]

            st.success(
                f"✅ 데이터 로드 완료: {len(actual_returns_df)}일, "
                f"{actual_returns_df.index.min().date()} ~ {actual_returns_df.index.max().date()}"
            )

        except FileNotFoundError as e:
            st.error(f"❌ 파일을 찾을 수 없습니다: {e}")
            st.info(
                "📁 다음 파일들을 확인하세요:\n- actual_portfolio_returns.csv\n- actual_portfolio_positions.csv\n- actual_portfolio_summary.csv")
            st.stop()
        except Exception as e:
            st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
            st.stop()

        # ===== 기간 선택 UI =====
        st.markdown("---")
        st.subheader("📅 분석 기간 설정")

        col_period1, col_period2, col_period3 = st.columns([1, 1, 1])

        with col_period1:
            # 빠른 선택
            quick_period = st.selectbox(
                "빠른 기간 선택",
                ["전체 기간", "최근 1개월", "최근 3개월", "최근 6개월", "최근 1년", "사용자 지정"],
                key="quick_period_selector"
            )

        # 기간 계산
        data_end_date = actual_returns_df.index.max()
        data_start_date = actual_returns_df.index.min()

        if quick_period == "전체 기간":
            selected_start = data_start_date
            selected_end = data_end_date
            manual_selection = False
        elif quick_period == "최근 1개월":
            selected_start = max(data_start_date, data_end_date - pd.Timedelta(days=30))
            selected_end = data_end_date
            manual_selection = False
        elif quick_period == "최근 3개월":
            selected_start = max(data_start_date, data_end_date - pd.Timedelta(days=90))
            selected_end = data_end_date
            manual_selection = False
        elif quick_period == "최근 6개월":
            selected_start = max(data_start_date, data_end_date - pd.Timedelta(days=180))
            selected_end = data_end_date
            manual_selection = False
        elif quick_period == "최근 1년":
            selected_start = max(data_start_date, data_end_date - pd.Timedelta(days=365))
            selected_end = data_end_date
            manual_selection = False
        else:  # 사용자 지정
            manual_selection = True
            with col_period2:
                selected_start = st.date_input(
                    "시작일",
                    value=data_start_date.date(),
                    min_value=data_start_date.date(),
                    max_value=data_end_date.date(),
                    key="custom_start_date"
                )
                selected_start = pd.Timestamp(selected_start)

            with col_period3:
                selected_end = st.date_input(
                    "종료일",
                    value=data_end_date.date(),
                    min_value=data_start_date.date(),
                    max_value=data_end_date.date(),
                    key="custom_end_date"
                )
                selected_end = pd.Timestamp(selected_end)

        # 선택된 기간으로 필터링
        filtered_returns = actual_returns_df[
            (actual_returns_df.index >= selected_start) &
            (actual_returns_df.index <= selected_end)
            ].copy()

        if filtered_returns.empty:
            st.warning("⚠️ 선택한 기간에 데이터가 없습니다.")
            st.stop()

        # 기간 정보 표시
        n_trading_days = len(filtered_returns)
        st.caption(
            f"📊 선택 기간: **{selected_start.strftime('%Y-%m-%d')}** ~ **{selected_end.strftime('%Y-%m-%d')}** "
            f"({n_trading_days}일)"
        )

        # ===== 기간 정의 (선택된 데이터 기준) =====
        periods_config = {
            '1D': 1,
            '1W': 5,
            '2W': 10,
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '12M': 252,
            'Inception': len(filtered_returns)
        }

        # ===== 기간별 성과 계산 함수 =====
        def calculate_period_metrics(returns_series, period_days):
            """기간별 성과 지표 계산"""
            if len(returns_series) < period_days:
                period_returns = returns_series
            else:
                period_returns = returns_series.tail(period_days)

            if len(period_returns) == 0:
                return {
                    'cumulative_return': 0.0,
                    'annualized_return': 0.0,
                    'annualized_volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'n_days': 0
                }

            # 누적 수익률
            cum_ret = (1 + period_returns).prod() - 1

            # 거래일 수
            n_days = len(period_returns)

            # 연율화 수익률
            if n_days > 0:
                ann_ret = (1 + cum_ret) ** (252 / n_days) - 1
            else:
                ann_ret = 0.0

            # 연율화 변동성
            ann_vol = period_returns.std() * np.sqrt(252)

            # Sharpe Ratio
            if ann_vol > 0:
                sharpe = ann_ret / ann_vol
            else:
                sharpe = 0.0

            # MDD 계산
            cum_series = (1 + period_returns).cumprod()
            running_max = cum_series.expanding().max()
            drawdown = (cum_series - running_max) / running_max
            mdd = drawdown.min()

            return {
                'cumulative_return': cum_ret,
                'annualized_return': ann_ret,
                'annualized_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': mdd,
                'n_days': n_days
            }

        # ===== 누적 수익률 그래프 =====
        st.markdown("---")
        st.subheader("📈 누적 수익률 추이")

        cum_returns = (1 + filtered_returns['Actual_Portfolio_Return']).cumprod() - 1

        fig_cum = go.Figure()

        fig_cum.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values * 10000,  # bp 단위
            mode='lines',
            name='누적 수익률',
            line=dict(color='#1f77b4', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)',
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.3f}bp<extra></extra>'
        ))

        fig_cum.update_layout(
            title=f"실제 포트폴리오 누적 수익률 (bp) - {selected_start.strftime('%Y-%m-%d')} ~ {selected_end.strftime('%Y-%m-%d')}",
            xaxis_title="날짜",
            yaxis_title="누적 수익률 (bp)",
            height=500,
            hovermode='x unified'
        )

        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_cum = apply_chart_font_settings(fig_cum)

        st.plotly_chart(fig_cum, use_container_width=True)

        # ===== 기간별 성과 테이블 =====
        st.markdown("---")
        st.subheader("📊 기간별 성과 지표")

        # 성과 계산
        performance_data = []

        for period_name, period_days in periods_config.items():
            metrics = calculate_period_metrics(
                filtered_returns['Actual_Portfolio_Return'],
                period_days
            )

            performance_data.append({
                '기간': period_name,
                '거래일': f"{metrics['n_days']}일",
                '누적 수익률 (bp)': f"{metrics['cumulative_return'] * 10000:.3f}",
                '연율화 수익률 (bp)': f"{metrics['annualized_return'] * 10000:.3f}",
                '연율화 변동성 (bp)': f"{metrics['annualized_volatility'] * 10000:.3f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'MDD (%)': f"{metrics['max_drawdown'] * 100:.2f}"
            })

        performance_df = pd.DataFrame(performance_data)

        # 스타일링
        def highlight_inception(row):
            if row['기간'] == 'Inception':
                return ['background-color: #e7f3ff; font-weight: bold'] * len(row)
            return [''] * len(row)

        styled_performance = performance_df.style.apply(highlight_inception, axis=1)

        st.dataframe(styled_performance, use_container_width=True)

        st.caption(f"💡 Inception = 선택된 전체 기간 ({n_trading_days}일)")

        # ===== 요약 통계 (선택 기간 기준) =====
        st.markdown("---")
        st.subheader("📋 선택 기간 요약 통계")

        # 선택 기간의 통계 계산
        selected_metrics = calculate_period_metrics(
            filtered_returns['Actual_Portfolio_Return'],
            len(filtered_returns)
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("누적 수익률", f"{selected_metrics['cumulative_return'] * 10000:.3f}bp")

        with col2:
            st.metric("연율화 수익률", f"{selected_metrics['annualized_return'] * 10000:.3f}bp")

        with col3:
            st.metric("연율화 변동성", f"{selected_metrics['annualized_volatility'] * 10000:.3f}bp")

        with col4:
            st.metric("Sharpe Ratio", f"{selected_metrics['sharpe_ratio']:.3f}")

        with col5:
            st.metric("Max Drawdown", f"{selected_metrics['max_drawdown'] * 100:.2f}%")

        st.caption(
            f"📅 분석 기간: {selected_start.strftime('%Y-%m-%d')} ~ {selected_end.strftime('%Y-%m-%d')} ({selected_metrics['n_days']}일)")

        # ===== 전체 기간 요약 (참고용) =====
        with st.expander("📊 전체 기간 요약 통계 (참고)", expanded=False):
            summary_dict = dict(zip(actual_summary_df['Metric'], actual_summary_df['Value']))

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                cum_ret_summary = float(summary_dict.get('Cumulative_Return', 0))
                st.metric("누적 수익률", f"{cum_ret_summary * 10000:.3f}bp")

            with col2:
                ann_ret_summary = float(summary_dict.get('Annualized_Return', 0))
                st.metric("연율화 수익률", f"{ann_ret_summary * 10000:.3f}bp")

            with col3:
                vol_summary = float(summary_dict.get('Volatility', 0))
                st.metric("연율화 변동성", f"{vol_summary * 10000:.3f}bp")

            with col4:
                mdd_summary = float(summary_dict.get('Max_Drawdown', 0))
                st.metric("Max Drawdown", f"{mdd_summary * 100:.2f}%")

            with col5:
                trading_days = int(summary_dict.get('Trading_Days', 0))
                st.metric("총 거래일", f"{trading_days}일")

            start_date = summary_dict.get('Start_Date', 'N/A')
            end_date = summary_dict.get('End_Date', 'N/A')
            st.caption(f"📅 전체 기간: {start_date} ~ {end_date}")

        # ===== Drawdown 차트 =====
        st.markdown("---")
        st.subheader("📉 Drawdown 추이")

        cum_series = (1 + filtered_returns['Actual_Portfolio_Return']).cumprod()
        running_max = cum_series.expanding().max()
        drawdown = (cum_series - running_max) / running_max

        fig_dd = go.Figure()

        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,  # % 단위
            mode='lines',
            name='Drawdown',
            line=dict(color='#d62728', width=2),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.1)',
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>'
        ))

        # MDD 표시
        mdd_value = drawdown.min()
        mdd_date = drawdown.idxmin()

        fig_dd.add_trace(go.Scatter(
            x=[mdd_date],
            y=[mdd_value * 100],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='x'),
            text=[f'MDD: {mdd_value * 100:.2f}%'],
            textposition='top center',
            name='Max Drawdown',
            showlegend=True,
            hovertemplate=f'MDD: {mdd_value * 100:.2f}%<br>{mdd_date.strftime("%Y-%m-%d")}<extra></extra>'
        ))

        fig_dd.update_layout(
            title=f"Drawdown (%) - {selected_start.strftime('%Y-%m-%d')} ~ {selected_end.strftime('%Y-%m-%d')}",
            xaxis_title="날짜",
            yaxis_title="Drawdown (%)",
            height=400,
            hovermode='x unified'
        )

        fig_dd.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_dd = apply_chart_font_settings(fig_dd)

        st.plotly_chart(fig_dd, use_container_width=True)

        # ===== 일별 수익률 분포 =====
        st.markdown("---")
        st.subheader("📊 일별 수익률 분포")

        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            # 히스토그램
            fig_hist = go.Figure()

            fig_hist.add_trace(go.Histogram(
                x=filtered_returns['Actual_Portfolio_Return'] * 10000,
                nbinsx=50,
                name='일별 수익률',
                marker_color='#1f77b4',
                opacity=0.7
            ))

            # 평균선
            mean_ret = filtered_returns['Actual_Portfolio_Return'].mean() * 10000
            fig_hist.add_vline(
                x=mean_ret,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"평균: {mean_ret:.3f}bp"
            )

            fig_hist.update_layout(
                title="일별 수익률 분포 (bp)",
                xaxis_title="일별 수익률 (bp)",
                yaxis_title="빈도",
                height=400,
                showlegend=False
            )

            fig_hist = apply_chart_font_settings(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_dist2:
            # 통계 요약
            st.markdown("#### 📈 분포 통계")

            returns_bp = filtered_returns['Actual_Portfolio_Return'] * 10000

            stats_col1, stats_col2 = st.columns(2)

            with stats_col1:
                st.metric("평균", f"{returns_bp.mean():.3f}bp")
                st.metric("중앙값", f"{returns_bp.median():.3f}bp")
                st.metric("표준편차", f"{returns_bp.std():.3f}bp")

            with stats_col2:
                st.metric("최대값", f"{returns_bp.max():.3f}bp")
                st.metric("최소값", f"{returns_bp.min():.3f}bp")

                # 양수/음수 비율
                positive_days = (returns_bp > 0).sum()
                total_days = len(returns_bp)
                win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
                st.metric("양수 비율", f"{win_rate:.1f}%")

        # ===== 월별 수익률 히트맵 =====
        st.markdown("---")
        st.subheader("🗓️ 월별 수익률 히트맵")

        # 월별 수익률 계산
        monthly_returns = filtered_returns['Actual_Portfolio_Return'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )

        if len(monthly_returns) > 0:
            # 연도와 월로 분리
            monthly_returns_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 10000  # bp 단위
            })

            # 피벗 테이블 생성
            pivot_monthly = monthly_returns_df.pivot(
                index='Year',
                columns='Month',
                values='Return'
            )

            # 월 이름으로 변경
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_monthly.columns = [month_names[i - 1] for i in pivot_monthly.columns]

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_monthly.values,
                x=pivot_monthly.columns,
                y=pivot_monthly.index,
                colorscale='RdYlGn',
                zmid=0,
                text=pivot_monthly.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 20},
                colorbar=dict(title="bp")
            ))

            fig_heatmap.update_layout(
                title=f"월별 수익률 (bp) - {selected_start.strftime('%Y-%m-%d')} ~ {selected_end.strftime('%Y-%m-%d')}",
                xaxis_title="월",
                yaxis_title="연도",
                height=max(300, len(pivot_monthly) * 40)  # 동적 높이
            )

            fig_heatmap = apply_chart_font_settings(fig_heatmap)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("선택한 기간에 월별 데이터가 충분하지 않습니다.")

        # ===== 포지션 정보 =====
        if not actual_positions_df.empty:
            st.markdown("---")
            st.subheader("📋 최근 포지션 정보")

            # 선택 기간 내의 포지션만 필터링
            period_positions = actual_positions_df[
                (actual_positions_df['Date'] >= selected_start) &
                (actual_positions_df['Date'] <= selected_end)
                ].copy()

            if not period_positions.empty:
                # 최근 날짜
                latest_date = period_positions['Date'].max()
                latest_positions = period_positions[
                    period_positions['Date'] == latest_date
                    ].copy()

                if not latest_positions.empty:
                    st.markdown(f"**📅 기준일: {latest_date.strftime('%Y-%m-%d')}**")

                    # 요약 통계
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

                    with col_p1:
                        n_pairs = len(latest_positions)
                        st.metric("활성 Pair", f"{n_pairs}개")

                    with col_p2:
                        total_pnl_bp = latest_positions['Position_PnL_bp'].sum()
                        st.metric("당일 총 P&L", f"{total_pnl_bp:.3f}bp")

                    with col_p3:
                        avg_size = latest_positions['Size'].mean()
                        st.metric("평균 포지션", f"{avg_size:.4f}")

                    with col_p4:
                        n_cash_pairs = latest_positions['Is_Cash_Pair'].sum()
                        st.metric("Cash Pair", f"{n_cash_pairs}개")

                    # 상세 테이블
                    with st.expander("📊 포지션 상세 정보", expanded=False):
                        display_positions = latest_positions[[
                            'Pair_ID', 'Pair', 'Size', 'Direction',
                            'Spread_Return_%', 'Position_PnL_bp'
                        ]].copy()

                        # 포맷팅
                        display_positions['Spread_Return_%'] = display_positions['Spread_Return_%'].apply(
                            lambda x: f"{x:.2f}%"
                        )
                        display_positions['Position_PnL_bp'] = display_positions['Position_PnL_bp'].apply(
                            lambda x: f"{x:.3f}"
                        )
                        display_positions['Size'] = display_positions['Size'].apply(
                            lambda x: f"{x:.4f}"
                        )

                        st.dataframe(display_positions, use_container_width=True)

                    # 기간 내 포지션 진입 이력
                    st.markdown("#### 📝 포지션 진입 이력")

                    entry_positions = period_positions[
                        period_positions['Event'].str.contains('ENTRY', case=False, na=False)
                    ].copy()

                    if not entry_positions.empty:
                        st.info(f"💡 선택 기간 내 {len(entry_positions)}건의 포지션 진입")

                        # 진입 날짜별로 그룹화
                        entry_by_date = entry_positions.groupby('Date').agg({
                            'Pair_ID': 'count',
                            'Pair': lambda x: ', '.join(x.unique())
                        }).reset_index()
                        entry_by_date.columns = ['진입일', 'Pair 수', 'Pair 목록']
                        entry_by_date['진입일'] = entry_by_date['진입일'].dt.strftime('%Y-%m-%d')

                        st.dataframe(entry_by_date, use_container_width=True)
            else:
                st.info("선택한 기간에 포지션 데이터가 없습니다.")

        # =========================================================================
        # 새로운 섹션: 포지션별 상세 통계 (Position Statistics)
        # =========================================================================
        st.markdown("---")
        st.subheader("📊 포지션별 상세 통계")

        try:
            # 포지션 통계 파일 로드
            position_stats_path = os.path.join(base_dir, 'actual_portfolio_position_statistics.csv')
            if os.path.exists(position_stats_path):
                position_stats_df = pd.read_csv(position_stats_path)

                if not position_stats_df.empty:
                    st.success(f"✅ {len(position_stats_df)}개 포지션 통계 로드 완료")

                    # 포지션 타입별 필터
                    col_filter1, col_filter2 = st.columns(2)
                    with col_filter1:
                        pos_types = ['전체'] + list(position_stats_df['Position_Type'].unique())
                        selected_pos_type = st.selectbox("포지션 타입", pos_types, key="pos_type_filter")

                    with col_filter2:
                        status_options = ['전체'] + list(position_stats_df['Status'].unique())
                        selected_status = st.selectbox("상태", status_options, key="status_filter")

                    # 필터링
                    filtered_stats = position_stats_df.copy()
                    if selected_pos_type != '전체':
                        filtered_stats = filtered_stats[filtered_stats['Position_Type'] == selected_pos_type]
                    if selected_status != '전체':
                        filtered_stats = filtered_stats[filtered_stats['Status'] == selected_status]

                    # Size를 bp로 변환 (소수 → bp)
                    if 'Size' in filtered_stats.columns:
                        filtered_stats['Size_bp'] = filtered_stats['Size'] * 10000

                    # TE (bp) 계산 - 공분산 행렬 기반 Marginal Contribution to TE
                    # filtered_stats 자체의 Long_Asset, Short_Asset, Size 정보를 사용하여 계산
                    total_te_bp = 0.0
                    filtered_stats['TE_bp'] = 0.0
                    filtered_stats['TE_Contribution_Pct'] = 0.0

                    try:
                        # 공분산 행렬 기반 TE 계산 시도
                        # filtered_stats에서 Long_Asset, Short_Asset, Size 정보를 직접 사용
                        has_asset_info = ('Long_Asset' in filtered_stats.columns and
                                          'Short_Asset' in filtered_stats.columns and
                                          'Size' in filtered_stats.columns)

                        if not returns_by_asset.empty and not w_opt_daily.empty and not w_bmk_daily.empty and has_asset_info:
                            asof = min(w_opt_daily.index.max(), w_bmk_daily.index.max())
                            Wopt_last = w_opt_daily.loc[asof].fillna(0.0)

                            cols = [c for c in returns_by_asset.columns if c in Wopt_last.index]
                            R = returns_by_asset[cols]
                            R_dec = _pc_ensure_decimal_returns(R)
                            C = _pc_build_recent_cov_constant_corr(R_dec, window=63, rho=0.25)

                            # 현재 포지션의 active weights 계산 - filtered_stats (position_stats_df) 사용
                            # Open 상태인 포지션만 사용 (Status == 'Open')
                            open_positions = filtered_stats[filtered_stats['Status'] == 'Open'].copy()

                            # Open 포지션이 없으면 전체 포지션 사용 (히스토리 분석용)
                            if open_positions.empty:
                                open_positions = filtered_stats.copy()

                            w_active = pd.Series(0.0, index=cols)

                            for _, row in open_positions.iterrows():
                                long_asset = str(row.get('Long_Asset', ''))
                                short_asset = str(row.get('Short_Asset', ''))
                                # Size는 소수 형태
                                size_decimal = float(row.get('Size', 0.0))
                                pos_bp = abs(size_decimal)  # 소수 형태

                                # Position_Type에 따라 처리
                                pos_type = str(row.get('Position_Type', 'Pair'))

                                if pos_type == 'Single':
                                    # 단일 포지션: Long_Asset 또는 Short_Asset 중 유효한 것만
                                    if long_asset and long_asset in w_active.index and long_asset != 'Cash':
                                        if size_decimal >= 0:
                                            w_active[long_asset] += pos_bp
                                        else:
                                            w_active[long_asset] -= pos_bp
                                    elif short_asset and short_asset in w_active.index and short_asset != 'Cash':
                                        if size_decimal >= 0:
                                            w_active[short_asset] -= pos_bp
                                        else:
                                            w_active[short_asset] += pos_bp
                                else:
                                    # Pair 포지션: Long 매수 + Short 매도
                                    if long_asset and long_asset in w_active.index:
                                        w_active[long_asset] += pos_bp
                                    if short_asset and short_asset in w_active.index and short_asset != 'Cash':
                                        w_active[short_asset] -= pos_bp

                            # 포트폴리오 전체 TE 계산
                            total_te_bp = _pc_te_bp_from_cov(w_active.values, C, 252)

                            # Marginal Contribution to TE 계산
                            # MCTE_i = (C @ w)_i / TE
                            # 포지션별 TE 기여 = |w_i| * MCTE_i
                            if total_te_bp > 0:
                                C_np = C.values if hasattr(C, 'values') else C
                                w_np = w_active.values
                                Cw = C_np @ w_np  # 공분산 행렬 * 가중치
                                portfolio_var = w_np @ Cw
                                te_annual = np.sqrt(portfolio_var * 252)

                                if te_annual > 0:
                                    mcte = Cw * np.sqrt(252) / te_annual  # Marginal contribution

                                    # Pair_ID별 TE 기여도 계산 (filtered_stats 기반)
                                    pair_te_contrib = {}
                                    for _, fs_row in filtered_stats.iterrows():
                                        pair_id = fs_row.get('Pair_ID', '')
                                        long_asset = str(fs_row.get('Long_Asset', ''))
                                        short_asset = str(fs_row.get('Short_Asset', ''))
                                        size_decimal = float(fs_row.get('Size', 0.0))
                                        pos_bp = abs(size_decimal)  # 소수 형태
                                        pos_type = str(fs_row.get('Position_Type', 'Pair'))

                                        te_contrib = 0.0
                                        if pos_type == 'Single':
                                            # 단일 포지션
                                            if long_asset in cols:
                                                long_idx = cols.index(long_asset)
                                                te_contrib += pos_bp * abs(mcte[long_idx]) * 10000
                                            elif short_asset in cols and short_asset != 'Cash':
                                                short_idx = cols.index(short_asset)
                                                te_contrib += pos_bp * abs(mcte[short_idx]) * 10000
                                        else:
                                            # Pair 포지션
                                            if long_asset in cols:
                                                long_idx = cols.index(long_asset)
                                                te_contrib += pos_bp * abs(mcte[long_idx]) * 10000
                                            if short_asset in cols and short_asset != 'Cash':
                                                short_idx = cols.index(short_asset)
                                                te_contrib += pos_bp * abs(mcte[short_idx]) * 10000

                                        pair_te_contrib[pair_id] = te_contrib

                                    # filtered_stats에 TE 기여도 매핑
                                    for idx, row in filtered_stats.iterrows():
                                        pair_id = row.get('Pair_ID', '')
                                        filtered_stats.at[idx, 'TE_bp'] = pair_te_contrib.get(pair_id, 0.0)

                                    # TE 기여도 비율 계산
                                    total_te_contrib = filtered_stats['TE_bp'].sum()
                                    if total_te_contrib > 0:
                                        filtered_stats['TE_Contribution_Pct'] = (filtered_stats['TE_bp'] / total_te_contrib) * 100
                                        # 총 TE와 일치하도록 스케일 조정
                                        scale_factor = total_te_bp / total_te_contrib if total_te_contrib > 0 else 1.0
                                        filtered_stats['TE_bp'] = filtered_stats['TE_bp'] * scale_factor

                            # session state에 저장 (리스크 시뮬레이션 섹션에서도 사용)
                            st.session_state['current_portfolio_te_bp'] = total_te_bp
                            st.session_state['cov_matrix'] = C
                            st.session_state['asset_cols'] = cols

                    except Exception as e:
                        st.warning(f"TE 계산 중 오류 발생: {e}")
                        total_te_bp = 0.0

                    # 요약 지표
                    col_s1, col_s2, col_s3, col_s4, col_s5, col_s6 = st.columns(6)

                    with col_s1:
                        total_return_bp = filtered_stats['Total_Return_bp'].sum()
                        st.metric("총 수익률", f"{total_return_bp:.2f}bp")

                    with col_s2:
                        avg_sharpe = filtered_stats['Sharpe_Ratio'].mean()
                        st.metric("평균 Sharpe", f"{avg_sharpe:.2f}")

                    with col_s3:
                        avg_win_rate = filtered_stats['Win_Rate_%'].mean()
                        st.metric("평균 Win Rate", f"{avg_win_rate:.1f}%")

                    with col_s4:
                        st.metric("총 TE", f"{total_te_bp:.2f}bp")

                    with col_s5:
                        avg_holding = filtered_stats['Holding_Days'].mean()
                        st.metric("평균 보유일", f"{avg_holding:.0f}일")

                    with col_s6:
                        n_positions = len(filtered_stats)
                        st.metric("포지션 수", f"{n_positions}개")

                    # 상세 테이블
                    with st.expander("📋 포지션별 상세 통계 테이블", expanded=True):
                        # 표시할 컬럼 선택 (Size_bp, TE_bp와 TE_Contribution_Pct 추가)
                        display_cols = [
                            'Pair_ID', 'Pair', 'Position_Type', 'Direction', 'Size_bp',
                            'Entry_Date', 'Exit_Date', 'Status', 'Holding_Days',
                            'Total_Return_bp', 'Avg_Daily_Return_bp', 'Annualized_Volatility',
                            'Sharpe_Ratio', 'Max_Drawdown', 'TE_bp', 'TE_Contribution_Pct',
                            'Win_Rate_%', 'Best_Day_bp', 'Worst_Day_bp'
                        ]
                        available_cols = [c for c in display_cols if c in filtered_stats.columns]

                        display_df = filtered_stats[available_cols].copy()

                        # 포트폴리오 총합 행 추가
                        total_row = pd.DataFrame([{
                            'Pair_ID': '합계',
                            'Pair': '포트폴리오 총합',
                            'Position_Type': '-',
                            'Direction': '-',
                            'Size_bp': filtered_stats['Size_bp'].sum() if 'Size_bp' in filtered_stats.columns else 0,
                            'Entry_Date': '-',
                            'Exit_Date': '-',
                            'Status': '-',
                            'Holding_Days': filtered_stats['Holding_Days'].mean() if 'Holding_Days' in filtered_stats.columns else 0,
                            'Total_Return_bp': filtered_stats['Total_Return_bp'].sum() if 'Total_Return_bp' in filtered_stats.columns else 0,
                            'Avg_Daily_Return_bp': filtered_stats['Avg_Daily_Return_bp'].mean() if 'Avg_Daily_Return_bp' in filtered_stats.columns else 0,
                            'Annualized_Volatility': None,  # 합계에서는 표시 안함
                            'Sharpe_Ratio': filtered_stats['Sharpe_Ratio'].mean() if 'Sharpe_Ratio' in filtered_stats.columns else 0,
                            'Max_Drawdown': filtered_stats['Max_Drawdown'].sum() if 'Max_Drawdown' in filtered_stats.columns else 0,
                            'TE_bp': total_te_bp,
                            'TE_Contribution_Pct': 100.0,  # 합계 = 100%
                            'Win_Rate_%': filtered_stats['Win_Rate_%'].mean() if 'Win_Rate_%' in filtered_stats.columns else 0,
                            'Best_Day_bp': filtered_stats['Best_Day_bp'].max() if 'Best_Day_bp' in filtered_stats.columns else 0,
                            'Worst_Day_bp': filtered_stats['Worst_Day_bp'].min() if 'Worst_Day_bp' in filtered_stats.columns else 0,
                        }])

                        display_df = pd.concat([display_df, total_row], ignore_index=True)

                        # 포맷팅
                        format_cols = {
                            'Size_bp': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                            'Total_Return_bp': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                            'Avg_Daily_Return_bp': lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x,
                            'Annualized_Volatility': lambda x: f"{x:.2%}" if pd.notna(x) and isinstance(x, (int, float)) else "-",
                            'Sharpe_Ratio': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                            'Max_Drawdown': lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x,
                            'TE_bp': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                            'TE_Contribution_Pct': lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x,
                            'Win_Rate_%': lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x,
                            'Best_Day_bp': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                            'Worst_Day_bp': lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
                        }

                        for col, fmt in format_cols.items():
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(fmt)

                        # 컬럼명 한글화
                        col_names = {
                            'Pair_ID': 'Pair ID',
                            'Pair': '페어/자산',
                            'Position_Type': '유형',
                            'Direction': '방향',
                            'Size_bp': '사이즈(bp)',
                            'Entry_Date': '진입일',
                            'Exit_Date': '청산일',
                            'Status': '상태',
                            'Holding_Days': '보유일',
                            'Total_Return_bp': '총수익(bp)',
                            'Avg_Daily_Return_bp': '일평균(bp)',
                            'Annualized_Volatility': '연변동성',
                            'Sharpe_Ratio': 'Sharpe',
                            'Max_Drawdown': 'MDD',
                            'TE_bp': 'TE(bp)',
                            'TE_Contribution_Pct': 'TE기여(%)',
                            'Win_Rate_%': 'Win Rate',
                            'Best_Day_bp': '최고일(bp)',
                            'Worst_Day_bp': '최저일(bp)',
                        }
                        display_df.rename(columns=col_names, inplace=True)

                        # 마지막 행(총합) 스타일 강조를 위해 표시
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # 수익 기여도 차트
                    st.markdown("#### 📈 포지션별 수익 기여도")

                    chart_df = filtered_stats[['Pair', 'Total_Return_bp']].copy()
                    chart_df = chart_df.sort_values('Total_Return_bp', ascending=True)

                    fig_contrib = go.Figure()
                    colors = ['#EF553B' if x < 0 else '#00CC96' for x in chart_df['Total_Return_bp']]

                    fig_contrib.add_trace(go.Bar(
                        y=chart_df['Pair'],
                        x=chart_df['Total_Return_bp'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{x:.2f}bp" for x in chart_df['Total_Return_bp']],
                        textposition='outside'
                    ))

                    fig_contrib.update_layout(
                        title="포지션별 총 수익 기여도 (bp)",
                        xaxis_title="수익률 (bp)",
                        yaxis_title="",
                        height=max(400, len(chart_df) * 30),
                        showlegend=False
                    )

                    st.plotly_chart(fig_contrib, use_container_width=True)

                    # TE 기여도 파이 차트 (새 TE_Contribution_Pct 사용)
                    if 'TE_Contribution_Pct' in filtered_stats.columns and filtered_stats['TE_Contribution_Pct'].sum() > 0:
                        st.markdown("#### 🎯 TE 기여도 분포")

                        # 총합 행 제외 (filtered_stats에는 아직 추가 안됨)
                        te_df = filtered_stats[filtered_stats['TE_Contribution_Pct'] > 0][['Pair', 'TE_bp', 'TE_Contribution_Pct']].copy()
                        if not te_df.empty:
                            fig_te = go.Figure(data=[go.Pie(
                                labels=te_df['Pair'],
                                values=te_df['TE_Contribution_Pct'],
                                hole=0.4,
                                textinfo='label+percent',
                                textposition='outside',
                                hovertemplate="<b>%{label}</b><br>TE: %{customdata:.2f}bp<br>기여도: %{percent}<extra></extra>",
                                customdata=te_df['TE_bp']
                            )])

                            fig_te.update_layout(
                                title=f"포지션별 TE 기여도 (총 TE: {total_te_bp:.2f}bp)",
                                height=450
                            )

                            st.plotly_chart(fig_te, use_container_width=True)

                    # 다운로드 버튼
                    csv_stats = filtered_stats.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 포지션 통계 다운로드 (CSV)",
                        data=csv_stats,
                        file_name=f"position_statistics_{selected_start.strftime('%Y%m%d')}_{selected_end.strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="download_position_stats"
                    )

                else:
                    st.info("포지션 통계 데이터가 비어있습니다.")
            else:
                st.warning("📁 actual_portfolio_position_statistics.csv 파일이 없습니다. itaa_v5.py를 실행하여 생성해주세요.")

        except Exception as e:
            st.error(f"포지션 통계 로드 중 오류: {e}")

        # =========================================================================
        # 새로운 섹션: 리스크 시뮬레이션 & P&L 분포
        # =========================================================================
        st.markdown("---")
        st.subheader("🔬 리스크 시뮬레이션 & 포지션 조정")

        # ===== 실제 포지션 데이터에서 최신 포지션 추출 =====
        try:
            # 최신 날짜의 포지션 추출
            latest_date = actual_positions_df['Date'].max()
            latest_positions = actual_positions_df[actual_positions_df['Date'] == latest_date].copy()

            if not latest_positions.empty:
                # 실제 포지션 데이터를 common_positions 형식으로 변환
                actual_common_positions = []

                for idx, row in latest_positions.iterrows():
                    pair_id = row.get('Pair_ID', f'P{idx:03d}')
                    pair_name = row.get('Pair', 'Unknown')

                    # Long/Short 자산 추출: 컬럼 값 우선, 없으면 Pair 문자열 파싱
                    long_asset = row.get('Long_Asset', None)
                    short_asset = row.get('Short_Asset', None)
                    long_asset = long_asset if isinstance(long_asset, str) and long_asset.strip() else None
                    short_asset = short_asset if isinstance(short_asset, str) and short_asset.strip() else None

                    if not long_asset:
                        if 'vs' in pair_name:
                            parts = pair_name.split(' vs ')
                            long_asset = parts[0].strip() if len(parts) > 0 else 'Unknown'
                            short_asset = parts[1].strip() if len(parts) > 1 else 'Unknown'
                        else:
                            long_asset = pair_name
                            short_asset = short_asset or 'Cash'
                    else:
                        # Long/Short 컬럼이 있으면 Cash 처리만 보완
                        if not short_asset:
                            short_asset = 'Cash'

                    # 실제 포지션 크기 (Size 또는 Position_PnL_bp 등에서 역산)
                    size = row.get('Size', 0.0)  # 실제 포지션 크기
                    signal = 2 if size > 0 else (-2 if size < 0 else 0)  # 크기 기반 신호 추정

                    # Cash pair 판단
                    is_cash = 'cash' in short_asset.lower() or 'tbill' in short_asset.lower()
                    leg_factor = 1 if is_cash else 2

                    # Size를 bp로 변환 (Size가 소수 형태라고 가정)
                    per_leg_bp = abs(size) * 10000  # 소수 → bp

                    actual_common_positions.append({
                        'Pair_ID': pair_id,
                        'Pair': pair_name,
                        'Long_Asset': long_asset,
                        'Short_Asset': short_asset,
                        'Signal': signal,
                        'Is_Cash_Pair': is_cash,
                        'Leg_Factor': leg_factor,
                        'Risk_Unit_3M_%': 5.0,  # 기본값 (실제 계산 필요 시 추가)
                        'Max_Loss_bp': 0.10,
                        'Per_Leg_Position_bp': per_leg_bp,
                        'Total_Notional_bp': per_leg_bp * leg_factor,
                        'Constraint_Method': 'Actual'
                    })

                common_positions = pd.DataFrame(actual_common_positions)

                st.info(f"📅 **최근 포지션 날짜**: {latest_date.strftime('%Y-%m-%d')} ({len(common_positions)}개 페어)")

            else:
                st.warning("실제 포지션 데이터가 없습니다.")
                common_positions = pd.DataFrame()

        except Exception as e:
            st.error(f"실제 포지션 데이터 처리 중 오류: {e}")
            # Fallback: Session state의 common_positions 사용
            if 'common_positions' in st.session_state and st.session_state.common_positions is not None:
                common_positions = st.session_state.common_positions
                st.warning("⚠️ 실제 포지션 로드 실패. Asset View 탭의 이론적 포지션을 사용합니다.")
            else:
                common_positions = pd.DataFrame()

        if not common_positions.empty:
            # ===== 현재 포지션 정보 테이블 =====
            st.markdown("### 📋 현재 포지션 현황")

            # 현재 포지션 테이블 생성
            current_pos_display = common_positions[[
                'Pair_ID', 'Pair', 'Long_Asset', 'Short_Asset', 'Signal',
                'Leg_Factor', 'Per_Leg_Position_bp', 'Total_Notional_bp',
                'Risk_Unit_3M_%', 'Max_Loss_bp'
            ]].copy()

            # 컬럼명 한글화
            current_pos_display.columns = [
                'Pair ID', '페어', 'Long 자산', 'Short 자산', 'Signal',
                'Leg Factor', '레그당 포지션 (bp)', '총 명목 (bp)',
                'Risk Unit 3M (%)', '최대손실 (bp)'
            ]

            # 숫자 포맷팅
            current_pos_display['레그당 포지션 (bp)'] = current_pos_display['레그당 포지션 (bp)'].apply(lambda x: f"{x:.3f}")
            current_pos_display['총 명목 (bp)'] = current_pos_display['총 명목 (bp)'].apply(lambda x: f"{x:.3f}")
            current_pos_display['Risk Unit 3M (%)'] = current_pos_display['Risk Unit 3M (%)'].apply(lambda x: f"{x:.2f}")
            current_pos_display['최대손실 (bp)'] = current_pos_display['최대손실 (bp)'].apply(lambda x: f"{x:.2f}")

            st.dataframe(current_pos_display, use_container_width=True, hide_index=True)

            # 요약 통계
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            with col_sum1:
                total_notional = common_positions['Total_Notional_bp'].abs().sum()
                st.metric("총 명목 포지션", f"{total_notional:.2f}bp")
            with col_sum2:
                avg_position = common_positions['Per_Leg_Position_bp'].abs().mean()
                st.metric("평균 레그당 포지션", f"{avg_position:.3f}bp")
            with col_sum3:
                n_pairs = len(common_positions)
                st.metric("활성 페어 수", f"{n_pairs}개")
            with col_sum4:
                avg_signal = common_positions['Signal'].abs().mean()
                st.metric("평균 Signal 강도", f"{avg_signal:.1f}")

            st.markdown("---")

            # ===== 1. 포지션 크기 조정 UI =====
            st.markdown("### 📊 1. 포지션 크기 조정 → 리스크 변화")
            st.caption("기존 포지션 변경 및 유니버스 내 모든 자산의 Long/Short 추가가 가능합니다.")

            # Session state 초기화
            if 'adjusted_positions_tab11' not in st.session_state:
                st.session_state.adjusted_positions_tab11 = {}
            if 'position_types_tab11' not in st.session_state:
                st.session_state.position_types_tab11 = {}
            if 'new_positions_tab11' not in st.session_state:
                st.session_state.new_positions_tab11 = []

            # 조정된 포지션 저장
            adjusted_sizes = {}
            position_types = {}

            # ===== 기존 포지션 조정 =====
            st.markdown("#### 🎚️ 기존 포지션 조정")
            st.caption("💡 포지션 타입과 크기를 변경할 수 있습니다.")

            for idx, row in common_positions.iterrows():
                pair_id = row['Pair_ID']
                pair_name = row['Pair']
                current_size = float(row['Per_Leg_Position_bp'])
                signal = float(row['Signal'])
                long_asset = row['Long_Asset']
                short_asset = row['Short_Asset']

                # 슬라이더 범위 설정 (현재 크기의 ±200%)
                abs_current = abs(current_size)
                max_size = max(abs_current * 3, 5.0)  # 최소 5bp

                with st.expander(f"**{pair_name}** (Signal: {signal:.0f})", expanded=(idx < 3)):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        # 포지션 타입 선택 - 5가지 옵션
                        pos_type_options = [
                            "Pair (L/S)",
                            f"Long: {long_asset}",
                            f"Long: {short_asset}",
                            f"Short: {long_asset}",
                            f"Short: {short_asset}"
                        ]
                        pos_type = st.selectbox(
                            "포지션 타입",
                            pos_type_options,
                            index=0,
                            key=f"pos_type_{pair_id}_tab11",
                            help="Pair: Long/Short 동시 포지션\n개별 자산별 Long 또는 Short 선택 가능"
                        )

                        # 포지션 타입 분류 (자산명과 방향 저장)
                        if pos_type == "Pair (L/S)":
                            position_types[pair_id] = {'type': 'pair', 'asset': None, 'direction': None}
                        elif pos_type == f"Long: {long_asset}":
                            position_types[pair_id] = {'type': 'single', 'asset': long_asset, 'direction': 'long'}
                        elif pos_type == f"Long: {short_asset}":
                            position_types[pair_id] = {'type': 'single', 'asset': short_asset, 'direction': 'long'}
                        elif pos_type == f"Short: {long_asset}":
                            position_types[pair_id] = {'type': 'single', 'asset': long_asset, 'direction': 'short'}
                        elif pos_type == f"Short: {short_asset}":
                            position_types[pair_id] = {'type': 'single', 'asset': short_asset, 'direction': 'short'}

                    with col2:
                        # 포지션 크기 입력
                        new_size = st.number_input(
                            f"포지션 (bp)",
                            min_value=-max_size,
                            max_value=max_size,
                            value=float(current_size),
                            step=0.01,
                            key=f"pos_size_{pair_id}_tab11",
                            label_visibility="collapsed"
                        )
                        adjusted_sizes[pair_id] = new_size

                    with col3:
                        change_pct = ((new_size - current_size) / current_size * 100) if current_size != 0 else 0
                        if abs(change_pct) > 0.1:
                            color = "🟢" if change_pct > 0 else "🔴"
                            st.markdown(f"{color} {change_pct:+.1f}%")
                        else:
                            st.markdown("➖")

                    # 포지션 타입별 설명
                    pos_info = position_types[pair_id]
                    if pos_info['type'] == 'pair':
                        st.info(f"⚖️ **Pair**: {long_asset} 매수 + {short_asset} 매도 각 {abs(new_size):.2f}bp")
                    else:
                        asset_name = pos_info['asset']
                        direction = pos_info['direction']
                        if direction == 'long':
                            st.info(f"📈 **Long**: {asset_name} {abs(new_size):.2f}bp 매수")
                        else:
                            st.info(f"📉 **Short**: {asset_name} {abs(new_size):.2f}bp 매도")

            # ===== 새 포지션 추가 UI =====
            st.markdown("---")
            st.markdown("#### ➕ 새 포지션 추가")
            st.caption("유니버스 내 모든 자산의 Long 또는 Short 포지션을 추가할 수 있습니다.")

            # 유니버스 내 모든 자산 리스트 생성
            if not returns_by_asset.empty:
                all_assets = sorted(returns_by_asset.columns.tolist())

                col_add1, col_add2, col_add3, col_add4 = st.columns([2, 1, 1, 1])

                with col_add1:
                    selected_asset = st.selectbox(
                        "자산 선택",
                        all_assets,
                        key="new_asset_select_tab11"
                    )

                with col_add2:
                    new_direction = st.selectbox(
                        "방향",
                        ["Long", "Short"],
                        key="new_direction_tab11"
                    )

                with col_add3:
                    new_position_size = st.number_input(
                        "포지션 크기 (bp)",
                        min_value=0.0,
                        max_value=100.0,
                        value=1.0,
                        step=0.1,
                        key="new_position_size_tab11"
                    )

                with col_add4:
                    if st.button("➕ 추가", key="add_position_btn_tab11"):
                        new_pos = {
                            'id': f"NEW_{len(st.session_state.new_positions_tab11):03d}",
                            'asset': selected_asset,
                            'direction': new_direction.lower(),
                            'size_bp': new_position_size
                        }
                        st.session_state.new_positions_tab11.append(new_pos)
                        st.success(f"✅ {new_direction}: {selected_asset} ({new_position_size:.2f}bp) 추가됨")

                # 추가된 새 포지션 표시
                if st.session_state.new_positions_tab11:
                    st.markdown("**추가된 포지션:**")
                    for i, pos in enumerate(st.session_state.new_positions_tab11):
                        col_p1, col_p2 = st.columns([4, 1])
                        with col_p1:
                            icon = "📈" if pos['direction'] == 'long' else "📉"
                            st.markdown(f"{icon} **{pos['direction'].capitalize()}: {pos['asset']}** - {pos['size_bp']:.2f}bp")
                        with col_p2:
                            if st.button("❌", key=f"remove_new_pos_{i}_tab11"):
                                st.session_state.new_positions_tab11.pop(i)
                                st.rerun()

            # Session state 저장
            st.session_state.adjusted_positions_tab11 = adjusted_sizes
            st.session_state.position_types_tab11 = position_types

            # ===== 리스크 변화 계산 및 표시 =====
            st.markdown("---")
            st.markdown("#### 📈 예상 리스크 변화 (TE 기준)")

            # 리스크 계산을 위한 준비
            if not returns_by_asset.empty and not w_opt_daily.empty and not w_bmk_daily.empty:
                # 최신 가중치
                asof = min(w_opt_daily.index.max(), w_bmk_daily.index.max())
                Wopt_last = w_opt_daily.loc[asof].fillna(0.0)
                Wbmk_last = w_bmk_daily.loc[asof].fillna(0.0)

                # 공분산 행렬
                cols = [c for c in returns_by_asset.columns if c in Wopt_last.index]
                R = returns_by_asset[cols]
                R_dec = _pc_ensure_decimal_returns(R)
                C = _pc_build_recent_cov_constant_corr(R_dec, window=63, rho=0.25)

                w_b = Wbmk_last.reindex(cols).fillna(0.0)

                # 현재 포지션의 active weights 계산
                w_active_current = pd.Series(0.0, index=cols)
                for i, row in enumerate(common_positions.itertuples()):
                    long_asset = str(row.Long_Asset)
                    short_asset = str(row.Short_Asset)
                    pos_bp = row.Per_Leg_Position_bp / 10000.0  # bp → 소수

                    # Pair 기준 (현재 포지션은 항상 Pair로 계산)
                    if long_asset in w_active_current.index:
                        w_active_current[long_asset] += pos_bp
                    if short_asset in w_active_current.index:
                        w_active_current[short_asset] -= pos_bp

                # 현재 TE 계산
                current_te_bp = _pc_te_bp_from_cov(w_active_current.values, C, 252)

                # 조정된 포지션의 active weights 계산
                w_active_adj = pd.Series(0.0, index=cols)

                # 1) 기존 포지션 (조정 반영)
                for i, row in enumerate(common_positions.itertuples()):
                    pid = row.Pair_ID
                    long_asset = str(row.Long_Asset)
                    short_asset = str(row.Short_Asset)

                    if pid in adjusted_sizes:
                        pos_bp = adjusted_sizes[pid] / 10000.0  # bp → 소수
                    else:
                        pos_bp = row.Per_Leg_Position_bp / 10000.0

                    # 포지션 타입에 따른 가중치 적용
                    pos_info = position_types.get(pid, {'type': 'pair', 'asset': None, 'direction': None})

                    if pos_info['type'] == 'single':
                        # 단일 자산 포지션
                        target_asset = pos_info['asset']
                        direction = pos_info['direction']
                        if target_asset in w_active_adj.index:
                            if direction == 'long':
                                w_active_adj[target_asset] += pos_bp
                            else:  # short
                                w_active_adj[target_asset] -= pos_bp
                    else:
                        # Pair: Long 매수 + Short 매도
                        if long_asset in w_active_adj.index:
                            w_active_adj[long_asset] += pos_bp
                        if short_asset in w_active_adj.index:
                            w_active_adj[short_asset] -= pos_bp

                # 2) 새 포지션 추가
                for new_pos in st.session_state.new_positions_tab11:
                    asset = new_pos['asset']
                    direction = new_pos['direction']
                    size_bp = new_pos['size_bp'] / 10000.0  # bp → 소수

                    if asset in w_active_adj.index:
                        if direction == 'long':
                            w_active_adj[asset] += size_bp
                        else:  # short
                            w_active_adj[asset] -= size_bp

                # 조정 후 TE 계산
                adj_te_bp = _pc_te_bp_from_cov(w_active_adj.values, C, 252)

                # 메트릭 표시
                col_te1, col_te2, col_te3 = st.columns(3)

                with col_te1:
                    st.metric("현재 TE", f"{current_te_bp:.2f}bp")

                with col_te2:
                    delta_te = adj_te_bp - current_te_bp
                    st.metric("조정 후 TE", f"{adj_te_bp:.2f}bp", delta=f"{delta_te:+.2f}bp")

                with col_te3:
                    te_change_pct = (delta_te/current_te_bp*100) if current_te_bp > 0 else 0
                    st.metric("TE 변화율", f"{te_change_pct:.1f}%")

                # 현재 TE를 session state에 저장 (포지션별 상세 통계에서 사용)
                st.session_state['current_portfolio_te_bp'] = current_te_bp
            else:
                st.warning("리스크 계산에 필요한 데이터가 없습니다.")
                current_te_bp = 0.0
                st.session_state['current_portfolio_te_bp'] = 0.0

            # ===== 2. 목표 리스크 → 포지션 역산 =====
            st.markdown("---")
            st.markdown("### 🎯 2. 목표 리스크 변화 → 포지션 크기 역산")
            st.caption("원하는 리스크 변화량을 입력하면 선택한 페어의 포지션 크기를 자동 계산합니다.")

            col_target1, col_target2 = st.columns(2)

            with col_target1:
                # 페어 선택
                pair_options = common_positions['Pair'].tolist()
                selected_pair_for_calc = st.selectbox(
                    "조정할 페어 선택",
                    pair_options,
                    key="selected_pair_risk_calc"
                )

                # 선택된 페어의 현재 정보
                selected_row = common_positions[common_positions['Pair'] == selected_pair_for_calc].iloc[0]
                current_pos_bp_calc = float(selected_row['Per_Leg_Position_bp'])
                risk_unit = float(selected_row['Risk_Unit_3M_%']) / 100.0
                leg_factor_calc = int(selected_row['Leg_Factor'])

                st.info(f"현재 포지션: {current_pos_bp_calc:.3f}bp | Risk Unit: {risk_unit*100:.2f}%")

            with col_target2:
                # 목표 리스크 변화 입력
                target_risk_change = st.number_input(
                    "목표 리스크 변화량 (bp)",
                    min_value=-10.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.01,
                    key="target_risk_change"
                )

                if target_risk_change != 0 and risk_unit > 0:
                    # 포지션 변화량 계산
                    # Risk = Position × Risk_Unit × Leg_Factor
                    # ΔRisk = ΔPosition × Risk_Unit × Leg_Factor
                    # ΔPosition = ΔRisk / (Risk_Unit × Leg_Factor)

                    delta_pos_bp = target_risk_change / (risk_unit * leg_factor_calc)
                    new_pos_bp_calc = current_pos_bp_calc + delta_pos_bp

                    st.success(f"📌 계산된 포지션 변화:")
                    st.markdown(f"- **포지션 변화량**: {delta_pos_bp:+.3f}bp")
                    st.markdown(f"- **새 포지션 크기**: {new_pos_bp_calc:.3f}bp")
                    st.markdown(f"- **변화율**: {(delta_pos_bp/current_pos_bp_calc*100) if current_pos_bp_calc != 0 else 0:+.1f}%")
                elif risk_unit == 0:
                    st.warning("Risk Unit이 0입니다. 계산할 수 없습니다.")

            # ===== 3. 손익 분포 그래프 (std 시나리오) =====
            st.markdown("---")
            st.markdown("### 📈 3. 손익 분포 그래프 (-3σ ~ +3σ)")
            st.caption("현재 포지션과 조정된 포지션의 시나리오별 예상 손익을 비교합니다.")

            st.markdown("""
            **계산 기준:**
            - **기준 수익률**: 각 자산/페어의 3개월 롤링 수익률 분포 (mean ± n×std)
            - **현재 P&L**: (현재 포지션 bp) × (시나리오 수익률) × (Pair의 leg_factor)
            - **조정 P&L**: (조정 포지션 bp) × (시나리오 수익률) × (포지션 타입별 leg_factor)
              - Pair: leg_factor (2 또는 1)
              - 단일 자산 Long/Short: 1
            - **전체 포트폴리오 P&L**: 각 페어별 P&L의 단순 합계 (상관관계 미반영)
            """)

            # 각 페어별 3M 롤링 리턴 통계 계산
            scenario_data = []
            std_levels = [-3, -2, -1, 0, 1, 2, 3]

            for idx, row in common_positions.iterrows():
                pair_id = row['Pair_ID']
                long_asset = row['Long_Asset']
                short_asset = row['Short_Asset']
                signal = float(row['Signal'])
                current_pos = float(row['Per_Leg_Position_bp']) / 10000.0  # bp → 소수
                adj_pos = adjusted_sizes.get(pair_id, row['Per_Leg_Position_bp']) / 10000.0
                leg_factor = int(row['Leg_Factor'])

                # 포지션 타입 가져오기
                pos_info = position_types.get(pair_id, {'type': 'pair', 'asset': None, 'direction': None})

                # 포지션 타입에 따른 수익률 계산
                if pos_info['type'] == 'single':
                    # 단일 자산 포지션
                    target_asset = pos_info['asset']
                    direction = pos_info['direction']

                    if target_asset in returns_by_asset.columns:
                        asset_returns = returns_by_asset[target_asset].dropna()
                        # Short인 경우 음수로 변환 (매도 포지션의 수익)
                        if direction == 'short':
                            asset_returns = -asset_returns
                        rolling_3m = asset_returns.rolling(window=63).sum().dropna()
                        if not rolling_3m.empty and len(rolling_3m) >= 20:
                            mean_ret = rolling_3m.mean()
                            std_ret = rolling_3m.std()
                            effective_leg_factor = 1  # 단일 자산 = 1 leg
                            pos_type_label = f"{direction.capitalize()}: {target_asset}"
                        else:
                            continue
                    else:
                        continue
                else:
                    # Pair: 기존 방식 (Long - Short)
                    pair_3m_returns = calculate_pair_3m_rolling_returns(
                        returns_by_asset, long_asset, short_asset, signal,
                        lookback_years=3, rolling_window=63
                    )

                    if not pair_3m_returns.empty and len(pair_3m_returns) >= 20:
                        mean_ret = pair_3m_returns.mean()
                        std_ret = pair_3m_returns.std()
                        effective_leg_factor = leg_factor
                        pos_type_label = 'Pair'
                    else:
                        continue

                for std_level in std_levels:
                    scenario_return = mean_ret + std_level * std_ret

                    # 현재 포지션 손익 (기존: Pair 기준 leg_factor 사용)
                    current_pnl = current_pos * scenario_return * leg_factor * 10000  # bp
                    # 조정 포지션 손익 (포지션 타입에 따른 effective_leg_factor 사용)
                    adj_pnl = adj_pos * scenario_return * effective_leg_factor * 10000  # bp

                    scenario_data.append({
                        'Pair': row['Pair'],
                        'Position_Type': pos_type_label,
                        'Scenario': f"{std_level}σ" if std_level != 0 else "Mean",
                        'Std_Level': std_level,
                        'Current_PnL_bp': current_pnl,
                        'Adjusted_PnL_bp': adj_pnl,
                        'Delta_PnL_bp': adj_pnl - current_pnl
                    })

            # 새 포지션의 시나리오 데이터도 추가
            new_position_scenario_data = []
            for new_pos in st.session_state.new_positions_tab11:
                asset = new_pos['asset']
                direction = new_pos['direction']
                size_bp = new_pos['size_bp'] / 10000.0  # bp → 소수

                if asset in returns_by_asset.columns:
                    asset_returns = returns_by_asset[asset].dropna()
                    if direction == 'short':
                        asset_returns = -asset_returns
                    rolling_3m = asset_returns.rolling(window=63).sum().dropna()
                    if not rolling_3m.empty and len(rolling_3m) >= 20:
                        mean_ret = rolling_3m.mean()
                        std_ret = rolling_3m.std()

                        for std_level in std_levels:
                            scenario_return = mean_ret + std_level * std_ret
                            new_pnl = size_bp * scenario_return * 10000  # bp

                            new_position_scenario_data.append({
                                'Pair': f"[NEW] {direction.capitalize()}: {asset}",
                                'Position_Type': f"{direction.capitalize()}: {asset}",
                                'Scenario': f"{std_level}σ" if std_level != 0 else "Mean",
                                'Std_Level': std_level,
                                'Current_PnL_bp': 0.0,  # 새 포지션은 현재 0
                                'Adjusted_PnL_bp': new_pnl,
                                'Delta_PnL_bp': new_pnl,
                                'Is_New': True
                            })

            # 기존 데이터에 Is_New 플래그 추가
            for item in scenario_data:
                item['Is_New'] = False

            # 합치기
            all_scenario_data = scenario_data + new_position_scenario_data

            if all_scenario_data:
                scenario_df = pd.DataFrame(all_scenario_data)

                # 포트폴리오 전체 손익 합계
                portfolio_pnl = scenario_df.groupby('Scenario').agg({
                    'Current_PnL_bp': 'sum',
                    'Adjusted_PnL_bp': 'sum',
                    'Std_Level': 'first'
                }).reset_index().sort_values('Std_Level')

                # 새 포지션만의 기여도 계산
                new_pos_pnl = scenario_df[scenario_df['Is_New'] == True].groupby('Scenario').agg({
                    'Adjusted_PnL_bp': 'sum',
                    'Std_Level': 'first'
                }).reset_index().sort_values('Std_Level')
                new_pos_pnl.rename(columns={'Adjusted_PnL_bp': 'New_Position_PnL_bp'}, inplace=True)

                # 병합
                if not new_pos_pnl.empty:
                    portfolio_pnl = portfolio_pnl.merge(new_pos_pnl[['Scenario', 'New_Position_PnL_bp']], on='Scenario', how='left')
                    portfolio_pnl['New_Position_PnL_bp'] = portfolio_pnl['New_Position_PnL_bp'].fillna(0)
                else:
                    portfolio_pnl['New_Position_PnL_bp'] = 0

                # 그래프 1: 전체 포트폴리오 손익 분포
                fig_pnl = go.Figure()

                # 현재 AP 포지션 P&L (기존)
                fig_pnl.add_trace(go.Bar(
                    name='기존 AP 포지션',
                    x=portfolio_pnl['Scenario'],
                    y=portfolio_pnl['Current_PnL_bp'],
                    marker_color='rgba(55, 128, 191, 0.8)',
                    text=portfolio_pnl['Current_PnL_bp'].apply(lambda x: f"{x:.2f}"),
                    textposition='outside'
                ))

                # 조정 후 전체 P&L
                fig_pnl.add_trace(go.Bar(
                    name='조정 후 전체',
                    x=portfolio_pnl['Scenario'],
                    y=portfolio_pnl['Adjusted_PnL_bp'],
                    marker_color='rgba(219, 64, 82, 0.8)',
                    text=portfolio_pnl['Adjusted_PnL_bp'].apply(lambda x: f"{x:.2f}"),
                    textposition='outside'
                ))

                fig_pnl.update_layout(
                    title="📊 포트폴리오 손익 분포: 기존 AP vs 조정 후",
                    xaxis_title="시나리오 (σ)",
                    yaxis_title="예상 손익 (bp)",
                    barmode='group',
                    height=500,
                    showlegend=True,
                    legend=dict(x=0.02, y=0.98)
                )

                fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                fig_pnl = apply_chart_font_settings(fig_pnl)

                st.plotly_chart(fig_pnl, use_container_width=True)

                # 새 포지션이 있으면 영향 분석 그래프 추가
                if st.session_state.new_positions_tab11 and portfolio_pnl['New_Position_PnL_bp'].abs().sum() > 0:
                    st.markdown("#### 🆕 새 포지션 추가 영향 분석")

                    # 스택 바 차트: 기존 AP + 새 포지션 = 조정 후
                    fig_stack = go.Figure()

                    # 기존 AP 조정 (새 포지션 제외)
                    existing_adjusted = portfolio_pnl['Adjusted_PnL_bp'] - portfolio_pnl['New_Position_PnL_bp']

                    fig_stack.add_trace(go.Bar(
                        name='기존 AP (조정)',
                        x=portfolio_pnl['Scenario'],
                        y=existing_adjusted,
                        marker_color='rgba(55, 128, 191, 0.8)',
                    ))

                    fig_stack.add_trace(go.Bar(
                        name='새 포지션 기여',
                        x=portfolio_pnl['Scenario'],
                        y=portfolio_pnl['New_Position_PnL_bp'],
                        marker_color='rgba(0, 204, 150, 0.8)',
                    ))

                    fig_stack.update_layout(
                        title="손익 구성: 기존 AP 조정 + 새 포지션 추가",
                        xaxis_title="시나리오 (σ)",
                        yaxis_title="예상 손익 (bp)",
                        barmode='stack',
                        height=450,
                        showlegend=True,
                        legend=dict(x=0.02, y=0.98)
                    )

                    fig_stack.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                    fig_stack = apply_chart_font_settings(fig_stack)

                    st.plotly_chart(fig_stack, use_container_width=True)

                # 시나리오별 상세 테이블
                with st.expander("📋 시나리오별 손익 상세 테이블", expanded=False):
                    pnl_table = portfolio_pnl.copy()
                    pnl_table['변화량 (bp)'] = pnl_table['Adjusted_PnL_bp'] - pnl_table['Current_PnL_bp']

                    display_pnl = pnl_table[['Scenario', 'Current_PnL_bp', 'Adjusted_PnL_bp', 'New_Position_PnL_bp', '변화량 (bp)']].copy()
                    display_pnl.columns = ['시나리오', '기존 AP (bp)', '조정 후 (bp)', '새 포지션 (bp)', '변화량 (bp)']

                    # 포맷팅
                    for col in ['기존 AP (bp)', '조정 후 (bp)', '새 포지션 (bp)', '변화량 (bp)']:
                        display_pnl[col] = display_pnl[col].apply(lambda x: f"{x:+.2f}" if x != 0 else "0.00")

                    st.dataframe(display_pnl, use_container_width=True, hide_index=True)

                # ===== 포지션 정보 및 리스크 비교 테이블 =====
                st.markdown("---")
                st.markdown("#### 📊 포지션 정보 및 리스크 비교")

                # 포지션 비교 데이터 생성
                position_comparison = []

                # 기존 포지션
                for idx, row in common_positions.iterrows():
                    pair_id = row['Pair_ID']
                    pair_name = row['Pair']
                    long_asset = row['Long_Asset']
                    short_asset = row['Short_Asset']
                    signal = float(row['Signal'])
                    leg_factor = int(row['Leg_Factor'])

                    # 현재 포지션
                    current_pos_bp = float(row['Per_Leg_Position_bp'])
                    current_notional_bp = current_pos_bp * leg_factor

                    # 조정 포지션
                    adj_pos_bp = adjusted_sizes.get(pair_id, current_pos_bp)
                    pos_info = position_types.get(pair_id, {'type': 'pair', 'asset': None, 'direction': None})

                    # 포지션 타입에 따른 effective leg factor
                    if pos_info['type'] == 'single':
                        effective_leg_factor = 1
                        pos_type_str = f"{pos_info['direction'].capitalize()}: {pos_info['asset']}"
                    else:
                        effective_leg_factor = leg_factor
                        pos_type_str = "Pair"

                    adj_notional_bp = adj_pos_bp * effective_leg_factor

                    # Risk Unit
                    risk_unit = float(row['Risk_Unit_3M_%'])

                    # 예상 리스크 (bp) - 근사치
                    # current_pos_bp는 이미 bp 단위이므로 * 10000 불필요
                    current_risk_bp = abs(current_pos_bp) * (risk_unit / 100.0) * leg_factor
                    adj_risk_bp = abs(adj_pos_bp) * (risk_unit / 100.0) * effective_leg_factor

                    position_comparison.append({
                        'Pair': pair_name,
                        'Signal': signal,
                        '포지션 타입': pos_type_str,
                        '현재 레그당 (bp)': current_pos_bp,
                        '조정 레그당 (bp)': adj_pos_bp,
                        '현재 총명목 (bp)': current_notional_bp,
                        '조정 총명목 (bp)': adj_notional_bp,
                        'Risk Unit (%)': risk_unit,
                        '현재 리스크 (bp)': current_risk_bp,
                        '조정 리스크 (bp)': adj_risk_bp,
                        '리스크 변화 (bp)': adj_risk_bp - current_risk_bp
                    })

                # 새 포지션 추가
                for new_pos in st.session_state.new_positions_tab11:
                    asset = new_pos['asset']
                    direction = new_pos['direction']
                    size_bp = new_pos['size_bp']
                    pos_type_str = f"{direction.capitalize()}: {asset}"

                    # 새 포지션은 현재 0, 조정에만 반영
                    position_comparison.append({
                        'Pair': f"[NEW] {asset}",
                        'Signal': 0.0,
                        '포지션 타입': pos_type_str,
                        '현재 레그당 (bp)': 0.0,
                        '조정 레그당 (bp)': size_bp,
                        '현재 총명목 (bp)': 0.0,
                        '조정 총명목 (bp)': size_bp,
                        'Risk Unit (%)': 5.0,  # 기본값
                        '현재 리스크 (bp)': 0.0,
                        '조정 리스크 (bp)': size_bp * 0.05,  # 근사치
                        '리스크 변화 (bp)': size_bp * 0.05
                    })

                position_comp_df = pd.DataFrame(position_comparison)

                # 포맷팅 함수
                def format_position_table(df):
                    return df.style.format({
                        'Signal': '{:.0f}',
                        '현재 레그당 (bp)': '{:.3f}',
                        '조정 레그당 (bp)': '{:.3f}',
                        '현재 총명목 (bp)': '{:.2f}',
                        '조정 총명목 (bp)': '{:.2f}',
                        'Risk Unit (%)': '{:.2f}',
                        '현재 리스크 (bp)': '{:.2f}',
                        '조정 리스크 (bp)': '{:.2f}',
                        '리스크 변화 (bp)': '{:+.2f}'
                    }).background_gradient(
                        subset=['리스크 변화 (bp)'],
                        cmap='RdYlGn_r',
                        vmin=-position_comp_df['리스크 변화 (bp)'].abs().max(),
                        vmax=position_comp_df['리스크 변화 (bp)'].abs().max()
                    )

                st.dataframe(format_position_table(position_comp_df), use_container_width=True, hide_index=True)

                # 전체 포트폴리오 요약
                st.markdown("##### 💼 전체 포트폴리오 요약")
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)

                total_current_notional = position_comp_df['현재 총명목 (bp)'].abs().sum()
                total_adj_notional = position_comp_df['조정 총명목 (bp)'].abs().sum()

                # TE 계산 (공분산 행렬 사용)
                try:
                    if not returns_by_asset.empty and not w_opt_daily.empty and not w_bmk_daily.empty:
                        # 최신 가중치
                        asof = min(w_opt_daily.index.max(), w_bmk_daily.index.max())
                        Wopt_last = w_opt_daily.loc[asof].fillna(0.0)
                        Wbmk_last = w_bmk_daily.loc[asof].fillna(0.0)

                        # 공분산 행렬
                        cols = [c for c in returns_by_asset.columns if c in Wopt_last.index]
                        R = returns_by_asset[cols]
                        R_dec = _pc_ensure_decimal_returns(R)
                        C = _pc_build_recent_cov_constant_corr(R_dec, window=63, rho=0.25)

                        w_b = Wbmk_last.reindex(cols).fillna(0.0)

                        # 현재 active weights 계산
                        w_active_current = pd.Series(0.0, index=cols)
                        for i, row in enumerate(common_positions.itertuples()):
                            pid = row.Pair_ID
                            long_asset = str(row.Long_Asset)
                            short_asset = str(row.Short_Asset)
                            pos_bp = row.Per_Leg_Position_bp / 10000.0  # bp → 소수

                            # Pair 기준
                            if long_asset in w_active_current.index:
                                w_active_current[long_asset] += pos_bp
                            if short_asset in w_active_current.index:
                                w_active_current[short_asset] -= pos_bp

                        # 조정 active weights 계산
                        w_active_adj = pd.Series(0.0, index=cols)

                        # 기존 포지션 반영
                        for i, row in enumerate(common_positions.itertuples()):
                            pid = row.Pair_ID
                            long_asset = str(row.Long_Asset)
                            short_asset = str(row.Short_Asset)

                            if pid in adjusted_sizes:
                                pos_bp = adjusted_sizes[pid] / 10000.0  # bp → 소수
                            else:
                                pos_bp = row.Per_Leg_Position_bp / 10000.0

                            pos_info = position_types.get(pid, {'type': 'pair', 'asset': None, 'direction': None})

                            if pos_info['type'] == 'single':
                                target_asset = pos_info['asset']
                                direction = pos_info['direction']
                                if target_asset in w_active_adj.index:
                                    if direction == 'long':
                                        w_active_adj[target_asset] += pos_bp
                                    else:
                                        w_active_adj[target_asset] -= pos_bp
                            else:
                                if long_asset in w_active_adj.index:
                                    w_active_adj[long_asset] += pos_bp
                                if short_asset in w_active_adj.index:
                                    w_active_adj[short_asset] -= pos_bp

                        # 새 포지션 반영
                        for new_pos in st.session_state.new_positions_tab11:
                            asset = new_pos['asset']
                            direction = new_pos['direction']
                            size_bp = new_pos['size_bp'] / 10000.0  # bp → 소수

                            if asset in w_active_adj.index:
                                if direction == 'long':
                                    w_active_adj[asset] += size_bp
                                else:  # short
                                    w_active_adj[asset] -= size_bp

                        # TE 계산
                        te_bp_here = _pc_te_bp_from_cov(w_active_current.values, C, 252)
                        adj_te_bp_here = _pc_te_bp_from_cov(w_active_adj.values, C, 252)
                    else:
                        # 공분산 데이터 없으면 근사치 사용
                        te_bp_here = position_comp_df['현재 리스크 (bp)'].sum()
                        adj_te_bp_here = position_comp_df['조정 리스크 (bp)'].sum()
                except Exception as e:
                    st.warning(f"TE 계산 중 오류: {e}")
                    te_bp_here = position_comp_df['현재 리스크 (bp)'].sum()
                    adj_te_bp_here = position_comp_df['조정 리스크 (bp)'].sum()

                with col_sum1:
                    st.metric(
                        "현재 총 명목",
                        f"{total_current_notional:.2f}bp"
                    )
                with col_sum2:
                    st.metric(
                        "조정 후 총 명목",
                        f"{total_adj_notional:.2f}bp",
                        delta=f"{total_adj_notional - total_current_notional:+.2f}bp"
                    )
                with col_sum3:
                    st.metric(
                        "현재 TE",
                        f"{te_bp_here:.2f}bp"
                    )
                with col_sum4:
                    st.metric(
                        "조정 후 TE",
                        f"{adj_te_bp_here:.2f}bp",
                        delta=f"{adj_te_bp_here - te_bp_here:+.2f}bp"
                    )

                st.markdown("---")

                # 페어별 상세 테이블
                with st.expander("📋 페어별 시나리오 상세", expanded=False):
                    # 포지션 타입 요약
                    if 'Position_Type' in scenario_df.columns:
                        type_summary = scenario_df.groupby('Pair')['Position_Type'].first().reset_index()
                        st.markdown("**포지션 타입 현황**")
                        type_counts = type_summary['Position_Type'].value_counts()
                        type_info = ", ".join([f"{t}: {c}개" for t, c in type_counts.items()])
                        st.caption(f"💡 {type_info}")

                        # 포지션 타입별 색상 표시
                        for _, type_row in type_summary.iterrows():
                            if type_row['Position_Type'] == 'Long Only':
                                icon = "📈"
                            elif type_row['Position_Type'] == 'Short Only':
                                icon = "📉"
                            else:
                                icon = "⚖️"

                    # 피벗 테이블 생성
                    pivot_current = scenario_df.pivot(
                        index='Pair', columns='Scenario', values='Current_PnL_bp'
                    )
                    pivot_adjusted = scenario_df.pivot(
                        index='Pair', columns='Scenario', values='Adjusted_PnL_bp'
                    )

                    st.markdown("**현재 포지션 손익 (bp)** - Pair 기준")
                    st.dataframe(
                        pivot_current.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True
                    )

                    st.markdown("**조정 포지션 손익 (bp)** - 선택한 포지션 타입 기준")
                    st.dataframe(
                        pivot_adjusted.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True
                    )

                # 리스크 요약 통계
                st.markdown("#### 📊 리스크 요약")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)

                current_worst = portfolio_pnl[portfolio_pnl['Std_Level'] == -3]['Current_PnL_bp'].values[0]
                adjusted_worst = portfolio_pnl[portfolio_pnl['Std_Level'] == -3]['Adjusted_PnL_bp'].values[0]
                current_best = portfolio_pnl[portfolio_pnl['Std_Level'] == 3]['Current_PnL_bp'].values[0]
                adjusted_best = portfolio_pnl[portfolio_pnl['Std_Level'] == 3]['Adjusted_PnL_bp'].values[0]

                with col_s1:
                    st.metric(
                        "현재 최악 (-3σ)",
                        f"{current_worst:.2f}bp"
                    )

                with col_s2:
                    st.metric(
                        "조정 후 최악 (-3σ)",
                        f"{adjusted_worst:.2f}bp",
                        delta=f"{adjusted_worst - current_worst:+.2f}bp"
                    )

                with col_s3:
                    st.metric(
                        "현재 최선 (+3σ)",
                        f"{current_best:.2f}bp"
                    )

                with col_s4:
                    st.metric(
                        "조정 후 최선 (+3σ)",
                        f"{adjusted_best:.2f}bp",
                        delta=f"{adjusted_best - current_best:+.2f}bp"
                    )
            else:
                st.warning("시나리오 손익 계산에 필요한 데이터가 부족합니다.")
        else:
            st.info("포지션 데이터가 없습니다. actual_portfolio_positions.csv 파일을 확인해주세요.")

        # ===== 데이터 다운로드 =====
        st.markdown("---")
        st.subheader("📥 데이터 다운로드")

        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            # 성과 테이블 다운로드
            csv_performance = performance_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 기간별 성과 다운로드 (CSV)",
                data=csv_performance,
                file_name=f"performance_by_period_{selected_start.strftime('%Y%m%d')}_{selected_end.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_performance"
            )

        with col_d2:
            # 일별 수익률 다운로드 (선택 기간)
            csv_returns = filtered_returns.to_csv().encode('utf-8-sig')
            st.download_button(
                label="📥 일별 수익률 다운로드 (CSV)",
                data=csv_returns,
                file_name=f"daily_returns_{selected_start.strftime('%Y%m%d')}_{selected_end.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_returns"
            )

        with col_d3:
            # 월별 수익률 다운로드
            if len(monthly_returns) > 0:
                csv_monthly = monthly_returns_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 월별 수익률 다운로드 (CSV)",
                    data=csv_monthly,
                    file_name=f"monthly_returns_{selected_start.strftime('%Y%m%d')}_{selected_end.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_monthly"
                )

        st.success(f"✅ 실제 포트폴리오 성과 분석 완료 ({selected_start.strftime('%Y-%m-%d')} ~ {selected_end.strftime('%Y-%m-%d')})")


if __name__ == "__main__":
    main()
