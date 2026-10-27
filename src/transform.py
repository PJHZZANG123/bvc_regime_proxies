"""
관측 벡터 변환 — 다섯 모형에 동일하게 적용한다.

배경 (E020a)
  원 관측(로그수익률, 흐름)은 초과첨도가 +8.7 ~ +13.3 으로 매우 두껍다.
  가우시안 HMM 을 그대로 적합하면 상태가 '시간적 국면'이 아니라
  '주변분포의 꼬리'를 쪼개는 데 쓰인다. 실제로 K=2 적합 시 두 상태의 평균이
  모두 ~0 이고 분산만 달랐으며(1.61 vs 0.43), 지속확률이 0.62 로 떨어져
  사전 기준(0.90)을 어떤 K 도 통과하지 못했다.

채택 (E020b, 2026-09-15 — Δ 산출 이전)
  부호보존 로그:  y = sign(x) · log1p(|x| / MAD),  이후 표준화
  꼬리만 눌러 초과첨도를 ~0 으로 낮추면서 크기의 상대적 순서와 비율을 보존한다.
  순위→정규 변환도 기준을 통과했으나 크기 정보를 전부 버리므로 채택하지 않았다.

MAD 는 전체 표본에서 한 번 계산한다(롤링 아님). 표준화와 같은 성격의
전역 스케일 조정이며, 시점별 정보를 쓰지 않으므로 선견 편의는 없다.
"""

import numpy as np

METHOD = "signed_log"


def _mad(x: np.ndarray) -> float:
    s = float(np.median(np.abs(x - np.median(x))))
    return s if s > 0 else float(np.std(x))


def signed_log(X: np.ndarray) -> np.ndarray:
    """열마다 sign(x)·log1p(|x|/MAD) 적용 후 표준화."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        x = X[:, j]
        y = np.sign(x) * np.log1p(np.abs(x) / _mad(x))
        out[:, j] = (y - y.mean()) / y.std()
    return out


def rank_normal(X: np.ndarray) -> np.ndarray:
    """대안(미채택, E063 메모). 순위를 정규분위수로 사상."""
    from scipy.stats import norm, rankdata
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(X)
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / n) for j in range(X.shape[1])])


def zscore(X: np.ndarray) -> np.ndarray:
    """변환 없이 표준화만(로버스트니스 비교용)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return (X - X.mean(0)) / X.std(0)


TRANSFORMS = {"signed_log": signed_log, "rank_normal": rank_normal, "zscore": zscore}


def apply(X, method: str = METHOD) -> np.ndarray:
    return TRANSFORMS[method](X)
