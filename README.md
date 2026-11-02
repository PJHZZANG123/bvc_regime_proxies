# bvc_regime_proxies

주문흐름 프록시는 국면을 재현하는가, 시기를 구분하는가: 암호화폐 실제 체결 방향을 이용한 평가 잣대의 검증

Regime Reproduction or Period Separation? Evaluating Order-Flow Proxies against Actual Trade Direction in Crypto Markets

이 저장소는 위 논문의 온라인 부록, 분석 코드, 파생 자료를 담는다. 저자 정보와 서지 사항은 게재가 확정된 뒤에 적는다.

## 구성

| 경로 | 내용 |
|---|---|
| `appendix/online_appendix.pdf` | 온라인 부록 A–H: 관측값 변환 비교, 강건성 전체 결과, 블록 길이별 구간, 추정·검정 설정, 연도별 일치도, 종가 위치 프록시(CLV) 결과, 자료 구축의 세부, 국면 수 선택 |
| `src/` | 수집, 봉 생성, 변수 생성, 은닉마르코프모형 추정, 국면 일치도 평가, 검정, 강건성 분석 코드 (주석은 한국어) |
| `data/bars/` | 4시간봉·일봉·1시간봉·달러봉의 OHLCV, 거래대금, CVD (BTCUSDT 무기한선물; ETHUSDT 는 4시간봉·일봉) |
| `data/features/` | 모형 입력 변수: 로그수익률, CVD 흐름, BVC 흐름, CLV 흐름, CLV 잔차, 롤링 표준편차 등 |
| `results/` | 표에 쓰인 결과(JSON), 실행 로그, 부트스트랩·순환 이동 검정 결과, 추정된 상태열(npz) |
| `figures/` | 논문과 부록의 그림 |
| `requirements.txt` | 고정한 패키지 버전 (Python 3.12.14, hmmlearn 0.3.3, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1) |

## 원자료

Binance 공개 저장소(data.binance.vision)의 BTCUSDT·ETHUSDT 무기한선물 집계 체결(aggTrades) 일별 파일(2021-01 ~ 2025-12)과 1분봉 klines 를 쓴다.
용량이 약 400 GB 라 저장소에 넣지 않았고, `src/fetch_all_daily.py` 로 내려받는다.
집계 번호의 결손 6곳(BTCUSDT 611,488개, 전체의 0.0218%)은 채우거나 지우지 않고 해당 봉에 표시만 달았다(부록 G).

## 재현 순서

1. 환경: `uv venv --python 3.12 .venv` 뒤 `uv pip install -r requirements.txt`
2. 수집과 조립 (원자료 필요): `fetch_all_daily.py` → `build_parquet_from_daily.py` → `audit_trades.py`, `scan_all_gaps.py`
3. 봉과 변수 (원자료 필요): `build_bars.py` → `verify_bars.py` → `build_features.py` (1시간봉·달러봉은 `build_bars_alt.py` → `run_alt_bars.py`)
4. 국면 모형: `select_k.py` → `fit_models.py` → `evaluate_agreement.py`
5. 검정과 강건성: `bootstrap_ci.py`, `circular_shift_null.py`, `time_structure.py`, `robustness_core.py`, `robustness_period.py`, `controls_extended.py`, `controls_extended_ci.py`, `eval_alt_k.py`
6. ETHUSDT 재현 검증: `eth_stage1.py` → `eth_stage2.py`
7. 표와 그림: `describe_sample.py`, `flow_correlations.py`, `bar_accuracy.py`, `chapter4_tables.py`, `appendix_tables.py`, `revision_tables.py`, `paper_figures.py`

4단계부터는 `data/features/` 의 파일만으로 돌릴 수 있다. 모든 스크립트는 저장소 최상위에서 `python src/<이름>.py` 로 실행한다.
hmmlearn 은 버전에 따라 EM 수렴 결과가 달라지므로 `requirements.txt` 의 버전을 그대로 쓴다.

## 라이선스

코드(`src/`)는 MIT 라이선스(`LICENSE`), 부록 PDF 와 파생 자료(`data/`, `results/`, `figures/`)는 CC BY 4.0 이다.
원자료의 권리는 Binance 에 있다.
