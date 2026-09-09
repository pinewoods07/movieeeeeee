import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("박스오피스 일별 데이터와 영화 정보를 결합해 총 관객 수를 예측하는 다중 회귀 모델입니다.")

# ----------------------------------------------------
# ⚠️ 사후 집계값 사용에 대한 안내
# ----------------------------------------------------
st.warning(
    "⚠️ **주의**: 이 앱에서 사용하는 변수(첫 주 관객, 톱10 유지일수, 일별 최대 스크린수 등)는 "
    "영화가 **개봉하고 시간이 지난 뒤에 집계된 값(사후 집계값)**입니다. "
    "따라서 여기서 나오는 예측 점수는 **개봉 전에 실제로 흥행을 예측하는 성능이 아니라**, "
    "'개봉 후 일정 기간이 지난 시점에 총 관객 수를 얼마나 잘 설명할 수 있는가'를 보여주는 것입니다."
)

# ----------------------------------------------------
# 1. 데이터 불러오기
# ----------------------------------------------------
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies

daily_df, movies_df = load_data()

# ----------------------------------------------------
# 2. 두 표를 영화코드(movieCd) 기준으로 결합
#    - movies 표에 있는 영화를 모두 사용
# ----------------------------------------------------
daily_summary = daily_df.groupby("영화코드").agg(
    daily_days_appeared=("날짜", "nunique"),
    daily_max_screen=("스크린수", "max"),
    daily_max_show=("상영횟수", "max"),
    daily_max_audi=("일관객", "max"),
).reset_index().rename(columns={"영화코드": "movieCd"})

merged = movies_df.merge(daily_summary, on="movieCd", how="left")

for col in ["daily_days_appeared", "daily_max_screen", "daily_max_show", "daily_max_audi"]:
    merged[col] = merged[col].fillna(0)

# ----------------------------------------------------
# 3. 영화코드 순 정렬 후, 열 편마다 앞 세 편을 테스트용으로 분리
# ----------------------------------------------------
merged = merged.sort_values("movieCd").reset_index(drop=True)

n_total = len(merged)
group_size = 10
merged["group_idx"] = np.arange(n_total) // group_size
merged["rank_in_group"] = np.arange(n_total) % group_size

is_test = merged["rank_in_group"] < 3
test_df = merged[is_test].copy()
train_df = merged[~is_test].copy()

target_col = "total_audi"

# 기준 기간
daily_dates = pd.to_datetime(daily_df["날짜"].astype(str), format="%Y%m%d")
period_start = daily_dates.min().strftime("%Y-%m-%d")
period_end = daily_dates.max().strftime("%Y-%m-%d")

# ----------------------------------------------------
# 공통 함수: 주어진 변수 목록으로 학습·평가
# ----------------------------------------------------
def fit_and_evaluate(feature_cols):
    use_cols = feature_cols + [target_col]
    train_clean = train_df.dropna(subset=use_cols).copy()
    test_clean = test_df.dropna(subset=use_cols).copy()

    if len(train_clean) < 2 or len(test_clean) < 1:
        return None

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]
    X_test = test_clean[feature_cols]
    y_test = test_clean[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "model": model,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_clipped": y_pred_clipped,
        "r2": r2,
        "mae": mae,
        "n_train": len(train_clean),
        "n_test": len(test_clean),
    }

# ----------------------------------------------------
# 4. [비교] 기본 변수 3개 vs 첫 주 관객 수 추가
# ----------------------------------------------------
st.subheader("📊 변수 조합별 예측 점수 비교")

BASIC_FEATURES = ["first_scrn", "first_show", "peak"]
EXTENDED_FEATURES = BASIC_FEATURES + ["first_week_audi"]

result_basic = fit_and_evaluate(BASIC_FEATURES)
result_extended = fit_and_evaluate(EXTENDED_FEATURES)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🅰️ 기본 변수 3개")
    st.caption(", ".join(BASIC_FEATURES) + " (first_scrn, first_show, peak)")
    if result_basic:
        st.metric("R² (결정계수)", f"{result_basic['r2']:.4f}")
        st.metric("MAE (평균 절대 오차)", f"{result_basic['mae']:,.0f} 명")
        st.caption(f"학습 {result_basic['n_train']}편 / 평가 {result_basic['n_test']}편")
    else:
        st.error("데이터 부족으로 계산 불가")

with col2:
    st.markdown("### 🅱️ 기본 변수 + 첫 주 관객 수")
    st.caption(", ".join(EXTENDED_FEATURES))
    if result_extended:
        st.metric("R² (결정계수)", f"{result_extended['r2']:.4f}")
        st.metric("MAE (평균 절대 오차)", f"{result_extended['mae']:,.0f} 명")
        st.caption(f"학습 {result_extended['n_train']}편 / 평가 {result_extended['n_test']}편")
    else:
        st.error("데이터 부족으로 계산 불가")

if result_basic and result_extended:
    diff_r2 = result_extended["r2"] - result_basic["r2"]
    st.info(
        f"💡 첫 주 관객 수를 추가하면 R²가 **{diff_r2:+.4f}** 변화했습니다. "
        "다만 첫 주 관객 수 역시 개봉 후에 집계되는 값이므로, 이 비교는 개봉 **전** 예측력이 아니라 "
        "'조기 흥행 지표를 추가로 알고 있을 때 설명력이 얼마나 느는지'를 보여주는 참고 자료입니다."
    )

st.divider()

# ----------------------------------------------------
# 5. 사용자 지정 변수 선택 (체크박스) - 자유 탐색용
# ----------------------------------------------------
st.subheader("🔧 직접 변수를 선택해 모델 만들어보기")

st.sidebar.header("🔧 모델에 사용할 변수 선택")

feature_options = {
    "first_scrn (첫 관측일 스크린수)": "first_scrn",
    "first_show (첫 관측일 상영횟수)": "first_show",
    "peak (성수기 개봉 여부)": "peak",
    "first_week_audi (첫 주 관객)": "first_week_audi",
    "days_in_top10 (톱10 유지일수)": "days_in_top10",
    "daily_days_appeared (일별 데이터 등장 일수)": "daily_days_appeared",
    "daily_max_screen (일별 최대 스크린수)": "daily_max_screen",
    "daily_max_show (일별 최대 상영횟수)": "daily_max_show",
    "daily_max_audi (일별 최대 일일관객)": "daily_max_audi",
}

selected_features = []
for label, col in feature_options.items():
    default_checked = col in BASIC_FEATURES
    if st.sidebar.checkbox(label, value=default_checked):
        selected_features.append(col)

if len(selected_features) == 0:
    st.warning("⚠️ 왼쪽에서 모델에 사용할 변수를 최소 1개 이상 선택해주세요.")
    st.stop()

result_custom = fit_and_evaluate(selected_features)

if result_custom is None:
    st.error("선택한 변수 조합으로는 학습/평가에 사용할 데이터가 부족합니다. 다른 변수를 선택해보세요.")
    st.stop()

test_clean = result_custom["test_clean"]
train_clean = result_custom["train_clean"]
y_test = result_custom["y_test"]
y_pred = result_custom["y_pred"]
y_pred_clipped = result_custom["y_pred_clipped"]
r2 = result_custom["r2"]
mae = result_custom["mae"]

# ----------------------------------------------------
# 6. 기준 정보 출력
# ----------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("학습에 사용한 영화 편수", f"{result_custom['n_train']} 편")
col2.metric("평가(테스트)에 사용한 영화 편수", f"{result_custom['n_test']} 편")
col3.metric("기준 기간", f"{period_start} ~ {period_end}")

st.write(f"**사용한 변수:** {', '.join(selected_features)}")

col_a, col_b = st.columns(2)
col_a.metric("R² (결정계수)", f"{r2:.4f}")
col_b.metric("MAE (평균 절대 오차)", f"{mae:,.0f} 명")

# ----------------------------------------------------
# 7. 1,000명 미만 예측 처리
# ----------------------------------------------------
FLOOR_VALUE = 1000

below_floor_mask = y_pred_clipped < FLOOR_VALUE
n_below_floor = int(below_floor_mask.sum())

y_pred_display = np.where(y_pred_clipped < FLOOR_VALUE, FLOOR_VALUE, y_pred_clipped)
x_actual_display = np.where(y_test.values < 1, 1, y_test.values)

st.write(f"🔻 **예측이 {FLOOR_VALUE:,}명보다 작게 나온 영화 수:** {n_below_floor} 편 (그래프 바닥에 표시됨)")

# ----------------------------------------------------
# 8. Plotly 산점도
# ----------------------------------------------------
st.subheader("📈 실제 총 관객 수 vs 예측 총 관객 수 (테스트 영화)")

fig = go.Figure()

axis_min = max(1, min(x_actual_display.min(), y_pred_display.min()) * 0.5)
axis_max = max(x_actual_display.max(), y_pred_display.max()) * 2
line_vals = [axis_min, axis_max]

fig.add_trace(go.Scatter(
    x=line_vals, y=line_vals,
    mode="lines",
    name="예측 = 실제 (기준선)",
    line=dict(color="gray", dash="dash")
))

normal_mask = ~below_floor_mask
fig.add_trace(go.Scatter(
    x=x_actual_display[normal_mask],
    y=y_pred_display[normal_mask],
    mode="markers",
    name="예측 ≥ 1,000명",
    text=test_clean["movieNm"].values[normal_mask],
    hovertemplate="영화: %{text}<br>실제: %{x:,.0f}명<br>예측: %{y:,.0f}명<extra></extra>",
    marker=dict(size=9, color="royalblue", opacity=0.75)
))

if n_below_floor > 0:
    fig.add_trace(go.Scatter(
        x=x_actual_display[below_floor_mask],
        y=y_pred_display[below_floor_mask],
        mode="markers",
        name=f"예측 < 1,000명 ({n_below_floor}편, 바닥 고정)",
        text=test_clean["movieNm"].values[below_floor_mask],
        hovertemplate="영화: %{text}<br>실제: %{x:,.0f}명<br>예측(바닥표시): %{y:,.0f}명<extra></extra>",
        marker=dict(size=9, color="crimson", symbol="triangle-down", opacity=0.85)
    ))

fig.update_xaxes(type="log", title="실제 총 관객 수 (로그 스케일)")
fig.update_yaxes(type="log", title="예측 총 관객 수 (로그 스케일)")
fig.update_layout(
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 9. 참고용: 테스트 데이터 상세 표
# ----------------------------------------------------
with st.expander("🔍 테스트 영화 상세 결과 보기"):
    result_table = test_clean[["movieCd", "movieNm", target_col]].copy()
    result_table["예측_총관객수"] = y_pred_clipped
    result_table["오차(예측-실제)"] = result_table["예측_총관객수"] - result_table[target_col]
    result_table = result_table.rename(columns={target_col: "실제_총관객수"})
    st.dataframe(result_table.reset_index(drop=True))
