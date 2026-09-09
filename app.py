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
#    - 요구사항: movies 표에 있는 영화를 모두 사용
#    - daily 데이터는 영화별 요약 통계(등장 일수, 최대 스크린수 등)를 만들어 병합에 활용
# ----------------------------------------------------
daily_summary = daily_df.groupby("영화코드").agg(
    daily_days_appeared=("날짜", "nunique"),
    daily_max_screen=("스크린수", "max"),
    daily_max_show=("상영횟수", "max"),
    daily_max_audi=("일관객", "max"),
).reset_index().rename(columns={"영화코드": "movieCd"})

# movies_df 를 기준(모두 사용)으로 left join
merged = movies_df.merge(daily_summary, on="movieCd", how="left")

# 결측치(daily에 없는 영화)는 0으로 채움
for col in ["daily_days_appeared", "daily_max_screen", "daily_max_show", "daily_max_audi"]:
    merged[col] = merged[col].fillna(0)

# ----------------------------------------------------
# 3. 영화코드 순으로 정렬 후, 열 편마다 앞 세 편을 테스트용으로 분리
# ----------------------------------------------------
merged = merged.sort_values("movieCd").reset_index(drop=True)

n_total = len(merged)
group_size = 10  # "열 편마다"
merged["group_idx"] = np.arange(n_total) // group_size
merged["rank_in_group"] = np.arange(n_total) % group_size

is_test = merged["rank_in_group"] < 3
test_df = merged[is_test].copy()
train_df = merged[~is_test].copy()

# ----------------------------------------------------
# 4. 사용할 변수(특성) 선택 - 체크박스
# ----------------------------------------------------
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
    default_checked = col in ["first_scrn", "first_show", "first_week_audi", "peak"]
    if st.sidebar.checkbox(label, value=default_checked):
        selected_features.append(col)

if len(selected_features) == 0:
    st.warning("⚠️ 왼쪽에서 모델에 사용할 변수를 최소 1개 이상 선택해주세요.")
    st.stop()

target_col = "total_audi"

# 선택한 변수 + 타깃 열에 결측치가 있는 행 제거
use_cols = selected_features + [target_col]
train_clean = train_df.dropna(subset=use_cols).copy()
test_clean = test_df.dropna(subset=use_cols).copy()

if len(train_clean) < 2 or len(test_clean) < 1:
    st.error("선택한 변수 조합으로는 학습/평가에 사용할 데이터가 부족합니다. 다른 변수를 선택해보세요.")
    st.stop()

X_train = train_clean[selected_features]
y_train = train_clean[target_col]
X_test = test_clean[selected_features]
y_test = test_clean[target_col]

# ----------------------------------------------------
# 5. 다중 회귀 모델 학습
# ----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# 관객수는 음수가 될 수 없으므로 0 이하 예측은 0으로 보정(로그 표시를 위해 아주 작은 값 처리는 아래에서)
y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# ----------------------------------------------------
# 6. 기준 정보 출력 (학습/평가 편수, 기준 기간)
# ----------------------------------------------------
st.subheader("📌 모델 학습 개요")

# 기준 기간: daily 데이터의 날짜 범위 (여덟 자리 숫자 -> 날짜로 변환)
daily_dates = pd.to_datetime(daily_df["날짜"].astype(str), format="%Y%m%d")
period_start = daily_dates.min().strftime("%Y-%m-%d")
period_end = daily_dates.max().strftime("%Y-%m-%d")

col1, col2, col3 = st.columns(3)
col1.metric("학습에 사용한 영화 편수", f"{len(train_clean)} 편")
col2.metric("평가(테스트)에 사용한 영화 편수", f"{len(test_clean)} 편")
col3.metric("기준 기간", f"{period_start} ~ {period_end}")

st.write(f"**사용한 변수:** {', '.join(selected_features)}")

col_a, col_b = st.columns(2)
col_a.metric("R² (결정계수)", f"{r2:.4f}")
col_b.metric("MAE (평균 절대 오차)", f"{mae:,.0f} 명")

# ----------------------------------------------------
# 7. 1,000명 미만 예측 처리 (그래프 바닥에 붙여 표시)
# ----------------------------------------------------
FLOOR_VALUE = 1000  # 로그축 하한 기준값

below_floor_mask = y_pred_clipped < FLOOR_VALUE
n_below_floor = int(below_floor_mask.sum())

# 그래프에 표시할 예측값: 1,000명 미만은 바닥값(1,000)으로 고정 표시
y_pred_display = np.where(y_pred_clipped < FLOOR_VALUE, FLOOR_VALUE, y_pred_clipped)
# 로그 스케일이므로 0은 그릴 수 없어 실제값도 최소 1로 보정
x_actual_display = np.where(y_test.values < 1, 1, y_test.values)

st.write(f"🔻 **예측이 {FLOOR_VALUE:,}명보다 작게 나온 영화 수:** {n_below_floor} 편 (그래프 바닥에 표시됨)")

# ----------------------------------------------------
# 8. Plotly 산점도 (로그-로그 축, 기준선 포함)
# ----------------------------------------------------
st.subheader("📊 실제 총 관객 수 vs 예측 총 관객 수 (테스트 영화)")

fig = go.Figure()

# 기준선 (예측 = 실제)
axis_min = max(1, min(x_actual_display.min(), y_pred_display.min()) * 0.5)
axis_max = max(x_actual_display.max(), y_pred_display.max()) * 2
line_vals = [axis_min, axis_max]

fig.add_trace(go.Scatter(
    x=line_vals, y=line_vals,
    mode="lines",
    name="예측 = 실제 (기준선)",
    line=dict(color="gray", dash="dash")
))

# 일반 예측 포인트 (1,000명 이상)
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

# 바닥에 붙인 포인트 (1,000명 미만 예측)
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
