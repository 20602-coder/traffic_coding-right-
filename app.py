import pickle
import numpy as np
import pandas as pd
import streamlit as st
from Orange.data import Table, Domain

# 페이지 기본 설정
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측기",
    page_icon="🚦",
    layout="wide"
)

# 모델 로드 (캐싱 적용)
@st.cache_resource
def load_orange_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# 예측 함수 (원본 도메인 및 전처리 파이프라인 자동 호환)
def predict_fatalities(model, sido: str, accident_cnt: float, minor_injury_cnt: float) -> float:
    # 1. 원본 입력 피처 도메인 가져오기 (원핫 인코딩 전 원본 구조)
    base_domain = getattr(model, "original_domain", None)
    if base_domain is None and getattr(model, "instances", None) is not None:
        base_domain = model.instances.domain
    if base_domain is None:
        base_domain = model.domain

    orig_attrs = base_domain.attributes

    # 2. 학습 데이터(instances)에서 기본값(평균/최빈값) 추출
    default_values = {}
    if getattr(model, "instances", None) is not None and len(model.instances) > 0:
        inst_attrs = model.instances.domain.attributes
        df_orig = pd.DataFrame(model.instances.X, columns=[attr.name for attr in inst_attrs])
        for attr in inst_attrs:
            if attr.is_continuous:
                default_values[attr.name] = float(df_orig[attr.name].mean())
            elif attr.is_discrete:
                mode_idx = int(df_orig[attr.name].mode()[0])
                default_values[attr.name] = attr.values[mode_idx]

    # 3. 사용자 입력값을 원본 피처 순서에 맞게 매핑
    row = []
    for attr in orig_attrs:
        name = attr.name
        if "시도" in name:
            row.append(sido)
        elif "사고" in name:
            row.append(float(accident_cnt))
        elif "경상" in name:
            row.append(float(minor_injury_cnt))
        else:
            # 중상자수, 부상신고자수 등 나머지 변수는 학습 데이터 평균값 사용
            fallback = 0.0 if attr.is_continuous else attr.values[0]
            row.append(default_values.get(name, fallback))

    # 4. 입력용 Table 생성 후 모델 예측 실행 (Orange3가 내부적으로 원핫 인코딩/정규화 자동 처리)
    input_domain = Domain(orig_attrs)
    input_table = Table.from_list(input_domain, [row])

    pred = model(input_table)[0]
    return max(0.0, float(pred))


# --- UI 구성 ---
st.title("🚦 교통사고 피해 예측 및 어린이 보호구역 안전 신호등")
st.markdown("시도별 사고건수와 경상자수를 조절하여 **예상 사망자 수**와 **위험 단계**를 실시간으로 확인하세요.")
st.divider()

MODEL_FILE = "traffic_casualty_model.pkcls"
try:
    model = load_orange_model(MODEL_FILE)
except FileNotFoundError:
    st.error(f"❌ `{MODEL_FILE}` 파일을 찾을 수 없습니다. `app.py`와 같은 폴더에 업로드해주세요.")
    st.stop()
except Exception as e:
    st.error(f"❌ 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 원본 도메인에서 시도 목록 추출
base_domain = getattr(model, "original_domain", None) or (
    model.instances.domain if getattr(model, "instances", None) is not None else model.domain
)

sido_options = ["서울", "경기", "대구"]
for attr in base_domain.attributes:
    if attr.is_discrete and "시도" in attr.name:
        sido_options = list(attr.values)
        break

# 화면 2단 분할
col_input, col_result = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("🎛️ 지역 및 사고 데이터 입력")
    selected_sido = st.selectbox(
        "시도 선택",
        options=sido_options,
        index=0,
        help="분석할 시·도 지역을 선택하세요."
    )

    accident_count = st.slider(
        "사고건수 (건)",
        min_value=0,
        max_value=500,
        value=50,
        step=1,
        help="해당 지역의 교통사고 발생 건수를 조절하세요."
    )

    minor_injury_count = st.slider(
        "경상자수 (명)",
        min_value=0,
        max_value=500,
        value=40,
        step=1,
        help="사고로 인한 경상자 수를 조절하세요."
    )

    st.caption("※ 모델에 포함된 기타 보조 변수(중상자수·부상신고자수 등)는 학습 데이터 평균값이 자동 반영됩니다.")

# 실시간 예측 계산
predicted_deaths = predict_fatalities(model, selected_sido, accident_count, minor_injury_count)

# 위험도 단계 판별
if predicted_deaths < 1.0:
    status_emoji = "🟢"
    status_label = "안전 (SAFE)"
    status_color = "#28a745"
    bg_color = "#eafaf1"
    status_desc = "예상 사망자 수가 1명 미만으로 비교적 안정적인 수준입니다."
    safety_tips = [
        "**스쿨존 30km/h 이하 서행 유지**: 어린이 보호구역 내 제한속도를 철저히 준수하세요.",
        "**횡단보도 앞 일시정지**: 보행자가 없더라도 신호등 없는 횡단보도 앞에서는 반드시 일시정지합니다.",
        "**주정차 금지 구역 준수**: 어린이 보호구역 내 불법 주정차는 운전자의 시야를 가리는 주원인입니다."
    ]
elif predicted_deaths < 3.0:
    status_emoji = "🟡"
    status_label = "주의 (CAUTION)"
    status_color = "#f39c12"
    bg_color = "#fef9e7"
    status_desc = "사고 규모에 따라 인명 피해 위험이 증가하고 있습니다. 각별한 주의가 필요합니다."
    safety_tips = [
        "**등·하교 시간대 우회 및 집중 주시**: 오전 8~9시, 오후 1~4시에는 어린이 돌발 횡단이 빈번합니다.",
        "**우회전 시 일시정지 철저**: 교차로 우회전 시 사각지대 보행자를 반드시 육안으로 확인하세요.",
        "**스마트폰·내비게이션 조작 금지**: 시선 분산 2초가 대형 사고로 이어질 수 있습니다."
    ]
else:
    status_emoji = "🔴"
    status_label = "위험 (DANGER)"
    status_color = "#e74c3c"
    bg_color = "#fdedec"
    status_desc = "고위험 구간입니다! 중대 교통사고 및 사망사고 발생 가능성이 매우 높습니다."
    safety_tips = [
        "**즉시 감속 및 방어운전 강화**: 스쿨존 진입 전부터 미리 속도를 줄이고 제동 거리를 확보하세요.",
        "**어린이 보행자 보호 최우선**: 주정차 차량 사이에서 어린이가 갑자기 튀어나올 수 있음을 항상 대비하세요.",
        "**학교 주변 집중 단속 및 안전 시설 점검**: 과속방지턱, 무인단속카메라, 방호울타리 구역을 각별히 주의하세요."
    ]

with col_result:
    st.subheader("📊 실시간 예측 결과")

    m1, m2 = st.columns(2)
    m1.metric(label=f"{selected_sido} 지역 예상 사망자 수", value=f"{predicted_deaths:.2f} 명")
    m2.metric(label="현재 위험 등급", value=f"{status_emoji} {status_label.split()[0]}")

    # 대형 신호등 카드
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: 3px solid {status_color};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-top: 12px;
            margin-bottom: 20px;
        ">
            <div style="font-size: 84px; line-height: 1.1;">{status_emoji}</div>
            <div style="font-size: 32px; font-weight: 800; color: {status_color}; margin-top: 8px;">
                {status_label}
            </div>
            <div style="font-size: 16px; color: #333333; margin-top: 8px;">
                {status_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 어린이 보호구역 안전 수칙 카드
    st.markdown(
        f"""
        <div style="
            background-color: #ffffff;
            border-left: 6px solid {status_color};
            border-top: 1px solid #e0e0e0;
            border-right: 1px solid #e0e0e0;
            border-bottom: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 18px 22px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        ">
            <h4 style="margin-top: 0; color: #1f2937;">🚸 어린이 보호구역(스쿨존) 필수 안전 수칙</h4>
            <ul style="margin-bottom: 0; color: #374151; line-height: 1.7;">
                <li>{safety_tips[0]}</li>
                <li>{safety_tips[1]}</li>
                <li>{safety_tips[2]}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
