import streamlit as st
import pandas as pd
from jiwer import wer
import io
import random
import time

# ---------------------------
# 유틸 함수들
# ---------------------------

def get_ab_mapping(case_id: str):
    """
    케이스별로 A/B에 어떤 시스템(Selvas / GPT4o)을 할당할지
    항상 같은 방식으로 랜덤 매핑 (블라인드 유지용)
    """
    rng = random.Random(str(case_id))
    if rng.random() < 0.5:
        return {"A": "Selvas", "B": "GPT4o"}
    else:
        return {"A": "GPT4o", "B": "Selvas"}


ERROR_LABELS = {
    0: "0: 정확",
    1: "1: 표기/언어 오류만 (typo, 단위 등, 의미 동일)",
    2: "2: 임상적 minor (의미 다르지만 TI-RADS/management 동일)",
    3: "3: 임상적 major (TI-RADS/management 달라짐)"
}

LIKERT_LABELS = {
    1: "1: 매우 부담됨",
    2: "2: 다소 부담됨",
    3: "3: 보통",
    4: "4: 부담 적음",
    5: "5: 거의 부담 없음"
}

# 위치(feature)는 제거하고 6개만 사용
FEATURES = [
    ("size", "Size (cm)"),
    ("composition", "Composition"),
    ("echogenicity", "Echogenicity"),
    ("shape", "Shape"),
    ("margin", "Margins"),
    ("calcification", "Calcifications"),
]

# ---------------------------
# Streamlit UI 시작
# ---------------------------

st.set_page_config(page_title="Thyroid STT 평가 앱", layout="wide")
st.title("Thyroid STT 평가 앱 (Selvas vs GPT‑4o)")

st.markdown("""
- 골드 스탠다드와 두 개의 STT 결과를 **A / B** 로 블라인드 표시합니다.  
- 각 케이스에서 두 리포트 모두에 대해  
  1) 교정 시작/종료 타이머  
  2) 교정된 최종 판독문 입력  
  3) 편집 부담(1–5점)  
  4) Linguistic error 유무  
  5) Feature error (0~3)  
  를 기록합니다.  
- Turing test처럼 **A/B 중 더 좋은 리포트**도 선택합니다.  
""")

# Custom CSS - 상단 고정, 하단만 스크롤
st.markdown("""
<style>
    /* 전체 앱 컨테이너의 스크롤 제거 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-height: 100vh;
        overflow: hidden;
    }
    
    /* 고정 상단 섹션 */
    .fixed-top {
        position: relative;
        background: white;
        z-index: 100;
        padding-bottom: 10px;
    }
    
    /* 스크롤 가능한 평가 영역 */
    .scrollable-evaluation {
        height: 55vh;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 20px;
        border: 2px solid #007bff;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* 저장 버튼 영역 */
    .fixed-bottom {
        position: relative;
        background: white;
        padding-top: 10px;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)


# ---- 사이드바: 엑셀 업로드 & 기본 설정 ----
st.sidebar.header("1. 엑셀 업로드")

uploaded_file = st.sidebar.file_uploader(
    "케이스 엑셀 파일 (.xlsx)", type=["xlsx"]
)

rater_id = st.sidebar.text_input("평가자 ID (예: R1, R2)", value="R1")

if uploaded_file is None:
    st.info("왼쪽에서 엑셀 파일을 먼저 업로드 해 주세요.")
    st.stop()

# 엑셀 읽기
try:
    base_df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"엑셀을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

if base_df.empty:
    st.error("엑셀이 비어 있습니다.")
    st.stop()

st.sidebar.subheader("2. 컬럼 매핑")

cols = list(base_df.columns)
if len(cols) < 4:
    st.error("엑셀에 최소 4개 이상의 컬럼이 필요합니다 (CaseID, Gold, Selvas, GPT4o 등).")
    st.stop()

# 인덱스 안전하게
def safe_index(i):
    return i if i < len(cols) else 0

case_col = st.sidebar.selectbox("Case ID 컬럼", cols, index=safe_index(0))
gold_col = st.sidebar.selectbox("Gold(Reference) 텍스트 컬럼", cols, index=safe_index(1))
selvas_col = st.sidebar.selectbox("Selvas STT 텍스트 컬럼", cols, index=safe_index(2))
gpt4o_col = st.sidebar.selectbox("GPT‑4o STT 텍스트 컬럼", cols, index=safe_index(3))

# ---- WER 자동 계산 ----
base_df["WER_Selvas"] = base_df.apply(
    lambda row: wer(str(row[gold_col]), str(row[selvas_col])) if pd.notnull(row[selvas_col]) else None,
    axis=1
)
base_df["WER_GPT4o"] = base_df.apply(
    lambda row: wer(str(row[gold_col]), str(row[gpt4o_col])) if pd.notnull(row[gpt4o_col]) else None,
    axis=1
)

# ---- 세션 상태 초기화 ----
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

if "eval_df" not in st.session_state:
    st.session_state.eval_df = pd.DataFrame()

# ---- 사이드바: 케이스 네비게이션 ----
total_cases = len(base_df)
st.sidebar.subheader("3. 케이스 이동")

st.sidebar.markdown(f"총 **{total_cases}** 케이스")

col_prev, col_next = st.sidebar.columns(2)
if col_prev.button("이전 케이스"):
    st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
if col_next.button("다음 케이스"):
    st.session_state.current_idx = min(total_cases - 1, st.session_state.current_idx + 1)

# 특정 인덱스로 점프
jump_to = st.sidebar.number_input(
    "특정 인덱스로 이동 (1 ~ N)",
    min_value=1, max_value=total_cases,
    value=st.session_state.current_idx + 1
)
if st.sidebar.button("이동"):
    st.session_state.current_idx = int(jump_to) - 1

# ---- 현재 케이스 정보 ----
row = base_df.iloc[st.session_state.current_idx]
case_id = row[case_col]
st.markdown(f"### 케이스 {case_id}  ( {st.session_state.current_idx + 1} / {total_cases} )")

# 상단 고정 영역 시작
st.markdown('<div class="fixed-top">', unsafe_allow_html=True)

# A/B 매핑 (블라인드용)
mapping = get_ab_mapping(case_id=str(case_id))

def get_text_for_label(label: str):
    sysname = mapping[label]
    if sysname == "Selvas":
        return row[selvas_col]
    else:
        return row[gpt4o_col]

# ---- 골드 스탠다드 표시 ----
with st.expander("Gold standard 판독문 (Reference)", expanded=True):
    st.write(row[gold_col])

# ---- A/B 리포트 표시 ----
colA, colB = st.columns(2)

with colA:
    st.subheader("Report A")
    st.write(get_text_for_label("A"))

with colB:
    st.subheader("Report B")
    st.write(get_text_for_label("B"))

st.markdown('</div>', unsafe_allow_html=True)
# 상단 고정 영역 끝

st.markdown("---")

# ---- 타이머 상태 초기화 함수 ----
def init_timer_state(case_id, label):
    key_prefix = f"{case_id}_{label}"
    elapsed_key = f"corr_elapsed_{key_prefix}"
    running_key = f"corr_running_{key_prefix}"
    start_key = f"corr_start_{key_prefix}"

    if elapsed_key not in st.session_state:
        st.session_state[elapsed_key] = 0.0
    if running_key not in st.session_state:
        st.session_state[running_key] = False
    if start_key not in st.session_state:
        st.session_state[start_key] = 0.0

    return elapsed_key, running_key, start_key

# ---- Report A/B 각각에 대한 평가 폼 ----

# 스크롤 가능한 평가 입력 영역 시작
st.markdown('<div class="scrollable-evaluation">', unsafe_allow_html=True)

st.markdown("### 각 리포트별 평가 입력")
st.markdown("각 리포트에 대해 교정 → 편집부담 → 오류 라벨링 순서로 입력합니다.")

def report_form(label: str, turing_choice: str, winner_system: str):
    """
    Report A 또는 B에 대한 입력 폼.
    return: dict (해당 리포트의 평가 정보)
    """
    sysname = mapping[label]  # 'Selvas' or 'GPT4o'
    if sysname == "Selvas":
        wer_value = row["WER_Selvas"]
    else:
        wer_value = row["WER_GPT4o"]

    st.markdown(f"#### Report {label}")

    # -------------------------
    # 1) 교정 시작/종료 타이머
    # -------------------------
    st.markdown("**1) 교정 시간 측정**")
    st.markdown(
        "- 위 Report 텍스트를 복사해서 아래에서 실제로 교정하세요.\n"
        "- `교정 시작`을 누르고 교정을 시작한 뒤, 완료되면 `교정 종료`를 눌러주세요.\n"
        "- 필요하면 `리셋`으로 시간을 다시 0초에서 시작할 수 있습니다."
    )

    elapsed_key, running_key, start_key = init_timer_state(case_id, label)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        if st.button("교정 시작", key=f"btn_start_{case_id}_{label}"):
            st.session_state[running_key] = True
            st.session_state[start_key] = time.time()
    with col_t2:
        if st.button("교정 종료", key=f"btn_stop_{case_id}_{label}"):
            if st.session_state[running_key]:
                elapsed = time.time() - st.session_state[start_key]
                st.session_state[elapsed_key] += elapsed
                st.session_state[running_key] = False
    with col_t3:
        if st.button("리셋", key=f"btn_reset_{case_id}_{label}"):
            st.session_state[elapsed_key] = 0.0
            st.session_state[running_key] = False

    corr_time = st.session_state[elapsed_key]

    st.write(f"현재 누적 교정 시간: **{corr_time:.1f} 초**")
    if st.session_state[running_key]:
        st.caption("상태: ⏱ 교정 진행 중")
    else:
        st.caption("상태: ⏸ 정지")

    # -------------------------
    # 2) 교정된 최종 판독문 입력
    # -------------------------
    st.markdown("**2) 교정된 최종 판독문**")
    corrected_text = st.text_area(
        "교정된 최종 판독문을 입력하세요.",
        key=f"corrected_{case_id}_{label}",
        height=200,
        placeholder="Report A/B 내용을 복사해서 붙여넣고, 실제로 수정한 최종 판독문을 여기에 남겨 주세요."
    )

    # -------------------------
    # 3) 편집 부담 (1~5점)
    # -------------------------
    st.markdown("**3) 편집 부담 평가 (1–5점)**")
    editing_burden = st.selectbox(
        "편집 부담 (1=매우 부담됨, 5=거의 없음)",
        list(LIKERT_LABELS.keys()),
        format_func=lambda x: LIKERT_LABELS[x],
        key=f"likert_editing_{case_id}_{label}"
    )

    st.markdown("---")
    st.markdown("**4) Linguistic / Feature error 라벨링 (교정 후 입력)**")

    # -------------------------
    # 4) Linguistic / Lexical Error
    # -------------------------
    lexical_error = st.selectbox(
        "Linguistic / Lexical error (typo, 단위, 순수 표기오류 등) 존재 여부",
        [0, 1],
        format_func=lambda x: "0: 없음" if x == 0 else "1: 있음",
        key=f"lex_{case_id}_{label}"
    )

    # -------------------------
    # 5) Feature error (0~3)
    # -------------------------
    st.markdown("Feature error (0=정확, 3=임상적 major)")

    feature_errors = {}
    cols_feat_1, cols_feat_2 = st.columns(2)
    for i, (feat_key, feat_label) in enumerate(FEATURES):
        col = cols_feat_1 if i % 2 == 0 else cols_feat_2
        with col:
            val = st.selectbox(
                feat_label,
                list(ERROR_LABELS.keys()),
                format_func=lambda x: ERROR_LABELS[x],
                key=f"{feat_key}_{case_id}_{label}"
            )
            feature_errors[feat_key] = val

    # -------------------------
    # 결과 dict
    # -------------------------
    result = {
        "case_id": case_id,
        "rater_id": rater_id,
        "label": label,          # A 또는 B
        "system": sysname,       # Selvas / GPT4o
        "WER": wer_value,
        "lexical_error": lexical_error,
        "editing_burden": editing_burden,
        "correction_time_sec": corr_time,
        "corrected_report_text": corrected_text,
        "turing_choice_label": turing_choice,
        "turing_winner_system": winner_system,
    }

    # feature errors 추가
    for fk, fv in feature_errors.items():
        result[f"feat_{fk}_error"] = fv

    return result


col_form_A, col_form_B = st.columns(2)

with col_form_A:
    # Turing test 변수를 먼저 초기화 (임시값)
    res_A = report_form("A", "", "")

with col_form_B:
    res_B = report_form("B", "", "")

st.markdown("---")

# ---- Turing test (A/B 선호도) - linguistic/feature error 라벨링 뒤로 이동 ----
st.markdown("### Turing test: 두 리포트 중 어떤 것을 최종 리포트로 쓰시겠습니까?")

turing_choice = st.radio(
    "선호 리포트 선택",
    ["A", "B", "둘 다 비슷함", "둘 다 사용하기 어렵다"],
    key=f"turing_{case_id}"
)

if turing_choice in ["A", "B"]:
    winner_system = mapping[turing_choice]  # Selvas / GPT4o
elif turing_choice == "둘 다 비슷함":
    winner_system = "Tie"
else:
    winner_system = "None"

# Turing test 결과를 res_A, res_B에 업데이트
res_A["turing_choice_label"] = turing_choice
res_A["turing_winner_system"] = winner_system
res_B["turing_choice_label"] = turing_choice
res_B["turing_winner_system"] = winner_system

st.markdown("---")

st.markdown('</div>', unsafe_allow_html=True)
# 스크롤 가능한 평가 입력 영역 끝

# 저장 버튼 영역 시작
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)

# ---- 저장 버튼 ----

def save_current_case():
    # 현재까지 저장된 결과 가져오기
    eval_df = st.session_state.eval_df

    # ⚠️ eval_df가 비어 있거나, 아직 case_id / rater_id 컬럼이 없을 수 있으므로
    # 그럴 때는 그냥 필터링 없이 새 row만 추가하도록 처리
    if (not eval_df.empty) and ("case_id" in eval_df.columns) and ("rater_id" in eval_df.columns):
        mask = (eval_df["case_id"] == case_id) & (eval_df["rater_id"] == rater_id)
        eval_df = eval_df[~mask]

    # 이번 케이스의 A/B 결과 2줄을 새로 DataFrame으로 만들고
    new_rows = pd.DataFrame([res_A, res_B])

    # 기존 eval_df와 합치기
    eval_df = pd.concat([eval_df, new_rows], ignore_index=True)

    # 세션에 다시 저장
    st.session_state.eval_df = eval_df

    st.success(f"케이스 {case_id} (평가자 {rater_id}) 저장 완료!")

col_s1, col_s2 = st.columns(2)
with col_s1:
    if st.button("현재 케이스 저장"):
        save_current_case()
with col_s2:
    if st.button("저장 후 다음 케이스로 이동"):
        save_current_case()
        st.session_state.current_idx = min(total_cases - 1, st.session_state.current_idx + 1)

st.markdown('</div>', unsafe_allow_html=True)
# 저장 버튼 영역 끝

# ---- 지금까지 입력한 결과 미리 보기 ----
st.markdown("### 지금까지 저장된 평가 결과 (요약)")
st.dataframe(st.session_state.eval_df)

# ---- 결과 엑셀 다운로드 ----
st.sidebar.subheader("4. 결과 저장")

if not st.session_state.eval_df.empty:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # 평가 결과
        st.session_state.eval_df.to_excel(writer, index=False, sheet_name="STT_eval")
        # 원본 데이터(선택) 같이 저장
        base_df.to_excel(writer, index=False, sheet_name="original_cases")
    output.seek(0)

    st.sidebar.download_button(
        label="결과 엑셀 다운로드",
        data=output,
        file_name=f"thyroid_STT_eval_results_{rater_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.sidebar.info("아직 저장된 평가가 없습니다.")