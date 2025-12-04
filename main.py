import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import koreanize_matplotlib # 한글 폰트 설정을 위해 사용
import numpy as np # 혼동 행렬 시각화를 위해 numpy 추가 (오류 수정)

# 한글 폰트 설정 적용
koreanize_matplotlib.use_font()

# --- 데이터 로드 및 모델 학습 함수 (캐싱 적용) ---
# Streamlit의 @st.cache_data 데코레이터를 사용하여 데이터 로드 및 학습된 모델을 캐싱합니다.
# 이렇게 하면 앱이 다시 실행될 때마다(예: 위젯 상호 작용 시) 시간이 많이 걸리는
# 데이터 로드 및 모델 학습 단계를 다시 실행하는 것을 방지할 수 있습니다.
@st.cache_data
def load_data():
    # 실제 환경에서는 파일을 Streamlit 앱과 같은 디렉토리에 두거나,
    # 사용자에게 업로드하도록 하는 등의 방법을 사용해야 합니다.
    # 이 예제에서는 파일이 앱 실행 경로에 있다고 가정합니다.
    try:
        df = pd.read_csv("earthquake_data_tsunami.csv")
        return df
    except FileNotFoundError:
        st.error("🚨 'earthquake_data_tsunami.csv' 파일을 찾을 수 없습니다. 파일을 앱과 같은 디렉토리에 넣어주세요.")
        return None

@st.cache_resource
def train_model(df):
    if df is None:
        return None, None, None

    # STEP 3. 필요한 열 선택
    X = df[["magnitude", "depth", "latitude", "longitude"]]    # 입력 변수
    y = df["tsunami"]  # 목표 변수

    # STEP 4. 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # STEP 5. 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# --- Streamlit 앱 구성 시작 ---
def main():
    st.set_page_config(page_title="쓰나미 발생 예측 모델 (Random Forest)")
    st.title("🌊 지진 데이터 기반 쓰나미 발생 예측 분석")
    st.markdown("---")

    # 1. 데이터 로드 및 모델 학습
    df = load_data()
    if df is None:
        return

    model, X_test, y_test = train_model(df)
    if model is None:
        return

    # 2. 예측 기능 섹션
    ## 예측 사이드바
    st.sidebar.header("🗺️ 예측 변수 입력")
    st.sidebar.write("쓰나미 발생 여부를 예측하기 위한 지진의 특성을 입력하세요.")

    # 사용자 입력 받기
    # 데이터프레임이 로드되었는지 확인 후 min/max 값을 사용합니다.
    if not df.empty:
      magnitude_min = float(df['magnitude'].min())
      magnitude_max = float(df['magnitude'].max())
      depth_min = float(df['depth'].min())
      depth_max = float(df['depth'].max())
      latitude_min = float(df['latitude'].min())
      latitude_max = float(df['latitude'].max())
      longitude_min = float(df['longitude'].min())
      longitude_max = float(df['longitude'].max())
    else: # 데이터가 없는 경우를 대비한 기본값
      magnitude_min, magnitude_max = 0.0, 10.0
      depth_min, depth_max = 0.0, 1000.0
      latitude_min, latitude_max = -90.0, 90.0
      longitude_min, longitude_max = -180.0, 180.0

    magnitude = st.sidebar.slider("진도 (Magnitude)", magnitude_min, magnitude_max, 5.0)
    depth = st.sidebar.slider("깊이 (Depth, km)", depth_min, depth_max, 50.0)
    latitude = st.sidebar.number_input("위도 (Latitude)", latitude_min, latitude_max, 35.0, step=0.01)
    longitude = st.sidebar.number_input("경도 (Longitude)", longitude_min, longitude_max, 130.0, step=0.01)

    # 예측 버튼
    if st.sidebar.button("쓰나미 예측 실행"):
        # 입력 데이터를 DataFrame 형태로 변환
        input_data = pd.DataFrame([[magnitude, depth, latitude, longitude]],
                                     columns=["magnitude", "depth", "latitude", "longitude"])

        # 예측 수행
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)

        st.subheader("📊 예측 결과")
        if prediction == 1:
            st.success("## ⚠️ 쓰나미 **발생 예측**!")
            st.write(f"**쓰나미 발생 확률:** **{prediction_proba[0][1]*100:.2f}%**")
        else:
            st.info("## ✅ 쓰나미 **미발생 예측**")
            st.write(f"**쓰나미 미발생 확률:** **{prediction_proba[0][0]*100:.2f}%**")

        st.markdown("---")


    # 3. 모델 분석 섹션 (탭 구성)
    st.header("🔬 모델 성능 및 분석")
    tab1, tab2, tab3 = st.tabs(["모델 성능 지표", "특성 중요도 시각화", "데이터 미리보기"])

    with tab1:
        st.subheader("모델 평가 결과")
        if model and X_test is not None:
            # STEP 6. 예측 및 평가
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            st.metric("테스트 데이터 정확도 (Accuracy)", f"{accuracy*100:.2f}%")

            st.markdown("**분류 리포트 (Classification Report)**")
            st.dataframe(pd.DataFrame(report).transpose())

            st.markdown("**혼동 행렬 (Confusion Matrix)**")
            # 혼동 행렬 시각화
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            plt.title('혼동 행렬', y=1.1)
            fig.colorbar(cax)
            ax.set_xticklabels([''] + [0, 1])
            ax.set_yticklabels([''] + [0, 1])
            plt.xlabel('예측 값 (Predicted)')
            plt.ylabel('실제 값 (Actual)')
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, f'{val}', ha='center', va='center', color='red' if i == j else 'black')
            st.pyplot(fig)
            
            st.markdown("> **레이블:** '0'은 쓰나미 미발생, '1'은 쓰나미 발생을 의미합니다.")


    with tab2:
        st.subheader("특성 중요도 시각화")
        # STEP 7. 중요 변수 시각화
        importances = model.feature_importances_
        feature_names = X_test.columns

        # Matplotlib을 사용하여 시각화
        fig, ax = plt.subplots()
        ax.bar(feature_names, importances)
        ax.set_title("Feature Importance (특성이 쓰나미 예측에 미치는 영향)")
        ax.set_ylabel("중요도")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig) # Streamlit에 Matplotlib 차트 표시 


    with tab3:
        st.subheader("데이터셋 미리보기")
        st.write(df.head())
        st.write(f"**전체 데이터 크기:** {df.shape[0]} 행, {df.shape[1]} 열")
        st.write("**컬럼 설명:**")
        st.markdown(
            """
            * `magnitude`: 진도 (지진의 크기)
            * `depth`: 깊이 (지진 발생 깊이, km)
            * `latitude`: 위도
            * `longitude`: 경도
            * `tsunami`: 쓰나미 발생 여부 (0: 미발생, 1: 발생)
            """
        )

if __name__ == "__main__":
    main()
    
