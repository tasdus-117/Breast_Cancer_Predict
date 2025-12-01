import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cấu hình trang
st.set_page_config(page_title="Dự đoán Ung thư vú", layout="wide")


# --- LOAD RESOURCES ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('lda_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('feature_names.pkl')
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def load_data():
    try:
        # Đọc dữ liệu từ đường dẫn Kaggle
        df = pd.read_csv("data.csv")
        # Xử lý sơ bộ giống file train
        df = df.drop(columns=['Unnamed: 32'], errors='ignore')
        return df
    except FileNotFoundError:
        return None


def main():
    st.title("🩺 Ứng dụng Dự đoán Ung thư vú")
    st.markdown("""
    Ứng dụng sử dụng thuật toán **Loigistic Regression** kết hợp **Linear Discriminant Analysis (LDA)**.
    Bạn có thể **chọn ID bệnh nhân** từ dữ liệu có sẵn để tự động điền các chỉ số và kiểm tra độ chính xác của mô hình.
    """)

    # Tải model và dữ liệu
    model, scaler, feature_names = load_artifacts()
    df_data = load_data()

    if model is None:
        st.error("Không tìm thấy file mô hình! Vui lòng chạy file `train_model.py` trước.")
        return

    # --- SIDEBAR: CHỌN BỆNH NHÂN ---
    st.sidebar.header("Chọn dữ liệu mẫu")

    selected_row = None
    actual_diagnosis = None

    if df_data is not None:
        # Tạo danh sách ID để chọn
        patient_ids = df_data['id'].astype(str).tolist()
        # Thêm tùy chọn nhập tay (None)
        option = st.sidebar.selectbox(
            "Chọn ID Bệnh nhân (để điền tự động):",
            ["Nhập thủ công"] + patient_ids
        )

        if option != "Nhập thủ công":
            # Lấy dòng dữ liệu tương ứng với ID đã chọn
            selected_id = int(option)
            selected_row = df_data[df_data['id'] == selected_id].iloc[0]

            # Lấy chẩn đoán thực tế để so sánh
            actual_diagnosis = selected_row['diagnosis']  # 'M' or 'B'

            # Hiển thị thông tin thực tế ở sidebar
            st.sidebar.divider()
            st.sidebar.markdown(f"**ID:** {selected_id}")
            if actual_diagnosis == 'M':
                st.sidebar.error(f"Thực tế: **Ác tính (M)**")
            else:
                st.sidebar.success(f"Thực tế: **Lành tính (B)**")
    else:
        st.sidebar.warning("Không tìm thấy file data.csv để load mẫu.")

    # --- FORM NHẬP LIỆU ---
    st.header("Thông số xét nghiệm")

    input_data = {}

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # Helper function để lấy giá trị default
        def get_default(feat_name):
            if selected_row is not None:
                return float(selected_row[feat_name])
            return 0.0

        # Nhóm Mean
        with col1:
            st.subheader("Chỉ số trung bình (Mean)")
            for feat in feature_names[:10]:
                input_data[feat] = st.number_input(
                    f"{feat}",
                    value=get_default(feat),
                    format="%.4f"
                )

        # Nhóm Standard Error
        with col2:
            st.subheader("Sai số chuẩn (SE)")
            for feat in feature_names[10:20]:
                input_data[feat] = st.number_input(
                    f"{feat}",
                    value=get_default(feat),
                    format="%.4f"
                )

        # Nhóm Worst
        with col3:
            st.subheader("Chỉ số tệ nhất (Worst)")
            for feat in feature_names[20:]:
                input_data[feat] = st.number_input(
                    f"{feat}",
                    value=get_default(feat),
                    format="%.4f"
                )

        submit_button = st.form_submit_button("🔍 Dự đoán kết quả")

    # --- XỬ LÝ DỰ ĐOÁN ---
    if submit_button:
        # Chuyển đổi dữ liệu đầu vào thành array
        input_df = pd.DataFrame([input_data])
        input_array = input_df.values

        # 1. Chuẩn hóa dữ liệu
        input_scaled = scaler.transform(input_array)

        # 2. Dự đoán
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # 3. Hiển thị kết quả
        st.divider()
        st.header("Kết quả phân tích từ mô hình:")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            # Quy ước từ train_model.py: 0 = Malignant (Ác tính), 1 = Benign (Lành tính)
            if prediction == 0:
                st.error(f"⚠️ **DỰ ĐOÁN: ÁC TÍNH (Malignant)**")
                st.write(f"Độ tin cậy: {probability[0] * 100:.2f}%")
            else:
                st.success(f"✅ **DỰ ĐOÁN: LÀNH TÍNH (Benign)**")
                st.write(f"Độ tin cậy: {probability[1] * 100:.2f}%")

        with col_res2:
            if actual_diagnosis:
                st.write("---")
                st.write("**So sánh với thực tế:**")
                pred_label = 'M' if prediction == 0 else 'B'
                if pred_label == actual_diagnosis:
                    st.info("👏 Mô hình dự đoán **ĐÚNG** với dữ liệu gốc.")
                else:
                    st.warning("⚠️ Mô hình dự đoán **SAI** so với dữ liệu gốc.")


if __name__ == "__main__":

    main()

