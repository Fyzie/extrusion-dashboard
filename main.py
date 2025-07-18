import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def login():
    st.sidebar.title("🔒 Login")
    password = st.sidebar.text_input("Enter password", type="password")
    if password == "irnow":
        return True
    else:
        st.sidebar.warning("🔑 Incorrect password")
        return False
        
st.set_page_config(page_title="Aluminium Extrusion Data Analytics", layout="wide")

if login():
    st.title("📊 Data Analytics")
    
    # ---------------------
    # SIDEBAR
    # ---------------------
    
    uploaded_file = st.sidebar.file_uploader("📁 Upload Excel File", type=["xlsx"])
    
    if uploaded_file:
        raw_df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ File loaded!")
    
        all_columns = raw_df.columns.tolist()
        time_col = st.sidebar.selectbox("1️⃣ Select Time Column", options=[""] + all_columns)
        profile_col = st.sidebar.selectbox("2️⃣ Select Profile Column", options=[""] + all_columns)
    
        param_cols = []
        if profile_col:
            numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
            param_cols = st.sidebar.multiselect("3️⃣ Select Parameter Columns", numeric_cols)
        
        param_ranges = {}
    
        if time_col and profile_col and param_cols:
            grouping_method = st.sidebar.selectbox("Grouping Mechanism", options=["None", "Time Diff", "Parameter"])
    
            time_gap_threshold = None
            group_param_col = None
    
            if grouping_method == "Time Diff":
                time_gap_threshold = st.sidebar.number_input("⏱️ Time Diff Threshold (seconds)", min_value=1, max_value=3600, value=30, step=1)
            elif grouping_method == "Parameter":
                group_param_col = st.sidebar.selectbox("Select Column for Grouping", options=[""] + param_cols)
    
            remove_zero = st.sidebar.checkbox("Remove Zero", value=False)
            zero_out_rows = st.sidebar.checkbox("Zero Out", value=False)
    
            individual_filter_states = {}
            for col in param_cols:
                key = f"filter_{col}"
                individual_filter_states[col] = st.sidebar.checkbox(f"{col} Filter", value=True, key=key)
    
            df = raw_df[[time_col, profile_col] + param_cols].copy()
            df.rename(columns={time_col: "TIME", profile_col: "PROFILE"}, inplace=True)
            df["TIME"] = pd.to_datetime(df["TIME"], format="%H:%M:%S", errors='coerce')
            df = df.dropna(subset=["TIME", "PROFILE"] + param_cols)
            df["PROFILE"] = df["PROFILE"].astype(str).str.strip()
    
            if "RAM_SPEED" in param_cols:
                df["RAM_SPEED"] = round(df["RAM_SPEED"] * 10 / 60)
    
            enabled_cols = [col for col in param_cols if individual_filter_states.get(col, False)]
    
            if remove_zero and enabled_cols:
                df = df[~(df[enabled_cols] == 0).any(axis=1)]
    
            if zero_out_rows and enabled_cols:
                mask = (df[enabled_cols] == 0).any(axis=1)
                df.loc[mask, enabled_cols] = 0
    
            for col in param_cols:
                col_data = pd.to_numeric(df[col], errors='coerce')
                min_val = float(col_data.min())
                max_val = float(col_data.max())
    
                if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                    st.sidebar.warning(f"⚠️ Cannot create range filter for `{col}`.")
                    continue
    
                if individual_filter_states[col]:
                    min_input = st.sidebar.number_input(
                        f"{col} Min", min_value=min_val, max_value=max_val,
                        value=min_val, step=0.1, format="%.3f", key=f"{col}_min_input"
                    )
                    max_input = st.sidebar.number_input(
                        f"{col} Max", min_value=min_val, max_value=max_val,
                        value=max_val, step=0.1, format="%.3f", key=f"{col}_max_input"
                    )
                    if min_input > max_input:
                        st.sidebar.warning(f"⚠️ For {col}, Min is greater than Max. Please adjust.")
                    else:
                        param_ranges[col] = (min_input, max_input)
                else:
                    param_ranges[col] = (min_val, max_val)
    
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Plotting", "🧮 Table Summary", "🖥️ Forecasting", "🔍 Recommendation"])
            
    # ---------------------
    # TAB 1: PLOTTING DASHBOARD
    # ---------------------
            with tab1:
                st.header("📈 Plotting Dashboard")
    
                selected_profiles = st.multiselect("🎯 Select PROFILE(s) to plot", df["PROFILE"].unique(), default=None)
                left_y = st.multiselect("📊 Left Y-axis", options=param_cols, default=[param_cols[0]])
                right_y = st.multiselect("📊 Right Y-axis", options=[col for col in param_cols if col not in left_y])
                min_points = st.slider("Min Points per Segment", 1, 50, 10)
                show_labels = st.checkbox("Show Labels", value=False)
    
                df_plot = df[df["PROFILE"].isin(selected_profiles)]
                for col, (min_v, max_v) in param_ranges.items():
                    if individual_filter_states.get(col, False):
                        if zero_out_rows and not remove_zero:
                            df_plot = df_plot[(df_plot[col] == 0) | ((df_plot[col] >= min_v) & (df_plot[col] <= max_v))]
                        else:
                            df_plot = df_plot[(df_plot[col] >= min_v) & (df_plot[col] <= max_v)]
    
    
                for profile in df_plot["PROFILE"].unique():
                    df_profile = df_plot[df_plot["PROFILE"] == profile].copy()
                    if grouping_method == "Time Diff":
                        df_profile["TIME_DIFF"] = df_profile["TIME"].diff().abs().dt.total_seconds()
                        df_profile["GROUP"] = (df_profile["TIME_DIFF"] >= time_gap_threshold).cumsum()
                    elif grouping_method == "Parameter" and group_param_col in df.columns:
                        df_profile["VAL"] = df_profile[group_param_col]
                        df_profile["PREV_VAL"] = df_profile["VAL"].shift()
                        df_profile["NEW_GROUP"] = (df_profile["VAL"] != df_profile["PREV_VAL"])
                        df_profile["GROUP"] = df_profile["NEW_GROUP"].cumsum().fillna(0).astype(int)
                    else:
                        df_profile["GROUP"] = 0
    
                    valid_groups = [g for g, gdf in df_profile.groupby("GROUP") if len(gdf) >= min_points]
                    if not valid_groups:
                        st.warning(f"No valid groups for {profile}")
                        continue
    
                    for group_id in valid_groups:
                        group_df = df_profile[df_profile["GROUP"] == group_id]
                        fig = go.Figure()
    
                        for col in left_y:
                            fig.add_trace(go.Scatter(
                                x=group_df["TIME"].dt.strftime("%H:%M:%S"), y=group_df[col],
                                mode='lines+markers', name=f"{col} (L)", yaxis='y1'
                            ))
    
                        for col in right_y:
                            fig.add_trace(go.Scatter(
                                x=group_df["TIME"].dt.strftime("%H:%M:%S"), y=group_df[col],
                                mode='lines+markers', name=f"{col} (R)", yaxis='y2'
                            ))
    
                        if show_labels:
                            time_labels = group_df["TIME"].dt.strftime("%M:%S")
                            for col in left_y:
                                fig.add_trace(go.Scatter(
                                    x=group_df["TIME"], y=group_df[col],
                                    mode='text',
                                    text=[f"{t} ({v:.1f})" for t, v in zip(time_labels, group_df[col])],
                                    textposition="top center", showlegend=False, yaxis='y1',
                                    textfont=dict(color='firebrick')
                                ))
                            for col in right_y:
                                fig.add_trace(go.Scatter(
                                    x=group_df["TIME"], y=group_df[col],
                                    mode='text',
                                    text=[f"{t} ({v:.1f})" for t, v in zip(time_labels, group_df[col])],
                                    textposition="bottom center", showlegend=False, yaxis='y2',
                                    textfont=dict(color='blue')
                                ))
    
                        if grouping_method == "Parameter":
                            id = str(group_id)
                        elif grouping_method == "Time Diff":
                            id = str(group_id + 1)
                        else:
                            id = str(0)
                        
                        fig.update_layout(
                            title=dict(
                                text=f"{profile} - Segment {id} (n={len(group_df)})",
                                y=0.95,
                                x=0.5,
                                xanchor='center',
                                yanchor='top'
                            ),
                            xaxis=dict(title="Time"),
                            yaxis=dict(title="Left Axis", side='left'),
                            yaxis2=dict(title="Right Axis", overlaying='y', side='right'),
                            height=500,
                            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                            margin=dict(t=70, b=100)
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # ---------------------
    # TAB 2: PROFILE SUMMARY
    # ---------------------
            with tab2:
                st.header("🧮 Profile Summary Dashboard")
    
                df_selected = df.copy()
    
                profiles = df_selected["PROFILE"].dropna().unique().tolist()
                profile_options = ["ALL"] + profiles
                selected_profiles_summary = st.multiselect("🎯 Select PROFILE(s) to filter", profile_options, default="ALL")
    
                if "ALL" in selected_profiles_summary:
                    filtered_df = df_selected.copy()
                else:
                    filtered_df = df_selected[df_selected["PROFILE"].isin(selected_profiles_summary)].copy()
    
                filter_query = []
                for col in param_cols:
                    if individual_filter_states.get(col, False):
                        if col in param_ranges:
                            min_val, max_val = param_ranges[col]
                            if zero_out_rows and not remove_zero:
                                filter_query.append(f"({col} == 0 or ({min_val} <= {col} <= {max_val}))")
                            else:
                                filter_query.append(f"{min_val} <= {col} <= {max_val}")
    
                if filter_query:
                    filtered_df = filtered_df.query(" and ".join(filter_query))
    
                filtered_df["TIME"] = pd.to_datetime(filtered_df["TIME"], format="%I:%M:%S %p", errors='coerce').dt.time
                filtered_df = filtered_df.dropna(subset=["TIME"])
                filtered_df["TIME"] = filtered_df["TIME"].astype(str)
    
                cols = filtered_df.columns.tolist()
                cols.insert(0, cols.pop(cols.index("TIME")))
                filtered_df = filtered_df[cols]
                filtered_df["Time_Seconds"] = filtered_df["TIME"].apply(
                    lambda t: int(t.split(":")[0]) * 3600 + int(t.split(":")[1]) * 60 + int(t.split(":")[2])
                )
                filtered_df["Prev_Profile"] = filtered_df["PROFILE"].shift()
                filtered_df["New_Group"] = filtered_df["Prev_Profile"] != filtered_df["PROFILE"]
    
                if grouping_method == "Time Diff":
                    filtered_df["Time_Diff"] = filtered_df["Time_Seconds"].diff().abs().fillna(0)
                    filtered_df["Time_Gap"] = filtered_df["Time_Diff"] >= time_gap_threshold
                    filtered_df["Group"] = (filtered_df["New_Group"] | filtered_df["Time_Gap"]).cumsum()
                elif grouping_method == "Parameter" and group_param_col in df.columns:
                    filtered_df["Val"] = filtered_df[group_param_col]
                    filtered_df["Prev_Val"] = filtered_df["Val"].shift()
                    filtered_df["New_Group"] = filtered_df["Val"] != filtered_df["Prev_Val"]
                    filtered_df["Group"] = filtered_df["New_Group"].cumsum().fillna(0).astype(int)
                else:
                    filtered_df["Group"] = 0
    
                profile_summary = pd.DataFrame({
                    "Num Datapoints": filtered_df["PROFILE"].value_counts(),
                    "Num Groups": filtered_df.groupby("PROFILE")["Group"].nunique(),
                })
    
                for col in param_cols:
                    if col in filtered_df.columns:
                        profile_summary[f"Min {col}"] = filtered_df.groupby("PROFILE")[col].min()
                        profile_summary[f"Max {col}"] = filtered_df.groupby("PROFILE")[col].max()
                        profile_summary[f"Avg {col}"] = filtered_df.groupby("PROFILE")[col].mean()
    
                profile_summary = profile_summary.fillna(0).sort_values("Num Datapoints", ascending=False)
    
                st.subheader("📄 Filtered Data")
                st.dataframe(filtered_df)
    
                st.subheader("📊 Profile Summary")
                st.dataframe(profile_summary)
    
    # ---------------------
    # TAB 3: AI ANALYTICS
    # ---------------------
            with tab3:
                st.header("🤖 Forecasting Trainer")
    
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
                ai_targets = st.multiselect("🎯 Select Target Variable(s)", numeric_cols)
                ai_inputs = st.multiselect("🧠 Select Input Features", numeric_cols)
    
                profile_filter = st.selectbox("🧪 Filter by Profile", ["ALL"] + sorted(df["PROFILE"].unique()), key="lstm_profile")
    
                df_model = df.copy()
                if profile_filter != "ALL":
                    df_model = df_model[df_model["PROFILE"] == profile_filter]
    
                for col, (min_v, max_v) in param_ranges.items():
                    if individual_filter_states.get(col, False):
                        if zero_out_rows and not remove_zero:
                            df_model = df_model[(df_model[col] == 0) | ((df_model[col] >= min_v) & (df_model[col] <= max_v))]
                        else:
                            df_model = df_model[(df_model[col] >= min_v) & (df_model[col] <= max_v)]
    
                seq_len = st.number_input("📏 Sequence Length", min_value=2, max_value=50, value=5, step=1)
    
                if ai_inputs and ai_targets:
                    df_scaled = df_model.copy()
    
                    input_scalers = {}
                    for col in ai_inputs:
                        scaler = StandardScaler()
                        df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
                        input_scalers[col] = scaler
    
                    target_scalers = {}
                    for col in ai_targets:
                        if col not in ai_inputs:
                            scaler = StandardScaler()
                            df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
                            target_scalers[col] = scaler
    
                    with st.expander("Training Data Preview"):
                        st.subheader("📊 Raw Input and Target Preview")
                        st.write(df_model[list(dict.fromkeys(ai_inputs + ai_targets))].head(10))
    
                        st.subheader("🧮 Scaled Input and Target Preview")
                        st.write(df_scaled[list(dict.fromkeys(ai_inputs + ai_targets))].head(10))
    
                    def create_sequences(df, input_cols, target_cols, seq_len):
                        X, Y = [], []
                        for i in range(len(df) - seq_len):
                            X.append(df[input_cols].iloc[i:i+seq_len].values)
                            Y.append(df[target_cols].iloc[i+seq_len].values)
                        return np.array(X), np.array(Y)
    
                    with st.spinner("🔄 Creating sequences..."):
                        X, Y = create_sequences(df_scaled, ai_inputs, ai_targets, seq_len)
    
                        df_unscaled = df_model.copy()
                        X_unscaled, Y_unscaled = [], []
                        for i in range(len(df_unscaled) - seq_len):
                            input_seq = df_unscaled.iloc[i:i+seq_len][ai_inputs]
                            target_row = df_unscaled.iloc[i+seq_len][ai_targets]
                            X_unscaled.append(input_seq)
                            Y_unscaled.append(target_row)
    
                    with st.expander("Sequence Data (Unscaled)"):
                        st.subheader("📚 Full Training Samples (Unscaled)")
                        st.write("🔹 First full input sequence (raw values):")
                        st.write(X_unscaled[0])
    
                        st.write("🎯 Corresponding target (raw values):")
                        st.write(pd.DataFrame([Y_unscaled[0].values], columns=ai_targets))
    
                    with st.expander("Sequence Data (Scaled)"):
                        st.subheader("🧩 Sample Sequence Data (Scaled)")
                        st.write("🔹 First input sequence (X[0]):")
                        st.write(pd.DataFrame(X[0], columns=ai_inputs))
    
                        st.write("🎯 Corresponding target (Y[0]):")
                        st.write(pd.DataFrame(Y[0:1], columns=ai_targets))
    
                    if st.button("🚀 Train Model"):
    
                        X_train = torch.tensor(X, dtype=torch.float32)
                        y_train = torch.tensor(Y, dtype=torch.float32)
    
                        st.write(f"📈 Training samples: {len(X_train)} | Input shape: {X_train.shape[1:]}")
    
                        class LSTMModel(nn.Module):
                            def __init__(self, input_size, hidden_size, output_size):
                                super().__init__()
                                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                                self.linear = nn.Linear(hidden_size, output_size)
    
                            def forward(self, x):
                                lstm_out, _ = self.lstm(x)
                                return self.linear(lstm_out[:, -1, :])
    
                        model = LSTMModel(input_size=len(ai_inputs), hidden_size=64, output_size=len(ai_targets))
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
                        with st.spinner("🔄 Training model..."):
                            epochs = 100
                            for epoch in range(epochs):
                                model.train()
                                optimizer.zero_grad()
                                y_pred = model(X_train)
                                loss = criterion(y_pred, y_train)
                                loss.backward()
                                optimizer.step()
    
                        st.success(f"✅ Model trained successfully! Final Loss: {loss.item():.4f}")
    
                        # Store everything needed for simulation
                        st.session_state["lstm_model"] = model
                        st.session_state["input_scalers"] = input_scalers
                        st.session_state["target_scalers"] = target_scalers
                        st.session_state["seq_len"] = seq_len
                        st.session_state["ai_inputs"] = ai_inputs
                        st.session_state["ai_targets"] = ai_targets
                else:
                    st.warning("Please select at least one input and one target variable to proceed.")
    
    # ---------------------
    # SIMULATION
    # ---------------------
                st.header("📈 Forecasting Simulation")
    
                model = st.session_state.get("lstm_model")
                if not model:
                    st.warning("Train a model first!")
                else:
                    profile_name = st.session_state["lstm_profile"]
                    input_cols = st.session_state["ai_inputs"]
                    target_cols = st.session_state["ai_targets"]
                    seq_len = st.session_state["seq_len"]
                    input_scalers = st.session_state["input_scalers"]
                    target_scalers = st.session_state["target_scalers"]
                    zero_out_rows = st.session_state.get("zero_out_rows", False)
    
                    if profile_name == "ALL":
                        raw_df = df.copy()
                    else:
                        raw_df = df[df["PROFILE"] == profile_name].copy()
    
                    filter_query = []
                    for col in input_cols:
                        if individual_filter_states.get(col, False) and col in param_ranges:
                            min_val, max_val = param_ranges[col]
                            if zero_out_rows:
                                filter_query.append(f"({col} == 0 or ({min_val} <= {col} <= {max_val}))")
                            else:
                                filter_query.append(f"{min_val} <= {col} <= {max_val}")
                    if filter_query:
                        raw_df = raw_df.query(" and ".join(filter_query))
    
                    df_unscaled = raw_df.copy().reset_index(drop=True)
    
                    for col in input_cols:
                        if col in input_scalers:
                            raw_df[col] = input_scalers[col].transform(raw_df[[col]])
    
                    scaled_target_cols = [col for col in target_cols if col in target_scalers]
                    for col in scaled_target_cols:
                        raw_df[col] = target_scalers[col].transform(raw_df[[col]])
    
                    def create_seq(df, input_cols, target_cols, seq_len):
                        X, Y = [], []
                        for i in range(len(df) - seq_len):
                            X.append(df[input_cols].iloc[i:i+seq_len].values)
                            Y.append(df[target_cols].iloc[i+seq_len].values)
                        return np.array(X), np.array(Y)
    
                    X_sim, Y_true = create_seq(raw_df, input_cols, target_cols, seq_len)
                    if len(X_sim) == 0:
                        st.warning("Not enough data after filtering to simulate.")
                    else:
                        X_sim = torch.tensor(X_sim, dtype=torch.float32)
    
                        model.eval()
                        with torch.no_grad():
                            preds = model(X_sim).numpy()
    
                        pred_df = pd.DataFrame(preds, columns=target_cols)
                        actual_df = pd.DataFrame(Y_true, columns=target_cols)
    
                        for col in target_cols:
                            if col in target_scalers:
                                pred_df[col] = target_scalers[col].inverse_transform(pred_df[[col]])
                                actual_df[col] = target_scalers[col].inverse_transform(actual_df[[col]])
                            elif col in input_scalers:
                                pred_df[col] = input_scalers[col].inverse_transform(pred_df[[col]])
                                actual_df[col] = input_scalers[col].inverse_transform(actual_df[[col]])
    
                        time_index = df_unscaled["TIME"].iloc[seq_len:].reset_index(drop=True)
                        profile_index = df_unscaled["PROFILE"].iloc[seq_len:].reset_index(drop=True)
                        if pd.api.types.is_datetime64_any_dtype(time_index):
                            time_index_str = time_index.dt.strftime("%H:%M:%S")
                        else:
                            time_index_str = time_index
    
                        for col in target_cols:
                            st.markdown(f"**📈 {col}: Actual vs Predicted over Time**")
                            st.line_chart(pd.DataFrame({
                                "Actual": actual_df[col].values,
                                "Predicted": pred_df[col].values
                            }, index=time_index_str))
    
                        # Show input feature traces
                        st.subheader("📥 Input Parameters Over Time")
                        input_inputs = df_unscaled[input_cols].iloc[seq_len:].reset_index(drop=True)
    
                        for col in input_cols:
                            if not col in target_cols:
                                st.markdown(f"**📌 {col}**")
                                st.line_chart(pd.DataFrame({col: input_inputs[col].values}, index=time_index_str))
    
                        # Show table
                        st.subheader("📊 Forecasting Table View")
    
                        input_inputs = input_inputs.drop(columns=[col for col in target_cols if col in input_inputs.columns])
    
                        result_table = pd.concat([
                            time_index.rename("TIME"),
                            profile_index.rename("PROFILE"),
                            input_inputs.reset_index(drop=True),
                            actual_df.reset_index(drop=True),
                            pred_df.rename(columns={col: f"{col}_PRED" for col in target_cols}).reset_index(drop=True)
                        ], axis=1)
    
                        st.dataframe(result_table)
    
            with tab4:
                st.header("📈 Optimization Recommendation")
    
                st.markdown("""
                To **recommend optimal input parameter configurations** for improving process performance
                while ensuring safety and quality standards.
    
                To proceed, it requires **multiple labeled process cycles** with ✅ OK cycles and
                ❌ NG (Not Good) cycles.
    
                The model will be able to learn what make a "good" vs. "bad" process configuration.
                """)
    
    else:
        st.info("📤 Upload an Excel file to begin.")
