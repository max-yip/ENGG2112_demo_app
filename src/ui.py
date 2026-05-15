import streamlit as st
import pandas as pd
from .data_loader import load_experiments, get_available_models
from .inference import load_model, process_video

def render_tab1():
    st.header("Model Metrics Viewer")
    experiments = load_experiments()
    
    if not experiments:
        st.warning("No experiments found or experiments.json is missing.")
        return
        
    # Get list of experiment names
    exp_names = [exp['name'] for exp in experiments]
    selected_name = st.selectbox("Select Model to View Metrics:", exp_names)
    
    # Find the selected experiment
    selected_exp = next((exp for exp in experiments if exp['name'] == selected_name), None)
    
    if selected_exp:
        st.subheader(f"Metrics for {selected_exp['name']} ({selected_exp.get('model', 'Unknown Base')})")
        
        # Display as a clean grid of metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("mAP@50", f"{selected_exp.get('map50', 0):.4f}")
        col2.metric("mAP@50-95", f"{selected_exp.get('map50_95', 0):.4f}")
        col3.metric("Precision", f"{selected_exp.get('P', 0):.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{selected_exp.get('R', 0):.4f}")
        col5.metric("Epochs", f"{selected_exp.get('epochs', 0)}")
        col6.metric("Image Size", f"{selected_exp.get('img_size', 0)}")

def render_tab2():
    st.header("Side-by-Side Video Inference")
    available_models = get_available_models()
    
    if not available_models:
        st.warning("No models found in the 'models' directory.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        model1_name = st.selectbox("Select Model 1:", available_models, key="mod1")
        use_sahi1 = False
        if "retinanet" not in model1_name.lower() and "faster" not in model1_name.lower():
            use_sahi1 = st.toggle("Use SAHI", key="sahi1")
    with col2:
        # Default to the same model if only one exists, or second model if possible
        model2_idx = 1 if len(available_models) > 1 else 0
        model2_name = st.selectbox("Select Model 2:", available_models, index=model2_idx, key="mod2")
        use_sahi2 = False
        if "retinanet" not in model2_name.lower() and "faster" not in model2_name.lower():
            use_sahi2 = st.toggle("Use SAHI", key="sahi2")
            
    conf_threshold = st.slider("Detection Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        
    # Play controls
    start_btn = st.button("Start Inference")
    stop_placeholder = st.empty()
    
    # Placeholders for video frames
    vid_col1, vid_col2 = st.columns(2)
    with vid_col1:
        st.subheader("Model 1 Output")
        img_placeholder1 = st.empty()
        metric_placeholder1 = st.empty()
        
    with vid_col2:
        st.subheader("Model 2 Output")
        img_placeholder2 = st.empty()
        metric_placeholder2 = st.empty()
        
    if start_btn:
        stop_btn = stop_placeholder.button("Stop Inference", key="stop_btn")
        
        with st.spinner("Loading models..."):
            m1, status1, path1 = load_model(model1_name)
            m2, status2, path2 = load_model(model2_name)
            
        if m1 is None:
            metric_placeholder1.error(f"Failed to load {model1_name}. {status1}")
        if m2 is None:
            metric_placeholder2.error(f"Failed to load {model2_name}. {status2}")
            
        # Try to find the demo video
        video_path = "ENGG2112 demo.mp4"
        
        # Start generator
        for frame1, frame2, metrics1, metrics2, is_finished in process_video(
            video_path, m1, m2, conf_threshold, 
            use_sahi1=use_sahi1, use_sahi2=use_sahi2, 
            path1=path1, path2=path2
        ):
            if stop_btn or is_finished:
                break
                
            if frame1 is not None:
                img_placeholder1.image(frame1, channels="RGB", width="stretch")
            if frame2 is not None:
                img_placeholder2.image(frame2, channels="RGB", width="stretch")
                
            # Update metrics text
            if "Speed" in metrics1:
                metric_placeholder1.text(f"Speed: {metrics1['Speed']}")
            elif "error" in metrics1:
                 metric_placeholder1.error(metrics1["error"])
                 
            if "Speed" in metrics2:
                metric_placeholder2.text(f"Speed: {metrics2['Speed']}")
            elif "error" in metrics2:
                 metric_placeholder2.error(metrics2["error"])
                 
        st.success("Inference completed or stopped. You can select new models and run again.")
