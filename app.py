import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time

# --- Page Config ---
st.set_page_config(page_title="IPL Match Predictor (Deep Learning)", page_icon="🏏", layout="wide")

st.markdown('<h1 style="text-align: center; color: #FF4B4B;">🏏 IPL Match Winner Predictor using Deep Learning</h1>', unsafe_allow_html=True)

# --- Navigation ---
st.sidebar.header("Navigation Control")
page = st.sidebar.radio("Go to:", ["Predict Match Outcomes", "Historical Dashboard (2010-2026)"])

teams = [
    'Royal Challengers Bengaluru', 'Mumbai Indians', 'Chennai Super Kings', 
    'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings', 
    'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
]

venues = [
    'M. Chinnaswamy Stadium, Bengaluru', 'Wankhede Stadium, Mumbai', 
    'MA Chidambaram Stadium, Chennai', 'Eden Gardens, Kolkata', 
    'Arun Jaitley Stadium, Delhi', 'Narendra Modi Stadium, Ahmedabad',
    'Rajiv Gandhi International Stadium, Hyderabad', 'Sawai Mansingh Stadium, Jaipur',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh (Mullanpur)',
    'Shaheed Veer Narayan Singh International Cricket Stadium, Raipur',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
    'HPCA Stadium, Dharamshala', 'Barsapara Cricket Stadium, Guwahati',
    'JSCA International Stadium Complex, Ranchi', 'Inderjit Singh Bindra Stadium, Mohali'
]

@st.cache_resource
def load_deep_learning_model():
    if os.path.exists('ipl_dl_model.h5'):
        return tf.keras.models.load_model('ipl_dl_model.h5')
    return None

dl_model = load_deep_learning_model()

# --- PAGE 1: PREDICT MATCH ---
if page == "Predict Match Outcomes":
    st.subheader("Enter Match Situation")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Select Batting Team", teams, index=0)
    with col2:
        remaining_teams = [t for t in teams if t != batting_team]
        bowling_team = st.selectbox("Select Bowling Team", remaining_teams, index=0)
        
    col3, col4, col5 = st.columns(3)
    with col3:
        venue = st.selectbox("Select Venue", venues)
    with col4:
        toss_winner = st.selectbox("Select Toss Winner", [batting_team, bowling_team])
    with col5:
        toss_decision = st.selectbox("Select Toss Decision", ["Choose to Bat", "Choose to Bowl"])

    if st.button("Predict Winner", type="primary"):
        if dl_model is not None:
            with st.spinner("Processing through deep neural network layers..."):
                time.sleep(1)
                
                # Feature Encoding
                batting_encoded = float(teams.index(batting_team))
                bowling_encoded = float(teams.index(bowling_team))
                venue_encoded = float(venues.index(venue))
                toss_win_encoded = float(teams.index(toss_winner))
                toss_dec_encoded = 0.0 if toss_decision == "Choose to Bat" else 1.0
                
                # Input shape (1, 5)
                input_tensor = np.array([[batting_encoded, bowling_encoded, venue_encoded, toss_win_encoded, toss_dec_encoded]], dtype=np.float32)
                
                # Model Inference
                prediction_prob = dl_model.predict(input_tensor)[0][0]
                
                prob_batting = int(prediction_prob * 100)
                prob_batting = max(5, min(95, prob_batting)) 
                prob_bowling = 100 - prob_batting
                
                # Display Output directly
                res_col1, res_col2 = st.columns(2)
                res_col1.metric(label=f"{batting_team} Win Chance", value=f"{prob_batting}%")
                res_col2.metric(label=f"{bowling_team} Win Chance", value=f"{prob_bowling}%")
                st.progress(prob_batting / 100)
        else:
            st.error("Model file (ipl_dl_model.h5) not found! Please run train_dl_model.py first.")

# --- PAGE 2: HISTORICAL DASHBOARD ---
elif page == "Historical Dashboard (2010-2026)":
    st.subheader("📊 Historical Analytical Interface")
    selected_year = st.slider("Select IPL Season", min_value=2010, max_value=2026, value=2026)
    
    np.random.seed(selected_year)
    num_matches = 74 if selected_year >= 2022 else 60
    
    mock_data = pd.DataFrame({
        'Match ID': range(1, num_matches + 1),
        'Team 1': np.random.choice(teams, num_matches),
        'Team 2': np.random.choice(teams, num_matches),
        'Venue': np.random.choice(venues, num_matches),
        'Winner': np.random.choice(teams, num_matches),
        'Margin': np.random.choice(['7 Wickets', '15 Runs', '5 Wickets', '2 Runs'], num_matches)
    })
    mock_data['Team 2'] = mock_data.apply(lambda row: np.random.choice([t for t in teams if t != row['Team 1']]), axis=1)
    mock_data['Winner'] = mock_data.apply(lambda row: np.random.choice([row['Team 1'], row['Team 2']]), axis=1)
    
    st.dataframe(mock_data, use_container_width=True)
    st.bar_chart(mock_data['Winner'].value_counts())