import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="Linear vs Logistic Regression Explorer", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tab-subheader {
        font-size: 28px !important;
        font-weight: bold;
        color: #8A2BE2;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .content-text {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #9370DB;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8A2BE2;
        transform: scale(1.05);
    }
    .highlight {
        background-color: #E6E6FA;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .quiz-question {
        background-color: #F0E6FA;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #8A2BE2;
    }
    .explanation {
        background-color: #E6F3FF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📊 Linear vs Logistic Regression Explorer 📊</h1>", unsafe_allow_html=True)
st.write('**Developed by : Venugopal Adep**')

# Functions
def generate_data(num_points, noise_level, logistic_x_shift):
    x = np.random.uniform(-5, 5, num_points)
    y_linear = 2*x + np.random.normal(0, noise_level, num_points)
    y_logistic = 1 / (1 + np.exp(-(x-logistic_x_shift)))
    y_logistic = np.where(np.random.rand(num_points) < y_logistic, 1, 0)
    return x, y_linear, y_logistic

def plot_data(x, y, mode, title, color, fit_line=None, fit_color=None, threshold=None):
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode=mode, marker=dict(color=color)))
    if fit_line is not None:
        fig.add_trace(go.Scatter(x=x, y=fit_line, mode='lines', line=dict(color=fit_color)))
    if threshold is not None:
        fig.add_shape(type='line', x0=x.min(), x1=x.max(), y0=threshold, y1=threshold,
                      line=dict(color='black', dash='dash'))
        fig.add_annotation(x=x.mean(), y=threshold, text=f'Decision Threshold: {threshold:.2f}',
                           showarrow=False, yshift=10)
    fig.update_layout(title=title, xaxis_title='Independent Variable', yaxis_title='Dependent Variable')
    return fig

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualization", "🧮 Interactive Calculator", "🎓 Learn More", "🧠 Quiz"])

with tab1:
    st.markdown("<h2 class='tab-subheader'>Linear and Logistic Regression Visualization</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<p class='content-text'>Adjust parameters to see how they affect the regressions:</p>", unsafe_allow_html=True)
        
        num_points = st.slider('Number of data points', 10, 100, 30)
        noise_level = st.slider('Noise level', 0.0, 2.0, 0.5)
        logistic_x_shift = st.slider('Logistic curve shift', -5.0, 5.0, 0.0)
        threshold = st.slider('Decision Threshold', 0.0, 1.0, 0.5)
        
        x, y_linear, y_logistic = generate_data(num_points, noise_level, logistic_x_shift)
        
    with col2:
        # Linear Regression
        p = np.polyfit(x, y_linear, 1)
        y_linear_pred = p[0]*x + p[1]
        fig_linear = plot_data(x, y_linear, 'markers', 'Linear Regression', 'blue', y_linear_pred, 'red')
        st.plotly_chart(fig_linear, use_container_width=True)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write(f"Linear Regression - Slope: {p[0]:.2f}, Intercept: {p[1]:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Logistic Regression
        model = LogisticRegression(random_state=0).fit(x.reshape(-1, 1), y_logistic)
        x_logistic_pred = np.linspace(x.min(), x.max(), 100)
        y_logistic_pred = model.predict_proba(x_logistic_pred.reshape(-1, 1))[:,1]
        fig_logistic = plot_data(x_logistic_pred, y_logistic_pred, 'lines', 'Logistic Regression', 'orange', threshold=threshold)
        fig_logistic.add_trace(go.Scatter(x=x, y=y_logistic, mode='markers', marker=dict(color='green')))
        st.plotly_chart(fig_logistic, use_container_width=True)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.write(f"Logistic Regression - Decision Threshold: {threshold:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='tab-subheader'>Interactive Regression Calculator</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<p class='content-text'>Enter your own data points:</p>", unsafe_allow_html=True)
        
        data_input = st.text_area("Enter x,y pairs (one per line):", "1,2\n2,4\n3,5\n4,4\n5,6")
        regression_type = st.radio("Select regression type:", ["Linear", "Logistic"])
        
        if st.button("Calculate Regression"):
            try:
                data = np.array([list(map(float, line.split(','))) for line in data_input.split('\n')])
                x = data[:, 0]
                y = data[:, 1]
                
                if regression_type == "Linear":
                    p = np.polyfit(x, y, 1)
                    y_pred = p[0]*x + p[1]
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    st.write(f"Linear Regression Results:")
                    st.write(f"Slope: {p[0]:.2f}")
                    st.write(f"Intercept: {p[1]:.2f}")
                    st.write(f"Equation: y = {p[0]:.2f}x + {p[1]:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    model = LogisticRegression(random_state=0).fit(x.reshape(-1, 1), y)
                    y_pred = model.predict_proba(x.reshape(-1, 1))[:,1]
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    st.write(f"Logistic Regression Results:")
                    st.write(f"Coefficient: {model.coef_[0][0]:.2f}")
                    st.write(f"Intercept: {model.intercept_[0]:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                fig = plot_data(x, y, 'markers', f'{regression_type} Regression', 'blue', y_pred, 'red')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab3:
    st.markdown("<h2 class='tab-subheader'>Learn More About Regression</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='content-text'>
    <b>Linear Regression:</b> Used when the dependent variable is continuous. It fits a straight line to model the relationship between the independent and dependent variables.
    
    <b>Example:</b> Predicting a student's test score based on the number of hours they studied.
    
    <b>Logistic Regression:</b> Used when the dependent variable is categorical, typically binary (0 or 1). It fits a sigmoidal curve to model the probability that the dependent variable equals 1, given the independent variable.
    
    <b>Example:</b> Predicting whether a customer will buy a product or not based on their age.
    
    Key differences:
    1. Output: Linear regression predicts continuous values, while logistic regression predicts probabilities.
    2. Function: Linear regression uses a linear equation, while logistic regression uses the logistic function.
    3. Assumptions: Linear regression assumes a linear relationship, while logistic regression doesn't.
    4. Applications: Linear regression is used for prediction and forecasting, while logistic regression is used for classification tasks.
    </p>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='tab-subheader'>Test Your Regression Knowledge 🧠</h2>", unsafe_allow_html=True)
    
    questions = [
        {
            "question": "Which type of regression would be most appropriate for predicting house prices based on square footage?",
            "options": ["Linear Regression", "Logistic Regression", "Both", "Neither"],
            "correct": 0,
            "explanation": "Linear regression is suitable for predicting continuous values like house prices based on features like square footage."
        },
        {
            "question": "In logistic regression, what does the output represent?",
            "options": ["Exact values", "Probabilities", "Categories", "Errors"],
            "correct": 1,
            "explanation": "Logistic regression outputs probabilities, typically representing the likelihood of an instance belonging to a particular class."
        },
        {
            "question": "Which regression type is more suitable for classifying emails as spam or not spam?",
            "options": ["Linear Regression", "Logistic Regression", "Both", "Neither"],
            "correct": 1,
            "explanation": "Logistic regression is ideal for binary classification tasks like spam detection, where the outcome is either spam (1) or not spam (0)."
        },
        {
            "question": "What shape does the logistic regression function typically have?",
            "options": ["Straight line", "Parabola", "S-curve (sigmoid)", "Circle"],
            "correct": 2,
            "explanation": "The logistic function has an S-shaped curve (sigmoid), which maps any input to a value between 0 and 1."
        }
    ]

    score = 0
    for i, q in enumerate(questions):
        st.markdown(f"<div class='quiz-question'>", unsafe_allow_html=True)
        st.markdown(f"<p class='content-text'><strong>Question {i+1}:</strong> {q['question']}</p>", unsafe_allow_html=True)
        user_answer = st.radio("Select your answer:", q['options'], key=f"q{i}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Check Answer", key=f"check{i}"):
                if q['options'].index(user_answer) == q['correct']:
                    st.success("Correct! 🎉")
                    score += 1
                else:
                    st.error("Incorrect. Try again! 🤔")
                st.markdown(f"<div class='explanation'><p>{q['explanation']}</p></div>", unsafe_allow_html=True)
        
        with col2:
            if i == 0:  # Example visualization for house prices
                x = np.array([1000, 1500, 2000, 2500, 3000])
                y = 100000 + 200 * x + np.random.normal(0, 10000, 5)
                fig = px.scatter(x=x, y=y, labels={'x': 'Square Footage', 'y': 'Price'})
                fig.add_trace(go.Scatter(x=x, y=100000 + 200 * x, mode='lines', name='Linear Regression'))
                fig.update_layout(title="House Price vs Square Footage")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 1:  # Logistic regression output visualization
                x = np.linspace(-5, 5, 100)
                y = 1 / (1 + np.exp(-x))
                fig = px.line(x=x, y=y, labels={'x': 'Input', 'y': 'Probability'})
                fig.update_layout(title="Logistic Regression Output")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 2:  # Spam classification visualization
                x = np.random.rand(100)
                y = (x > 0.5).astype(int)
                fig = px.scatter(x=x, y=y, labels={'x': 'Feature', 'y': 'Spam (1) or Not Spam (0)'})
                fig.update_layout(title="Email Spam Classification")
                st.plotly_chart(fig, use_container_width=True)
            elif i == 3:  # Sigmoid function visualization
                x = np.linspace(-10, 10, 100)
                y = 1 / (1 + np.exp(-x))
                fig = px.line(x=x, y=y, labels={'x': 'Input', 'y': 'Output'})
                fig.update_layout(title="Logistic (Sigmoid) Function")
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Show Final Score"):
        st.markdown(f"<p class='tab-subheader'>Your score: {score}/{len(questions)}</p>", unsafe_allow)