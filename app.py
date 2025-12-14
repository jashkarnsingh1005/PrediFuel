# import pandas as pd
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from sklearn.ensemble import RandomForestRegressor
# import pickle as pk
# import base64
# from streamlit_extras import add_vertical_space
# import streamlit.components.v1 as components
# from annotated_text import annotated_text
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import plotly.express as px
# import plotly.graph_objs as go
# from scipy import stats
# from sklearn.ensemble import RandomForestRegressor
# from src.Predictive_Maintenance.pipelines.prediction_pipeline import prediction
# import time
# import os
# import joblib
# import streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = "AIzaSyCRSQtxzUJ9fQY7GTuI5lrV2wGHLfKPo_A"
# genai.configure(api_key=GOOGLE_API_KEY)

# st.set_page_config(
#     page_title="DriveEco",  # Title of the app
#     page_icon=  "logo.webp"   ,    
#     layout="wide"           # Layout option (optional)
# )
# # Load pre-trained models and scaled data
# loaded_model = pk.load(open("trained_model_lr.sav", "rb"))
# scaled_data = pk.load(open("scaled_data.sav", "rb"))

# @st.cache_data
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Set background images and styles


# def input_converter(inp):
#     vcl = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 'SUV: Small', 'SUV: Standard', 'Minivan', 'Station wagon: Small', 'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle', 'Pickup truck: Standard']
#     trans = ['AV', 'AM', 'M', 'AS', 'A']
#     fuel = ["D", "E", "X", "Z"]
#     lst = []
#     for i in range(6):
#         if type(inp[i]) == str:
#             if inp[i] in vcl:
#                 lst.append(vcl.index(inp[i]))
#             elif inp[i] in trans:
#                 lst.append(trans.index(inp[i]))
#             elif inp[i] in fuel:
#                 if fuel.index(inp[i]) == 0:
#                     lst.extend([1, 0, 0, 0])
#                     break
#                 elif fuel.index(inp[i]) == 1:
#                     lst.extend([0, 1, 0, 0])
#                     break
#                 elif fuel.index(inp[i]) == 2:
#                     lst.extend([0, 0, 1, 0])
#                     break
#                 elif fuel.index(inp[i]) == 3:
#                     lst.extend([0, 0, 0, 1])
#         else:
#             lst.append(inp[i])

#     arr = np.asarray(lst)
#     arr = arr.reshape(1, -1)
#     arr = scaled_data.transform(arr)
#     prediction = loaded_model.predict(arr)

#     return f"The Fuel Consumption L/100km is {round(prediction[0], 2)}"



# def fuel_analysis(subpage):
#     # Main heading
#     st.markdown(
#         """
#         <h1 style='color: #d2b356; text-align: center; margin-bottom:50px;'>FUEL CONSUMPTION</h1>
#         """,
#         unsafe_allow_html=True
#     )

#     # Load dataset
#     df = pd.read_csv('final_data.csv')

#     # Prediction functionality
#     if subpage == "Prediction":
#         st.markdown(
#             """
#             <h1 style='color: white; font-size:25px; margin-top:70px;'>Prediction</h1>
#             """,
#             unsafe_allow_html=True
#         )

#         # Styling the selectbox and buttons
#         st.markdown(
#             """
#             <style>
#                 .stSelectbox > label,
#                 .stNumberInput > label {
#                     color: #d2b356;
#                 }
#                 .stButton>button {
#                     background-color: transparent;
#                     color: #d2b356;
#                     border: 2px solid #d2b356;
#                     border-radius: 4px;
#                     padding: 10px 20px;
#                     font-size: 16px;
#                     cursor: pointer;
#                 }
#                 .stButton>button:hover {
#                     background-color: #d2b356;
#                     color: white;
#                 }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )

#         # Vehicle options for prediction
#         vehicle = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 
#                    'SUV: Small', 'SUV: Standard', 'Minivan', 'Station wagon: Small', 
#                    'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle', 
#                    'Pickup truck: Standard']
#         transmission = ['AV', 'AM', 'M', 'AS', 'A']
#         fuel = ["D", "E", "X", "Z"]

#         # Prediction input widgets
#         Vehicle_class = st.selectbox("Enter Vehicle class", options=vehicle)
#         Engine_size = st.selectbox("Select Engine Size (please enter value in this range [1-7])", options=[1, 2, 3, 4, 5, 6, 7])
#         Cylinders = st.number_input("Enter number of Cylinders (please enter value in this range [1-16])", min_value=1, max_value=16)
#         Transmission = st.selectbox("Select the Transmission", transmission)
#         Co2_Rating = st.number_input("Enter CO2 Rating (please enter value in this range [1-10])", min_value=1, max_value=10)
#         Fuel_type = st.selectbox("Select the Fuel type", fuel)

#         # Session state to store and reset predictions
#         if 'prediction' not in st.session_state:
#             st.session_state.prediction = None

#         # Add predict and reset buttons
#         col1, col2, _ = st.columns([1, 1, 4])

#         with col1:
#             predict_button = st.button("Predict 🔍")
#         with col2:
#             reset_button = st.button("Reset ❌")

#         # Handle button clicks
#         if predict_button:
#             result = input_converter([Vehicle_class, Engine_size, Cylinders, Transmission, Co2_Rating, Fuel_type])
#             st.session_state.prediction = result  # Store prediction in session state

#         if reset_button:
#             st.session_state.prediction = None  # Reset the prediction

#         # Display the prediction result
#         if st.session_state.prediction:
#             st.markdown(f"<h2 style='color:white;'><b>{st.session_state.prediction}</b>!</h2>", unsafe_allow_html=True)

#     # Visualization functionality
#     elif subpage == "Visualization":
#         st.markdown(
#             "<h1 style='color: white; font-size:25px; margin-top:70px;'>Visualization</h1>",
#             unsafe_allow_html=True
#         )

#         # Sidebar filters
#         st.sidebar.header("Filter Options")
#         vehicle_class_filter = st.sidebar.multiselect("Select Vehicle Classes", df['Vehicle Class'].unique(), default=df['Vehicle Class'].unique())
#         transmission_filter = st.sidebar.multiselect("Select Transmissions", df['Transmission'].unique(), default=df['Transmission'].unique())
#         fuel_type_filter = st.sidebar.multiselect("Select Fuel Types", df['Fuel Type'].unique(), default=df['Fuel Type'].unique())

#         # Apply filters safely
#         df_filtered = df[
#             (df['Vehicle Class'].isin(vehicle_class_filter)) &
#             (df['Transmission'].isin(transmission_filter)) &
#             (df['Fuel Type'].isin(fuel_type_filter))
#         ]

#         if df_filtered.empty:
#             st.warning("No data available for the selected filters. Please adjust your selections.")
#             return

#         # Visualization options
#         viz_option = st.selectbox("Choose visualization type", [
#             "Transmission Distribution",
#             "Fuel Type Distribution",
#         ])

#         chart_type = st.selectbox("Choose chart type", ["Bar", "Pie"])

#         fig = None  # Initialize fig to avoid UnboundLocalError

#         # Visualization logic
#         if viz_option == "Transmission Distribution":
#             df_transmission = df_filtered['Transmission'].value_counts().reset_index()
#             df_transmission.columns = ['Transmission', 'Count']

#             if chart_type == "Bar":
#                 fig = px.bar(df_transmission, x='Transmission', y='Count', title='Transmission Distribution')
#             elif chart_type == "Pie":
#                 fig = px.pie(df_transmission, names='Transmission', values='Count', title='Transmission Distribution')

#         elif viz_option == "Fuel Type Distribution":
#             df_fuel_type = df_filtered['Fuel Type'].value_counts().reset_index()
#             df_fuel_type.columns = ['Fuel Type', 'Count']

#             if chart_type == "Bar":
#                 fig = px.bar(df_fuel_type, x='Fuel Type', y='Count', title='Fuel Type Distribution')
#             elif chart_type == "Pie":
#                 fig = px.pie(df_fuel_type, names='Fuel Type', values='Count', title='Fuel Type Distribution')



#         # Check if fig was created, otherwise show a warning
#         if fig:
#             st.plotly_chart(fig)
#         else:
#             st.warning("This visualization type is not supported for the selected chart type.")
            
# def co2_emission(subpage):
#     st.markdown(
#         """
#         <h1 style='color: #d2b356; text-align: center; margin-bottom:50px;'>CO2 EMISSION</h1>
#         """,
#         unsafe_allow_html=True
#     )

#     # Load the vehicle dataset
#     df = pd.read_csv('co2 Emissions.csv')

#     # Map fuel types and remove natural gas
#     fuel_type_mapping = {
#         "Z": "Premium Gasoline",
#         "X": "Regular Gasoline",
#         "D": "Diesel",
#         "E": "Ethanol(E85)",
#         "N": "Natural Gas"
#     }
    
#     df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
#     df = df[df["Fuel Type"] != "Natural Gas"].reset_index(drop=True)

#     # Remove outliers
#     df_new = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
#     df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]
    
#     if subpage == 'Visualization':
#         st.markdown(
#             """
#             <h1 style='color: white; font-size:25px; margin-top:70px;'>Visualization</h1>
#             """,
#             unsafe_allow_html=True
#         )
#         st.markdown("""
#             <style>
#             .stSelectbox label {
#                 color: #d2b356;
#             }
#             </style>
#             """, unsafe_allow_html=True)

#         # Sidebar filters
#         st.sidebar.header("Filter Options")
        
#         # Ensure required columns exist
#         required_columns = {'Fuel Type', 'Make', 'Vehicle Class', 'Transmission', 'CO2 Emissions(g/km)', 'Cylinders', 'Engine Size(L)'}
#         missing_columns = required_columns - set(df.columns)

#         if missing_columns:
#             st.error(f"Missing columns in dataset: {missing_columns}")
#             return
        
#         # Filtering options
#         fuel_types = df['Fuel Type'].unique()
#         selected_fuel_types = st.sidebar.multiselect('Select Fuel Type(s):', options=fuel_types, default=fuel_types)

#         makes = df['Make'].unique()
#         selected_makes = st.sidebar.multiselect('Select Car Make(s):', options=makes, default=makes)

#         vehicle_classes = df['Vehicle Class'].unique()
#         selected_vehicle_classes = st.sidebar.multiselect('Select Vehicle Class(es):', options=vehicle_classes, default=vehicle_classes)

#         transmissions = df['Transmission'].unique()
#         selected_transmissions = st.sidebar.multiselect('Select Transmission(s):', options=transmissions, default=transmissions)

#         # Apply filters
#         df_filtered = df[
#             (df['Fuel Type'].isin(selected_fuel_types)) &
#             (df['Make'].isin(selected_makes)) &
#             (df['Vehicle Class'].isin(selected_vehicle_classes)) &
#             (df['Transmission'].isin(selected_transmissions))
#         ]

#         if df_filtered.empty:
#             st.warning("No data available for the selected filters. Try modifying your selections.")
#             return

#         # Visualization type selector
#         visualization_type = st.selectbox("Choose Visualization Type:", [
#             "Car Companies Distribution",
#             "Top Car Models Distribution",
#             "Vehicle Classes Distribution",
#             "Engine Sizes Distribution",
#             "Cylinder Counts Distribution",
#             "Fuel Types Distribution",
#             "CO2 Emissions Distribution",
#             "CO2 Emissions by Make",
#             "CO2 Emissions by Vehicle Class",
#             "CO2 Emissions by Engine Size",
#             "CO2 Emissions by Cylinders",
#             "CO2 Emissions by Fuel Type"
#         ])

#         # Determine chart types
#         if visualization_type in ["Car Companies Distribution", "Top Car Models Distribution",
#                                   "Vehicle Classes Distribution", "Engine Sizes Distribution",
#                                   "Cylinder Counts Distribution", "Fuel Types Distribution"]:
#             chart_type = st.selectbox("Select chart type:", ["Bar", "Pie"])
#         elif visualization_type == "CO2 Emissions Distribution":
#             chart_type = st.selectbox("Select chart type:", ["Histogram", "Boxplot"])
#         else:
#             chart_type = st.selectbox("Select chart type:", ["Bar", "Line"])

#         # Generate plots
#         fig = None  # Ensure fig is always defined

#         if visualization_type == "Car Companies Distribution":
#             df_brand = df_filtered['Make'].value_counts().reset_index()
#             df_brand.columns = ['Make', 'Count']
#             fig = px.bar(df_brand, x='Make', y='Count', title='Car Companies Distribution') if chart_type == "Bar" else px.pie(df_brand, names='Make', values='Count', title='Car Companies Distribution')

#         elif visualization_type == "Top Car Models Distribution":
#             df_model = df_filtered['Model'].value_counts().reset_index().head(25)
#             df_model.columns = ['Model', 'Count']
#             fig = px.bar(df_model, x='Model', y='Count', title='Top 25 Car Models') if chart_type == "Bar" else px.pie(df_model, names='Model', values='Count', title='Top 25 Car Models')

#         elif visualization_type == "Vehicle Classes Distribution":
#             df_vehicle_class = df_filtered['Vehicle Class'].value_counts().reset_index()
#             df_vehicle_class.columns = ['Vehicle Class', 'Count']
#             fig = px.bar(df_vehicle_class, x='Vehicle Class', y='Count', title='Vehicle Classes Distribution') if chart_type == "Bar" else px.pie(df_vehicle_class, names='Vehicle Class', values='Count', title='Vehicle Classes Distribution')

#         elif visualization_type == "Engine Sizes Distribution":
#             df_engine_size = df_filtered['Engine Size(L)'].value_counts().reset_index()
#             df_engine_size.columns = ['Engine Size(L)', 'Count']
#             fig = px.bar(df_engine_size, x='Engine Size(L)', y='Count', title='Engine Sizes Distribution') if chart_type == "Bar" else px.pie(df_engine_size, names='Engine Size(L)', values='Count', title='Engine Sizes Distribution')

#         elif visualization_type == "Cylinder Counts Distribution":
#             df_cylinders = df_filtered['Cylinders'].value_counts().reset_index()
#             df_cylinders.columns = ['Cylinders', 'Count']
#             fig = px.bar(df_cylinders, x='Cylinders', y='Count', title='Cylinder Counts Distribution') if chart_type == "Bar" else px.pie(df_cylinders, names='Cylinders', values='Count', title='Cylinder Counts Distribution')

#         elif visualization_type == "Fuel Types Distribution":
#             df_fuel_type = df_filtered['Fuel Type'].value_counts().reset_index()
#             df_fuel_type.columns = ['Fuel Type', 'Count']
#             fig = px.bar(df_fuel_type, x='Fuel Type', y='Count', title='Fuel Types Distribution') if chart_type == "Bar" else px.pie(df_fuel_type, names='Fuel Type', values='Count', title='Fuel Types Distribution')

#         elif visualization_type == "CO2 Emissions Distribution":
#             fig = px.histogram(df_filtered, x='CO2 Emissions(g/km)', nbins=50, title='CO2 Emissions Distribution') if chart_type == "Histogram" else px.box(df_filtered, x='CO2 Emissions(g/km)', title='CO2 Emissions Boxplot')

#         elif visualization_type.startswith("CO2 Emissions by"):
#             group_column = visualization_type.split("by ")[1]
#             df_co2 = df_filtered.groupby(group_column)['CO2 Emissions(g/km)'].mean().reset_index()
#             fig = px.bar(df_co2, x=group_column, y='CO2 Emissions(g/km)', title=f'CO2 Emissions by {group_column}') if chart_type == "Bar" else px.line(df_co2, x=group_column, y='CO2 Emissions(g/km)', title=f'CO2 Emissions by {group_column}')

#         # Display the figure
#         if fig:
#             st.plotly_chart(fig)
#         else:
#             st.error("Failed to generate the chart. Please check the dataset and selected filters.")


#     elif subpage == 'Prediction':
#         X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
#         y = df_new_model['CO2 Emissions(g/km)']

#         # Train a RandomForest model
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X, y)

#         # Prediction section
#         st.markdown(
#             """
#             <style>
#             .slider .stSlider .st-bc {
#                 color: white;
#             }
#             .stSlider .st-bc {
#                 color: white;
#             }
#             </style>
#             <h1 style='color: white; font-size:25px; margin-top:70px;'>Predict CO2 Emission</h1>
#             """,
#             unsafe_allow_html=True
#         )
#         st.markdown("""<style>.stSelectbox label {color: #d2b356;}</style>""", unsafe_allow_html=True)
        
#         # Get user inputs for prediction
#         engine_size = st.slider('Engine Size (L)', min_value=float(X['Engine Size(L)'].min()), max_value=float(X['Engine Size(L)'].max()), value=float(X['Engine Size(L)'].mean()))
#         cylinders = st.slider('Cylinders', min_value=int(X['Cylinders'].min()), max_value=int(X['Cylinders'].max()), value=int(X['Cylinders'].mean()))
#         fuel_consumption = st.slider('Fuel Consumption Comb (L/100 km)', min_value=float(X['Fuel Consumption Comb (L/100 km)'].min()), max_value=float(X['Fuel Consumption Comb (L/100 km)'].max()), value=float(X['Fuel Consumption Comb (L/100 km)'].mean()))

#         # Make prediction
#         input_data = [[engine_size, cylinders, fuel_consumption]]
#         prediction = model.predict(input_data)

#         # Display the prediction
#         st.markdown(f"<p style='color: white;'>Predicted CO2 Emissions: {prediction[0]:.2f} g/km</p>", unsafe_allow_html=True)


# def about():
#     st.markdown("<h1 style='text-align: center; color: #d2b356; margin-bottom: 50px;'>ABOUT</h1>", unsafe_allow_html=True)
    
#     # Create two columns for the image and text
#     # col1 = st.columns([1])  # Adjust the proportions as needed
    
#     # with col1:
#     #     # Add image to the left column
#     #     st.image("images/fotor1.jpg", use_column_width=True)
    
#     # with col1:
#         # Add text and subheading to the right column
#     st.markdown("<h2 style='color: white;'>Predictive Analysis on Fuel Consumption, CO2 Emissions, and Engine Failure</h2>", unsafe_allow_html=True)
#     st.write("<p style='color: white;'>Our web application provides advanced tools for analyzing and predicting fuel consumption, CO2 emissions, and engine failure in vehicles.</p>", unsafe_allow_html=True)
        
#     st.markdown("<h3 style='color: white;'>Problem Statement</h3>", unsafe_allow_html=True)
#     st.write("<p style='color: white;'>In today's world, managing vehicle efficiency and minimizing environmental impact are crucial. High fuel consumption and CO2 emissions contribute significantly to environmental pollution, while unexpected engine failures can lead to costly repairs and downtime. Our application aims to address these issues by offering predictive analytics that help users make informed decisions to optimize vehicle performance and reduce emissions.</p>", unsafe_allow_html=True)
    
#     # Create two columns for "Our Solution" text and video
#     sol_col1, sol_col2 = st.columns([2, 1])  # Adjust proportions to fit text and video

#     with sol_col1:
#         # Add text for "Our Solution" to the left column
#         st.markdown("<h3 style='color: white; margin-top: 50px;'>Our Solution</h3>", unsafe_allow_html=True)
#         st.write("""
#         <p style='color: white;'>Our tool leverages machine learning models and advanced data analysis techniques to predict fuel consumption patterns, estimate CO2 emissions, and anticipate potential engine failures. By providing these insights, we enable users to improve fuel efficiency, reduce their carbon footprint, and proactively maintain their vehicles.</p>

#         <h4 style='color: white;'>1. Import Modules</h4>
#         <p style='color: white;'>1.1 Import data into DataFrame</p>
        
#         <h4 style='color: white;'>2. Data Cleaning</h4>
#         <p style='color: white;'>2.1 Creating a new DataFrame with necessary columns.<br>
#         2.2 Checking for null values.<br>

#         <h4 style='color: white;'>3. Exploratory Data Analysis (E.D.A.)</h4>
#         <h5 style='color: white;'>Univariate Analysis:</h5>
#         <p style='color: white;'>3.1 Checking mean, median, standard deviation, and quantiles.<br>
#         3.2 Frequency distribution for categorical columns.<br>
#         3.3 Frequency distribution for numerical columns.</p>
    
#         """, unsafe_allow_html=True)

#     with sol_col2:
#         st.markdown("<h3 style='color: white; margin-top: 180px;'></h3>", unsafe_allow_html=True)
#         st.write("""

#         <h5 style='color: white;'>4. Bivariate Analysis:</h5>
#         <p style='color: white;'>4.1 Comparing all columns.<br>

#         <h4 style='color: white;'>5. Data Preprocessing</h4>
#         <p style='color: white;'>5.1 Outlier analysis.<br>
#         5.2 Ordinal encoding.<br>
#         5.3 One-Hot encoding on categorical columns.<br>
#         5.4 Splitting data into train and test sets.<br>
#         5.5 Feature scaling.</p>
        
#         <h4 style='color: white;'>6. Training Models</h4>
        
#         <h4 style='color: white;'>7. Model Deployment</h4>
#         """, unsafe_allow_html=True)



#     # Create a section for team members
#     # st.markdown("<h2 style='text-align: center; color: #d2b356; margin-top: 50px;'>Meet Our Team</h2>", unsafe_allow_html=True)

# def home_page():
#     # Convert local image to base64
#     img_file = 'images/fotor.jpg'  # Update this with the correct path
#     img_base64 = get_img_as_base64(img_file)

#     # Convert icon images to base64
#     icon1 = get_img_as_base64("images/icon1.png")  # Update with the correct path
#     icon2 = get_img_as_base64("images/icon2.png")  # Update with the correct path
#     icon3 = get_img_as_base64("images/icon3.png")  # Update with the correct path
#     icon4 = get_img_as_base64("images/icon4.png")  # Update with the correct path
#     icon5 = get_img_as_base64("images/straight.png")  
#     st.markdown(f"""
#     <style>
#         .home-container {{
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             height: 80vh;
#         }}
#         .home-text {{
#             flex: 1;
#             text-align: left;
#         }}
#         .home-image {{
#             flex: 1;
#             padding: 20px;
#         }}
#         .home-image img {{
#             width: 800px;
#             height: 500px;
#         }}
#         .icon-row {{
#             display: flex;
#             justify-content: space-around;
#             margin-top: 20px;
#             padding-top: 50px; /* Adds space between the image and icons */
#             padding-bottom: 30px;
#         }}
#         .icon-row img {{
#             width: 80px;
#             height: 80px;
#         }}
#         .icon-inline {{
#             vertical-align: middle;
#             margin-bottom: 15px;
#         }}
#         .icon-bottom {{
#             display: block;
#             margin-top: 20px;
#             text-align: center;
#         }}
#         .button-left {{
#             background-color: transparent; /* Green */
#             border: 2px solid #d2b356 !important;
#             color: #d2b356;
#             padding: 10px 20px; /* Adjust padding as needed */
#             text-align: left;
#             text-decoration: none;
#             display: inline-block;
#             font-size: 16px;
#             margin: 4px 2px;
#             cursor: pointer;
#             border-radius: 4px;
#             float: left; /* Aligns the button to the left */
#             margin-right: 20px; 
#             margin-top: 60px;
#         }}
#     </style>
#     <div class="home-container">
#         <div class="home-text">
#             <img src="data:image/png;base64,{icon5}" class="icon-inline" alt="Icon Above Heading">
#             <h1 style="color: #FFFFFF; font-size:30px; padding-top:60px;">
#                 <span style="color: #d2b356; font-size:55px;">From data to decisions: </span> 
#                 Optimize fuel, cut emissions, ensure engine health.
#             </h1>
#             <div style="text-align: left; margin-top: 20px;">
#                 <a href="?page=About">
#                     <button class="button-left">
#                         Learn More
#                     </button>
#                 </a>
#             </div>
#             <p style="color: whitesmoke; padding-top: 130px;">
#                 DriveEco is a cutting-edge predictive analytics platform offering deep insights into fuel consumption, CO2 emissions, and engine failure. 
#                 With advanced machine learning algorithms and real-time data visualization, it enables users to monitor fuel usage, forecast environmental impact, 
#                 and anticipate engine issues. Ideal for both fleet management and individual vehicles, DriveEco excels in proactive maintenance and environmental stewardship.
#             </p>
#             <img src="data:image/png;base64,{icon5}" class="icon-bottom" alt="Icon Below Info">
#         </div>
#         <div class="home-image">
#             <img src="data:image/jpeg;base64,{img_base64}" alt="Home Image">
#         </div>
#     </div>
#     <div class="icon-row">
#         <img src="data:image/png;base64,{icon3}" alt="Icon 1">
#         <img src="data:image/png;base64,{icon4}" alt="Icon 2">
#         <img src="data:image/png;base64,{icon1}" alt="Icon 3">
#         <img src="data:image/png;base64,{icon2}" alt="Icon 4">
#     </div>
#     """, unsafe_allow_html=True)




# # Integrate this into your existing main function
# def main():
#     logo_file = "logo.webp"  # Update this with the path to your logo file
#     logo_base64 = get_img_as_base64(logo_file)

#     # Add the logo and background to the sidebar
#     st.sidebar.markdown(f"""
#     <style>
#     .sidebar {{
#         background-image: url('data:image/png;base64,{logo_base64}'); /* Use your desired image URL */
#         background-size: cover; /* Cover the entire sidebar */
#         background-position: center;
#         background-repeat: no-repeat;
#         height: 100vh; /* Full height for the sidebar */
#         padding: 20px; /* Add padding if needed */
#         box-sizing: border-box; /* Ensure padding is included in total width/height */
#     }}
#     .sidebar-logo {{
#         display: flex;
#         justify-content: center;
#         margin-bottom: 20px;
#     }}
#     .sidebar-logo img {{
#         width: 170px;
#         max-width: 100px;  /* Adjust this value as needed */
#     }}
#     </style>
#     <div class="sidebar-logo">
#         <img src="data:image/png;base64,{logo_base64}" alt="Logo">
#     </div>
#     """, unsafe_allow_html=True)

#     st.sidebar.title("PrediFuel")

#     # Main navbar radio button
#     page = st.sidebar.radio("", ["HOME", "ABOUT", "FUEL ANALYSIS", "CO2 EMISSION"])

#     if page == "HOME":
#         home_page()
    
#     elif page == "FUEL ANALYSIS":
#         subpage = st.sidebar.radio("FUEL ANALYSIS", ["Visualization", "Prediction"])
#         fuel_analysis(subpage)

#     elif page == "CO2 EMISSION":
#         subpage = st.sidebar.radio("CO2 EMISSION", ["Visualization", "Prediction"])
#         co2_emission(subpage)

#     elif page == "ABOUT":
#         about()


# if __name__ == "__main__":
#     main()














import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import pickle as pk
import base64
from streamlit_extras import add_vertical_space
import streamlit.components.v1 as components
from annotated_text import annotated_text
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from src.Predictive_Maintenance.pipelines.prediction_pipeline import prediction
import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# GOOGLE_API_KEY = "AIzaSyCRSQtxzUJ9fQY7GTuI5lrV2wGHLfKPo_A"
GOOGLE_API_KEY = "AIzaSyC2i5MlldsCBQjPeG6Nb3v4ojUSQP1U4To"
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="DriveEco",  # Title of the app
    page_icon=  "logo.webp"   ,    
    layout="wide"           # Layout option (optional)
)
# Load pre-trained models and scaled data
loaded_model = pk.load(open("D:\\sem4pbl\\FUEL_CONSUMPTION_ANALYSIS-main\\trained_model_lr.sav", "rb"))
scaled_data = pk.load(open("D:\\sem4pbl\\FUEL_CONSUMPTION_ANALYSIS-main\\scaled_data.sav", "rb"))

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set background images and styles


def input_converter(inp):
    vcl = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 'SUV: Small', 'SUV: Standard', 'Minivan', 'Station wagon: Small', 'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle', 'Pickup truck: Standard']
    trans = ['AV', 'AM', 'M', 'AS', 'A']
    fuel = ["D", "E", "X", "Z"]
    lst = []
    for i in range(6):
        if type(inp[i]) == str:
            if inp[i] in vcl:
                lst.append(vcl.index(inp[i]))
            elif inp[i] in trans:
                lst.append(trans.index(inp[i]))
            elif inp[i] in fuel:
                if fuel.index(inp[i]) == 0:
                    lst.extend([1, 0, 0, 0])
                    break
                elif fuel.index(inp[i]) == 1:
                    lst.extend([0, 1, 0, 0])
                    break
                elif fuel.index(inp[i]) == 2:
                    lst.extend([0, 0, 1, 0])
                    break
                elif fuel.index(inp[i]) == 3:
                    lst.extend([0, 0, 0, 1])
        else:
            lst.append(inp[i])

    arr = np.asarray(lst)
    arr = arr.reshape(1, -1)
    arr = scaled_data.transform(arr)
    prediction = loaded_model.predict(arr)

    return f"The Fuel Consumption L/100km is {round(prediction[0], 2)}"



def fuel_analysis(subpage):
    # Main heading
    st.markdown(
        """
        <h1 style='color: #d2b356; text-align: center; margin-bottom:50px;'>FUEL CONSUMPTION</h1>
        """,
        unsafe_allow_html=True
    )

    # Load dataset
    df = pd.read_csv('final_data.csv')

    # Prediction functionality
    if subpage == "Prediction":
        st.markdown(
            """
            <h1 style='color: white; font-size:25px; margin-top:70px;'>Prediction</h1>
            """,
            unsafe_allow_html=True
        )

        # Styling the selectbox and buttons
        st.markdown(
            """
            <style>
                .stSelectbox > label,
                .stNumberInput > label {
                    color: #d2b356;
                }
                .stButton>button {
                    background-color: transparent;
                    color: #d2b356;
                    border: 2px solid #d2b356;
                    border-radius: 4px;
                    padding: 10px 20px;
                    font-size: 16px;
                    cursor: pointer;
                }
                .stButton>button:hover {
                    background-color: #d2b356;
                    color: white;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Vehicle options for prediction
        vehicle = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 
                   'SUV: Small', 'SUV: Standard', 'Minivan', 'Station wagon: Small', 
                   'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle', 
                   'Pickup truck: Standard']
        transmission = ['AV', 'AM', 'M', 'AS', 'A']
        fuel = ["D", "E", "X", "Z"]

        # Prediction input widgets
        Vehicle_class = st.selectbox("Enter Vehicle class", options=vehicle)
        Engine_size = st.selectbox("Select Engine Size (please enter value in this range [1-7])", options=[1, 2, 3, 4, 5, 6, 7])
        Cylinders = st.number_input("Enter number of Cylinders (please enter value in this range [1-16])", min_value=1, max_value=16)
        Transmission = st.selectbox("Select the Transmission", transmission)
        Co2_Rating = st.number_input("Enter CO2 Rating (please enter value in this range [1-10])", min_value=1, max_value=10)
        Fuel_type = st.selectbox("Select the Fuel type", fuel)

        # Session state to store and reset predictions
        if 'prediction' not in st.session_state:
            st.session_state.prediction = None

        # Add predict and reset buttons
        col1, col2, _ = st.columns([1, 1, 4])

        with col1:
            predict_button = st.button("Predict 🔍")
        with col2:
            reset_button = st.button("Reset ❌")

        # Handle button clicks
        if predict_button:
            result = input_converter([Vehicle_class, Engine_size, Cylinders, Transmission, Co2_Rating, Fuel_type])
            st.session_state.prediction = result  # Store prediction in session state

        if reset_button:
            st.session_state.prediction = None  # Reset the prediction

        # Display the prediction result
        if st.session_state.prediction:
            st.markdown(f"<h2 style='color:white;'><b>{st.session_state.prediction}</b>!</h2>", unsafe_allow_html=True)

    # Visualization functionality
    elif subpage == "Visualization":
        st.markdown(
            "<h1 style='color: white; font-size:25px; margin-top:70px;'>Visualization</h1>",
            unsafe_allow_html=True
        )

        # Sidebar filters
        st.sidebar.header("Filter Options")
        vehicle_class_filter = st.sidebar.multiselect("Select Vehicle Classes", df['Vehicle Class'].unique(), default=df['Vehicle Class'].unique())
        transmission_filter = st.sidebar.multiselect("Select Transmissions", df['Transmission'].unique(), default=df['Transmission'].unique())
        fuel_type_filter = st.sidebar.multiselect("Select Fuel Types", df['Fuel Type'].unique(), default=df['Fuel Type'].unique())

        # Apply filters safely
        df_filtered = df[
            (df['Vehicle Class'].isin(vehicle_class_filter)) &
            (df['Transmission'].isin(transmission_filter)) &
            (df['Fuel Type'].isin(fuel_type_filter))
        ]

        if df_filtered.empty:
            st.warning("No data available for the selected filters. Please adjust your selections.")
            return

        # Visualization options
        viz_option = st.selectbox("Choose visualization type", [
            "Transmission Distribution",
            "Fuel Type Distribution",
        ])

        chart_type = st.selectbox("Choose chart type", ["Bar", "Pie"])

        fig = None  # Initialize fig to avoid UnboundLocalError

        # Visualization logic
        if viz_option == "Transmission Distribution":
            df_transmission = df_filtered['Transmission'].value_counts().reset_index()
            df_transmission.columns = ['Transmission', 'Count']

            if chart_type == "Bar":
                fig = px.bar(df_transmission, x='Transmission', y='Count', title='Transmission Distribution')
            elif chart_type == "Pie":
                fig = px.pie(df_transmission, names='Transmission', values='Count', title='Transmission Distribution')

        elif viz_option == "Fuel Type Distribution":
            df_fuel_type = df_filtered['Fuel Type'].value_counts().reset_index()
            df_fuel_type.columns = ['Fuel Type', 'Count']

            if chart_type == "Bar":
                fig = px.bar(df_fuel_type, x='Fuel Type', y='Count', title='Fuel Type Distribution')
            elif chart_type == "Pie":
                fig = px.pie(df_fuel_type, names='Fuel Type', values='Count', title='Fuel Type Distribution')



        # Check if fig was created, otherwise show a warning
        if fig:
            st.plotly_chart(fig)
        else:
            st.warning("This visualization type is not supported for the selected chart type.")
            
def co2_emission(subpage):
    st.markdown(
        """
        <h1 style='color: #d2b356; text-align: center; margin-bottom:50px;'>CO2 EMISSION</h1>
        """,
        unsafe_allow_html=True
    )

    # Load the vehicle dataset
    df = pd.read_csv('co2 Emissions.csv')

    # Map fuel types and remove natural gas
    fuel_type_mapping = {
        "Z": "Premium Gasoline",
        "X": "Regular Gasoline",
        "D": "Diesel",
        "E": "Ethanol(E85)",
        "N": "Natural Gas"
    }
    
    df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
    df = df[df["Fuel Type"] != "Natural Gas"].reset_index(drop=True)

    # Remove outliers
    df_new = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
    df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]
    
    if subpage == 'Visualization':
        st.markdown(
            """
            <h1 style='color: white; font-size:25px; margin-top:70px;'>Visualization</h1>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
            <style>
            .stSelectbox label {
                color: #d2b356;
            }
            </style>
            """, unsafe_allow_html=True)

        # Sidebar filters
        st.sidebar.header("Filter Options")
        
        # Ensure required columns exist
        required_columns = {'Fuel Type', 'Make', 'Vehicle Class', 'Transmission', 'CO2 Emissions(g/km)', 'Cylinders', 'Engine Size(L)'}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            st.error(f"Missing columns in dataset: {missing_columns}")
            return
        
        # Filtering options
        fuel_types = df['Fuel Type'].unique()
        selected_fuel_types = st.sidebar.multiselect('Select Fuel Type(s):', options=fuel_types, default=fuel_types)

        makes = df['Make'].unique()
        selected_makes = st.sidebar.multiselect('Select Car Make(s):', options=makes, default=makes)

        vehicle_classes = df['Vehicle Class'].unique()
        selected_vehicle_classes = st.sidebar.multiselect('Select Vehicle Class(es):', options=vehicle_classes, default=vehicle_classes)

        transmissions = df['Transmission'].unique()
        selected_transmissions = st.sidebar.multiselect('Select Transmission(s):', options=transmissions, default=transmissions)

        # Apply filters
        df_filtered = df[
            (df['Fuel Type'].isin(selected_fuel_types)) &
            (df['Make'].isin(selected_makes)) &
            (df['Vehicle Class'].isin(selected_vehicle_classes)) &
            (df['Transmission'].isin(selected_transmissions))
        ]

        if df_filtered.empty:
            st.warning("No data available for the selected filters. Try modifying your selections.")
            return

        # Visualization type selector
        visualization_type = st.selectbox("Choose Visualization Type:", [
            "Car Companies Distribution",
            "Top Car Models Distribution",
            "Vehicle Classes Distribution",
            "Engine Sizes Distribution",
            "Cylinder Counts Distribution",
            "Fuel Types Distribution",
            "CO2 Emissions Distribution",
            "CO2 Emissions by Make",
            "CO2 Emissions by Vehicle Class",
            "CO2 Emissions by Engine Size",
            "CO2 Emissions by Cylinders",
            "CO2 Emissions by Fuel Type"
        ])

        # Determine chart types
        if visualization_type in ["Car Companies Distribution", "Top Car Models Distribution",
                                  "Vehicle Classes Distribution", "Engine Sizes Distribution",
                                  "Cylinder Counts Distribution", "Fuel Types Distribution"]:
            chart_type = st.selectbox("Select chart type:", ["Bar", "Pie"])
        elif visualization_type == "CO2 Emissions Distribution":
            chart_type = st.selectbox("Select chart type:", ["Histogram", "Boxplot"])
        else:
            chart_type = st.selectbox("Select chart type:", ["Bar", "Line"])

        # Generate plots
        fig = None  # Ensure fig is always defined

        if visualization_type == "Car Companies Distribution":
            df_brand = df_filtered['Make'].value_counts().reset_index()
            df_brand.columns = ['Make', 'Count']
            fig = px.bar(df_brand, x='Make', y='Count', title='Car Companies Distribution') if chart_type == "Bar" else px.pie(df_brand, names='Make', values='Count', title='Car Companies Distribution')

        elif visualization_type == "Top Car Models Distribution":
            df_model = df_filtered['Model'].value_counts().reset_index().head(25)
            df_model.columns = ['Model', 'Count']
            fig = px.bar(df_model, x='Model', y='Count', title='Top 25 Car Models') if chart_type == "Bar" else px.pie(df_model, names='Model', values='Count', title='Top 25 Car Models')

        elif visualization_type == "Vehicle Classes Distribution":
            df_vehicle_class = df_filtered['Vehicle Class'].value_counts().reset_index()
            df_vehicle_class.columns = ['Vehicle Class', 'Count']
            fig = px.bar(df_vehicle_class, x='Vehicle Class', y='Count', title='Vehicle Classes Distribution') if chart_type == "Bar" else px.pie(df_vehicle_class, names='Vehicle Class', values='Count', title='Vehicle Classes Distribution')

        elif visualization_type == "Engine Sizes Distribution":
            df_engine_size = df_filtered['Engine Size(L)'].value_counts().reset_index()
            df_engine_size.columns = ['Engine Size(L)', 'Count']
            fig = px.bar(df_engine_size, x='Engine Size(L)', y='Count', title='Engine Sizes Distribution') if chart_type == "Bar" else px.pie(df_engine_size, names='Engine Size(L)', values='Count', title='Engine Sizes Distribution')

        elif visualization_type == "Cylinder Counts Distribution":
            df_cylinders = df_filtered['Cylinders'].value_counts().reset_index()
            df_cylinders.columns = ['Cylinders', 'Count']
            fig = px.bar(df_cylinders, x='Cylinders', y='Count', title='Cylinder Counts Distribution') if chart_type == "Bar" else px.pie(df_cylinders, names='Cylinders', values='Count', title='Cylinder Counts Distribution')

        elif visualization_type == "Fuel Types Distribution":
            df_fuel_type = df_filtered['Fuel Type'].value_counts().reset_index()
            df_fuel_type.columns = ['Fuel Type', 'Count']
            fig = px.bar(df_fuel_type, x='Fuel Type', y='Count', title='Fuel Types Distribution') if chart_type == "Bar" else px.pie(df_fuel_type, names='Fuel Type', values='Count', title='Fuel Types Distribution')

        elif visualization_type == "CO2 Emissions Distribution":
            fig = px.histogram(df_filtered, x='CO2 Emissions(g/km)', nbins=50, title='CO2 Emissions Distribution') if chart_type == "Histogram" else px.box(df_filtered, x='CO2 Emissions(g/km)', title='CO2 Emissions Boxplot')

        elif visualization_type.startswith("CO2 Emissions by"):
            group_column = visualization_type.split("by ")[1]
            df_co2 = df_filtered.groupby(group_column)['CO2 Emissions(g/km)'].mean().reset_index()
            fig = px.bar(df_co2, x=group_column, y='CO2 Emissions(g/km)', title=f'CO2 Emissions by {group_column}') if chart_type == "Bar" else px.line(df_co2, x=group_column, y='CO2 Emissions(g/km)', title=f'CO2 Emissions by {group_column}')

        # Display the figure
        if fig:
            st.plotly_chart(fig)
        else:
            st.error("Failed to generate the chart. Please check the dataset and selected filters.")


    elif subpage == 'Prediction':
        X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
        y = df_new_model['CO2 Emissions(g/km)']

        # Train a RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Prediction section
        st.markdown(
            """
            <style>
            .slider .stSlider .st-bc {
                color: white;
            }
            .stSlider .st-bc {
                color: white;
            }
            </style>
            <h1 style='color: white; font-size:25px; margin-top:70px;'>Predict CO2 Emission</h1>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""<style>.stSelectbox label {color: #d2b356;}</style>""", unsafe_allow_html=True)
        
        # Get user inputs for prediction
        engine_size = st.slider('Engine Size (L)', min_value=float(X['Engine Size(L)'].min()), max_value=float(X['Engine Size(L)'].max()), value=float(X['Engine Size(L)'].mean()))
        cylinders = st.slider('Cylinders', min_value=int(X['Cylinders'].min()), max_value=int(X['Cylinders'].max()), value=int(X['Cylinders'].mean()))
        fuel_consumption = st.slider('Fuel Consumption Comb (L/100 km)', min_value=float(X['Fuel Consumption Comb (L/100 km)'].min()), max_value=float(X['Fuel Consumption Comb (L/100 km)'].max()), value=float(X['Fuel Consumption Comb (L/100 km)'].mean()))

        # Make prediction
        input_data = [[engine_size, cylinders, fuel_consumption]]
        prediction = model.predict(input_data)

        # Display the prediction
        st.markdown(f"<p style='color: white;'>Predicted CO2 Emissions: {prediction[0]:.2f} g/km</p>", unsafe_allow_html=True)


def about():
    st.markdown("<h1 style='text-align: center; color: #d2b356; margin-bottom: 50px;'>ABOUT</h1>", unsafe_allow_html=True)
    
    # Create two columns for the image and text
    # col1 = st.columns([1])  # Adjust the proportions as needed
    
    # with col1:
    #     # Add image to the left column
    #     st.image("images/fotor1.jpg", use_column_width=True)
    
    # with col1:
        # Add text and subheading to the right column
    st.markdown("<h2 style='color: white;'>Predictive Analysis on Fuel Consumption, CO2 Emissions, and Engine Failure</h2>", unsafe_allow_html=True)
    st.write("<p style='color: white;'>Our web application provides advanced tools for analyzing and predicting fuel consumption, CO2 emissions, and engine failure in vehicles.</p>", unsafe_allow_html=True)
        
    st.markdown("<h3 style='color: white;'>Problem Statement</h3>", unsafe_allow_html=True)
    st.write("<p style='color: white;'>In today's world, managing vehicle efficiency and minimizing environmental impact are crucial. High fuel consumption and CO2 emissions contribute significantly to environmental pollution, while unexpected engine failures can lead to costly repairs and downtime. Our application aims to address these issues by offering predictive analytics that help users make informed decisions to optimize vehicle performance and reduce emissions.</p>", unsafe_allow_html=True)
    
    # Create two columns for "Our Solution" text and video
    sol_col1, sol_col2 = st.columns([2, 1])  # Adjust proportions to fit text and video

    with sol_col1:
        # Add text for "Our Solution" to the left column
        st.markdown("<h3 style='color: white; margin-top: 50px;'>Our Solution</h3>", unsafe_allow_html=True)
        st.write("""
        <p style='color: white;'>Our tool leverages machine learning models and advanced data analysis techniques to predict fuel consumption patterns, estimate CO2 emissions, and anticipate potential engine failures. By providing these insights, we enable users to improve fuel efficiency, reduce their carbon footprint, and proactively maintain their vehicles.</p>

        <h4 style='color: white;'>1. Import Modules</h4>
        <p style='color: white;'>1.1 Import data into DataFrame</p>
        
        <h4 style='color: white;'>2. Data Cleaning</h4>
        <p style='color: white;'>2.1 Creating a new DataFrame with necessary columns.<br>
        2.2 Checking for null values.<br>

        <h4 style='color: white;'>3. Exploratory Data Analysis (E.D.A.)</h4>
        <h5 style='color: white;'>Univariate Analysis:</h5>
        <p style='color: white;'>3.1 Checking mean, median, standard deviation, and quantiles.<br>
        3.2 Frequency distribution for categorical columns.<br>
        3.3 Frequency distribution for numerical columns.</p>
    
        """, unsafe_allow_html=True)

    with sol_col2:
        st.markdown("<h3 style='color: white; margin-top: 180px;'></h3>", unsafe_allow_html=True)
        st.write("""

        <h5 style='color: white;'>4. Bivariate Analysis:</h5>
        <p style='color: white;'>4.1 Comparing all columns.<br>

        <h4 style='color: white;'>5. Data Preprocessing</h4>
        <p style='color: white;'>5.1 Outlier analysis.<br>
        5.2 Ordinal encoding.<br>
        5.3 One-Hot encoding on categorical columns.<br>
        5.4 Splitting data into train and test sets.<br>
        5.5 Feature scaling.</p>
        
        <h4 style='color: white;'>6. Training Models</h4>
        
        <h4 style='color: white;'>7. Model Deployment</h4>
        """, unsafe_allow_html=True)



    # Create a section for team members
    # st.markdown("<h2 style='text-align: center; color: #d2b356; margin-top: 50px;'>Meet Our Team</h2>", unsafe_allow_html=True)

def home_page():
    # Convert local image to base64
    img_file = 'images/fotor.jpg'  # Update this with the correct path
    img_base64 = get_img_as_base64(img_file)

    # Convert icon images to base64
    icon1 = get_img_as_base64("images/icon1.png")  # Update with the correct path
    icon2 = get_img_as_base64("images/icon2.png")  # Update with the correct path
    icon3 = get_img_as_base64("images/icon3.png")  # Update with the correct path
    icon4 = get_img_as_base64("images/icon4.png")  # Update with the correct path
    icon5 = get_img_as_base64("images/straight.png")  
    st.markdown(f"""
    <style>
        .home-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 80vh;
        }}
        .home-text {{
            flex: 1;
            text-align: left;
        }}
        .home-image {{
            flex: 1;
            padding: 20px;
        }}
        .home-image img {{
            width: 800px;
            height: 500px;
        }}
        .icon-row {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding-top: 50px; /* Adds space between the image and icons */
            padding-bottom: 30px;
        }}
        .icon-row img {{
            width: 80px;
            height: 80px;
        }}
        .icon-inline {{
            vertical-align: middle;
            margin-bottom: 15px;
        }}
        .icon-bottom {{
            display: block;
            margin-top: 20px;
            text-align: center;
        }}
        .button-left {{
            background-color: transparent; /* Green */
            border: 2px solid #d2b356 !important;
            color: #d2b356;
            padding: 10px 20px; /* Adjust padding as needed */
            text-align: left;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            float: left; /* Aligns the button to the left */
            margin-right: 20px; 
            margin-top: 60px;
        }}
    </style>
    <div class="home-container">
        <div class="home-text">
            <img src="data:image/png;base64,{icon5}" class="icon-inline" alt="Icon Above Heading">
            <h1 style="color: #FFFFFF; font-size:30px; padding-top:60px;">
                <span style="color: #d2b356; font-size:55px;">From data to decisions: </span> 
                Optimize fuel, cut emissions, ensure engine health.
            </h1>
            <div style="text-align: left; margin-top: 20px;">
                <a href="?page=About">
                    <button class="button-left">
                        Learn More
                    </button>
                </a>
            </div>
            <p style="color: whitesmoke; padding-top: 130px;">
                DriveEco is a cutting-edge predictive analytics platform offering deep insights into fuel consumption, CO2 emissions, and engine failure. 
                With advanced machine learning algorithms and real-time data visualization, it enables users to monitor fuel usage, forecast environmental impact, 
                and anticipate engine issues. Ideal for both fleet management and individual vehicles, DriveEco excels in proactive maintenance and environmental stewardship.
            </p>
            <img src="data:image/png;base64,{icon5}" class="icon-bottom" alt="Icon Below Info">
        </div>
        <div class="home-image">
            <img src="data:image/jpeg;base64,{img_base64}" alt="Home Image">
        </div>
    </div>
    <div class="icon-row">
        <img src="data:image/png;base64,{icon3}" alt="Icon 1">
        <img src="data:image/png;base64,{icon4}" alt="Icon 2">
        <img src="data:image/png;base64,{icon1}" alt="Icon 3">
        <img src="data:image/png;base64,{icon2}" alt="Icon 4">
    </div>
    """, unsafe_allow_html=True)






def query_page():
    st.title("QUERY AREA")
    
    new_chat_id = f'{time.time()}'
    MODEL_ROLE = 'ai'
    AI_AVATAR_ICON = '✨'

    # Create a data/ folder if it doesn't already exist
    os.makedirs('data/', exist_ok=True)

    # Load past chats (if available)
    try:
        past_chats: dict = joblib.load('data/past_chats_list')
    except:
        past_chats = {}

    # Sidebar allows a list of past chats
    with st.sidebar:
        st.write('# Past Chats')
        if st.session_state.get('chat_id') is None:
            st.session_state.chat_id = st.selectbox(
                label='Pick a past chat',
                options=[new_chat_id] + list(past_chats.keys()),
                format_func=lambda x: past_chats.get(x, 'New Chat'),
                placeholder='_',
            )
        else:
            st.session_state.chat_id = st.selectbox(
                label='Pick a past chat',
                options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
                index=1,
                format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
                placeholder='_',
            )
        # Save new chats after a message has been sent to AI
        st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

    # Chat history (allows to ask multiple questions)
    try:
        st.session_state.messages = joblib.load(
            f'data/{st.session_state.chat_id}-st_messages'
        )
        st.session_state.gemini_history = joblib.load(
            f'data/{st.session_state.chat_id}-gemini_messages'
        )
        print('Loaded old cache')
    except:
        st.session_state.messages = []
        st.session_state.gemini_history = []
        print('Created new cache')
    
    st.session_state.model = genai.GenerativeModel('gemini-pro')
    st.session_state.chat = st.session_state.model.start_chat(
        history=st.session_state.gemini_history,
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(
            name=message['role'],
            avatar=message.get('avatar'),
        ):
            st.markdown(message['content'])

    # React to user input
    if prompt := st.chat_input('Your message here...'):
        # Save this as a chat for later
        if st.session_state.chat_id not in past_chats.keys():
            past_chats[st.session_state.chat_id] = st.session_state.chat_title
            joblib.dump(past_chats, 'data/past_chats_list')

        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append(
            dict(
                role='user',
                content=prompt,
            )
        )

        # Send message to AI
        response = st.session_state.chat.send_message(
            prompt,
            stream=True,
        )

        # Display assistant response in chat message container
        with st.chat_message(
            name=MODEL_ROLE,
            avatar=AI_AVATAR_ICON,
        ):
            message_placeholder = st.empty()
            full_response = ''
            assistant_response = response

            # Streams in a chunk at a time
            for chunk in response:
                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    time.sleep(0.05)
                    message_placeholder.write(full_response + '▌')
            message_placeholder.write(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            dict(
                role=MODEL_ROLE,
                content=full_response,
                avatar=AI_AVATAR_ICON,
            )
        )
        st.session_state.gemini_history = st.session_state.chat.history

        # Save to file
        joblib.dump(
            st.session_state.messages,
            f'data/{st.session_state.chat_id}-st_messages',
        )
        joblib.dump(
            st.session_state.gemini_history,
            f'data/{st.session_state.chat_id}-gemini_messages',
        )

# Integrate this into your existing main function
def main():
    logo_file = "D:\\sem4pbl\\logo.jpg"  # Update this with the path to your logo file
    logo_base64 = get_img_as_base64(logo_file)

    # Add the logo and background to the sidebar
    st.sidebar.markdown(f"""
    <style>
    .sidebar {{
        background-image: url('data:image/png;base64,{logo_base64}'); /* Use your desired image URL */
        background-size: cover; /* Cover the entire sidebar */
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh; /* Full height for the sidebar */
        padding: 20px; /* Add padding if needed */
        box-sizing: border-box; /* Ensure padding is included in total width/height */
    }}
    .sidebar-logo {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .sidebar-logo img {{
        width: 170px;
        max-width: 100px;  /* Adjust this value as needed */
    }}
    </style>
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("PrediFuel")

    # Main navbar radio button
    page = st.sidebar.radio("", ["HOME", "ABOUT", "FUEL ANALYSIS", "CO2 EMISSION", "QUERY", "REAL-TIME ESTIMATION"])

    if page == "HOME":
        home_page()
    
    elif page == "FUEL ANALYSIS":
        subpage = st.sidebar.radio("FUEL ANALYSIS", ["Visualization", "Prediction"])
        fuel_analysis(subpage)

    elif page == "CO2 EMISSION":
        subpage = st.sidebar.radio("CO2 EMISSION", ["Visualization", "Prediction"])
        co2_emission(subpage)

    elif page == "ABOUT":
        about()

    elif page == "QUERY":
        query_page()
    
    elif page == "REAL-TIME ESTIMATION":
        real_time_estimation()  # Call the new function for real-time estimation


import tempfile
import cv2
import torch
import streamlit as st
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')
import streamlit as st
import time
import IPython
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import os
import streamlit as st
import torch
import argparse
  

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Global variables for counting objects
data_car = []
data_bus = []
data_truck = []
data_motor = []
already = []
line_pos = 0.6

def detect(opt, stframe, car, bus, truck, motor, line, fps_rate, class_id):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    # choose custom class from streamlit
    opt.classes = class_id
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    sum_fps = 0
    line_pos = line
    save_vid = True
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        prev_time = time.time()
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                
                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        #count
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        # count_obj(bboxes,w,h,id, names[c], data_car, data_bus, data_truck, data_motor)
                        count_obj(bboxes,w,h,id, names[c], line_pos)
                        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                # count vehicle
                color = (0,255,0)
                color_car = (0,150,255)
                color_bus = (0,255,0)
                color_truck = (255,0,0)
                color_motor = (255,255,0)
                start_point = (0, int(line_pos*h))
                end_point = (w, int(line_pos*h))
                cv2.line(im0, start_point, end_point, color, thickness=2)
                thickness = 3
                org = (20, 70)
                distance_height = 100
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2
                # cv2.putText(im0, 'car: ' + str(len(data_car)), org, font, fontScale, color_car, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'bus: ' + str(len(data_bus)), (org[0], org[1] + distance_height), font, fontScale, color_bus, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'truck: ' + str(len(data_truck)), (org[0], org[1] + distance_height*2), font, fontScale, color_truck, thickness, cv2.LINE_AA)
                # cv2.putText(im0, 'motor: ' + str(len(data_motor)), (org[0], org[1] + distance_height*3), font, fontScale, color_motor, thickness, cv2.LINE_AA)

                # cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 60, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                vid_writer.write(im0)

                # show fps
                curr_time = time.time()
                fps_ = curr_time - prev_time
                fps_ = round(1/round(fps_, 3),1)
                prev_time = curr_time
                sum_fps += fps_

                stframe.image(im0, channels="BGR", use_column_width=True)
                car.markdown(f"<h3> {str(len(data_car))} </h3>", unsafe_allow_html=True)
                bus.write(f"<h3> {str(len(data_bus))} </h3>", unsafe_allow_html=True)
                truck.write(f"<h3> {str(len(data_truck))} </h3>", unsafe_allow_html=True)
                motor.write(f"<h3> {str(len(data_motor))} </h3>", unsafe_allow_html=True)
                fps_rate.markdown(f"<h3> {fps_} </h3>", unsafe_allow_html=True)
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print("Average FPS", round(1 / (sum(list(t)) / 1000), 1))
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    
def load_counts_from_file():
    """Loads counts of vehicles from a text file."""
    if os.path.exists('vehicle_count.txt'):
        with open('vehicle_count.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Car count:' in line:
                    data_car.extend([i for i in range(int(line.split(': ')[1].strip()))])
                elif 'Bus count:' in line:
                    data_bus.extend([i for i in range(int(line.split(': ')[1].strip()))])
                elif 'Truck count:' in line:
                    data_truck.extend([i for i in range(int(line.split(': ')[1].strip()))])
                elif 'Motorcycle count:' in line:
                    data_motor.extend([i for i in range(int(line.split(': ')[1].strip()))])



def save_counts_to_file():
    """Saves the counts of vehicles to a text file."""
    with open('vehicle_count.txt', 'w') as f:
        f.write(f"Car count: {len(data_car)}\n")
        f.write(f"Bus count: {len(data_bus)}\n")
        f.write(f"Truck count: {len(data_truck)}\n")
        f.write(f"Motorcycle count: {len(data_motor)}\n")

def reset_counts():
    """Resets the vehicle counts to zero."""
    global data_car, data_bus, data_truck, data_motor
    data_car, data_bus, data_truck, data_motor = [], [], [], []
    save_counts_to_file()  # Save the reset counts to the file

def count_obj(box, w, h, id, label, line_pos):
    global data_car, data_bus, data_truck, data_motor, already
    center_coordinates = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))

    # Classify one time per id
    if center_coordinates[1] > (h * line_pos):
        if id not in already:
            already.append(id)
            if label == 'car' and id not in data_car:
                data_car.append(id)
            elif label == 'bus' and id not in data_bus:
                data_bus.append(id)
            elif label == 'truck' and id not in data_truck:
                data_truck.append(id)
            elif label == 'motorcycle' and id not in data_motor:
                data_motor.append(id)

            # Save counts to file after counting a new vehicle
            save_counts_to_file()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='best_new.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/motor.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='evaluate inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Expand
    return opt

def real_time_estimation():
    global is_running
    st.title('Vehicle Detection and Counting')
    st.markdown('<h3 style="color: red">with Yolov5 and Deep SORT</h3>', unsafe_allow_html=True)
    load_counts_from_file() 
    # Video upload and setting up
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

    if video_file_buffer:
        st.sidebar.text('Input video')
        st.sidebar.video(video_file_buffer)
        video_path = os.path.join('videos', video_file_buffer.name)
        with open(video_path, 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    # Custom class selection
    custom_class = st.sidebar.checkbox('Custom classes')
    assigned_class_id = [0, 1, 2, 3]
    names = ['car', 'motorcycle', 'truck', 'bus']

    if custom_class:
        assigned_class_id = []
        assigned_class = st.sidebar.multiselect('Select custom classes', names)
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    line = st.sidebar.number_input('Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)

    status = st.empty()
    stframe = st.empty()

    car, bus, truck, motor = st.columns(4)
    
    with car:
        st.markdown('**Car**')
        car_text = st.markdown('__')
    
    with bus:
        st.markdown('**Bus**')
        bus_text = st.markdown('__')

    with truck:
        st.markdown('**Truck**')
        truck_text = st.markdown('__')
    
    with motor:
        st.markdown('**Motorcycle**')
        motor_text = st.markdown('__')

    fps_col, _, _, _ = st.columns(4)
    
    with fps_col:
        st.markdown('**FPS**')
        fps_text = st.markdown('__')

    # Start, Stop, and Reset buttons
    track_button = st.sidebar.button('START')
    stop_button = st.sidebar.button('STOP')
    reset_button = st.sidebar.button('RESET')

    if track_button:
        is_running = True
       
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = video_path

        status.markdown('<font size="4"> **Status:** Running... </font>', unsafe_allow_html=True)
        
        with torch.no_grad():
            while is_running:
                detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text, assigned_class_id)
            
            # Ensure the status updates when the loop exits
            status.markdown('<font size="4"> **Status:** Stopped </font>', unsafe_allow_html=True)

    # Handle the Stop button click
    if stop_button:
        is_running = False
        status.markdown('<font size="4"> **Status:** Stopped </font>', unsafe_allow_html=True)

        # Save vehicle counts to a file
        save_counts_to_file()

        # Read saved counts and display in the corresponding columns
        with open('vehicle_count.txt', 'r') as f:
            lines = f.readlines()
            car_count = int(lines[0].split(":")[1].strip())
            bus_count = int(lines[1].split(":")[1].strip())
            truck_count = int(lines[2].split(":")[1].strip())
            motorcycle_count = int(lines[3].split(":")[1].strip())

            car_text.markdown(f"**{car_count}**")
            bus_text.markdown(f"**{bus_count}**")
            truck_text.markdown(f"**{truck_count}**")
            motor_text.markdown(f"**{motorcycle_count}**")

            # Estimation section
            average_co2_emissions = {
                'Car': 170,       # Average CO2 emissions per car in grams/km
                'Bus': 97,       # Average CO2 emissions per bus in grams/km
                'Truck': 400,     # Average CO2 emissions per truck in grams/km
                'Motorcycle': 114   # Average CO2 emissions per motorcycle in grams/km
            }

            total_emission = (car_count * average_co2_emissions['Car'] +
                            bus_count * average_co2_emissions['Bus'] +
                            truck_count * average_co2_emissions['Truck'] +
                            motorcycle_count * average_co2_emissions['Motorcycle'])

            estimation_text = "### Estimation of Total CO2 Emissions"
            car_emission_text = f"Cars: {car_count} x {average_co2_emissions['Car']} g/km = {car_count * average_co2_emissions['Car']} g/km"
            bus_emission_text = f"Buses: {bus_count} x {average_co2_emissions['Bus']} g/km = {bus_count * average_co2_emissions['Bus']} g/km"
            truck_emission_text = f"Trucks: {truck_count} x {average_co2_emissions['Truck']} g/km = {truck_count * average_co2_emissions['Truck']} g/km"
            motorcycle_emission_text = f"Motorcycles: {motorcycle_count} x {average_co2_emissions['Motorcycle']} g/km = {motorcycle_count * average_co2_emissions['Motorcycle']} g/km"

            # Display the estimation texts
            st.markdown(estimation_text)
            st.markdown(car_emission_text)
            st.markdown(bus_emission_text)
            st.markdown(truck_emission_text)
            st.markdown(motorcycle_emission_text)
            total_population_text = f"<span style='color: #d2b356; font-size: 34px;'>Total CO2 Emission: {total_emission} g/km</span>"

            st.markdown(total_population_text, unsafe_allow_html=True)


    # Handle the Reset button click
    if reset_button:
        reset_counts()
        st.success("Counts have been reset to zero.")




if __name__ == "__main__":
    main()
