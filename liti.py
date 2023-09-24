import pandas as pd
# import panel as pn
# import matplotlib.pyplot as plt
# import plotly.express as px 
import streamlit as st
import numpy as np
import datetime
import locale
# import mplcursors
# import plotly.graph_objects as go
# from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *
# from streamlit.components.v1 import html
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# set the locale to India
locale.setlocale(locale.LC_ALL, 'en_IN')
locale.setlocale(locale.LC_NUMERIC, 'en_IN')

# read data from the existing excel file
# read by default 1st sheet of an excel file
df = pd.read_excel('jabaru upi.xlsx')


#javascript


st.set_page_config(page_title="DATASET",
                   page_icon=":bar_chart:",
                   layout="wide")

default_date_in = df['DATE'].min()
default_date_out = df['DATE'].max()

#sidebar

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
    

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
st.sidebar.header("Please select a filter here:")



acc_no = st.sidebar.multiselect(
    "Select the Account No:",
    options=df["Account No"].unique(),
    default=df["Account No"].unique()
)
is_date_range = st.sidebar.checkbox("Filter by date range", True, key="date_range_toggle")
if is_date_range:
    # filter by date range
    selected_date_in = st.sidebar.date_input(
        "Show FROM:",
        value=default_date_in
    )

    selected_date_out = st.sidebar.date_input(
        "TO:",
        value=default_date_out
        
    )
    # combine selected date and time into a single datetime object
    selected_datetime_in = pd.to_datetime(str(selected_date_in))
    selected_datetime_out = pd.to_datetime(str(selected_date_out))

    
    filtered_df = df[(df["Account No"].isin(acc_no)) & (
        (df["DATE"].dt.date >= selected_datetime_in.date())  & 
        (df["VALUE_DATE"].dt.date <= selected_datetime_out.date())
    )]
else:
    selected_transaction = st.sidebar.date_input(
        "Search transaction by a particular date:",
        value=default_date_in
    )

    selected_time= st.sidebar.time_input(
        "Select a time:",
        value=datetime.time(0, 0)
    )
    
    selected_transaction = pd.to_datetime(str(selected_transaction)+' '+str(selected_time))
    filtered_df = df[(df["Account No"].isin(acc_no)) & (
        (df["DATE"].dt.date == selected_transaction.date())  | 
        (df["VALUE_DATE"].dt.date == selected_transaction.date())
    )]



#main page

st.title(":bar_chart: TRANSACTION Dashboard")
st.markdown("##")

st.dataframe(filtered_df)

total_with = int(filtered_df["WITHDRAWAL_AMT"].sum())
total_dep = int(filtered_df["DEPOSI_AMT"].sum())
average_bal=int(filtered_df["BALANCE_AMT"].mean())
left_column,middle_column,right_column   = st.columns(3)
with left_column:
    st.subheader("Total money withdrawn")
    st.subheader(f"{locale.currency(total_with, grouping=True)}")
with middle_column:
    st.subheader("Total money deposited")
    st.subheader(f"{locale.currency(total_dep, grouping=True)}")
with right_column:
    st.subheader("Average balance amount")
    if average_bal>=0:
        st.subheader(f"{locale.currency(abs(average_bal), grouping=True)}")
    else:
        st.subheader(f"- {locale.currency(abs(average_bal), grouping=True)}")
st.markdown("---")

#graphical representation

amount_by_account_no = (filtered_df.groupby(by=["Account No"]))
deposited_amt = filtered_df["DEPOSI_AMT"]
fig = px.bar(
    amount_by_account_no,
    x=filtered_df["Account No"],
    y=filtered_df["DEPOSI_AMT"],
    orientation='v',
    title="<b>Money deposited</b>",
    color_discrete_sequence=["#1DE23D"] * len(amount_by_account_no),
    template="plotly_white",
)

# Update the hovertemplate
fig.update_traces(
    hovertemplate="<br>".join([
        "Account No: %{x}",
        "Amount: ₹%{y:,.0f}",
    ]),
)

fig.update_layout(
    height=600,  # Set the desired height
    width=800,   # Set the desired width
)
st.plotly_chart(fig, use_container_width=True, raw=True)

# create a dictionary to store the filtered dataframes for each account number
filtered_dfs = {}

# loop through each account number and filter the dataframe
for acc in df["Account No"].unique():
    filtered_dfs[acc] = df[df["Account No"] == acc]

# create a bar chart with clickable bars
fig = px.bar(filtered_df,
             x="Account No",
             y="WITHDRAWAL_AMT",
             color_discrete_sequence=["#FF0600"] * len(amount_by_account_no))

fig.update_layout(
    title="<b>Money withdrawn</b>",
    template="plotly_white",
    height=600,  # Set the desired height
    width=800,   # Set the desired width
)

fig.update_traces(
    
    hovertemplate="<br>".join([
        "Account No: %{x}",
        "Amount: ₹%{y:,.0f}",
    ]),
    
)


for i, acc_no in enumerate(filtered_df['Account No']):
    link = ''
    fig.data[0].x[i] = f'<a href="{link}">{acc_no}</a>'
    
# display the chart
st.plotly_chart(fig, use_container_width=True, raw=True)



# Sidebar options
st.sidebar.header("Cluster Graph Options")
cols = ['BALANCE_AMT', 'WITHDRAWAL_AMT']
x = st.sidebar.selectbox('Select x-axis column:', cols)
ndf=filtered_df
y = filtered_df['Account No']
# Filter the DataFrame based on all years
ndf['Year'] = pd.to_datetime(ndf['DATE']).dt.year
years = ndf['Year'].unique().tolist()
dibi = ndf[ndf['Year'].isin(years)].copy()


# Perform clustering on the filtered data
leng = len(filtered_df['Account No'].unique())
kmeans = KMeans(n_clusters=leng)
kmeans.fit(dibi[[x]])
dibi['Cluster'] = kmeans.labels_

# Define your own color palette for the clusters
cluster_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00AA']

# Create the scatter plot with the custom cluster colors
fig = px.scatter(dibi, x=x, y='Year', color='Cluster', color_discrete_sequence=cluster_colors, hover_data=['Account No'], title='Cluster Graph')

fig.update_layout(height=600, width=800)

# Add hover labels
fig.update_traces(text=dibi['Account No'])

# Set y-axis tickformat to display the whole values
fig.update_yaxes(tickformat=".f")

# Display the plot using Streamlit
st.plotly_chart(fig, use_container_width=True, raw=True)




#funudflow graph bruh

#z-score ad
def zskorad(data):
    data_cpy = pd.Series(False, index=data.index)
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    for i in data:
        z = (i - mean) / std
        if abs(z) > threshold:
            data_cpy[i] = True
    return data_cpy

#Local Outlier Factor 
def lofad(data):
    data_cpy = pd.Series(False, index=data.index)
    
    # Reshape the data to 2D array format
    data = np.array(balance_amt).reshape(-1, 1)

    # Create the LOF model
    lof = LocalOutlierFactor(contamination="auto")
    
    anomalies = -lof.fit_predict(data)
    
    # Mark detected anomalies as True in the data_cpy series
    data_cpy[anomalies == -1] = True
    
    return data_cpy

#one class SVM
def ocSVM(data):
    #copying the dataset for further use
    data_cpy = pd.Series(False, index=data.index)
    
    # Reshape the data to 2D array format
    data = np.array(balance_amt).reshape(-1, 1)
    
    #creating the model
    svm = OneClassSVM(nu=0.31)
    #Feed the dataset into the model
    svm.fit(data)
    #predict the anomalies
    anomalies=svm.predict(data)
    
    # Mark detected anomalies as True in the data_cpy series
    data_cpy[anomalies == -1] = True
# If an element in anomalies is -1, 
# it means the corresponding data point is classified as an anomaly or outlier.
# If an element in anomalies is 1, 
# it means the corresponding data point is classified as a normal or inlier instance.
    return data_cpy

def svm_sgd(data,accno):
    
    data_cpy = pd.Series(False, index=data.index)
        
    # Reshape the data to 2D array format
    data = np.array(data).reshape(-1, 1)
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # Create the SGDClassifier model with linear kernel
    sgd = SGDClassifier(loss='huber', penalty='l2', class_weight={-1: 0.1, 1: 0.9})
    
    # Fit the model to the scaled data
    sgd.fit(scaled_data,accno)
    
    # Predict the anomalies
    anomalies = sgd.predict(scaled_data)
    
    # Mark detected anomalies as True in the data_cpy series
    data_cpy[anomalies == -1] = True
    return data_cpy

#ADTK
#animaly detection
data=filtered_df
# Sort the data by date
data['DATE'] = pd.to_datetime(data['DATE'])
data.sort_values(by='DATE', inplace=True)

# Set the date column as the index
data.set_index('DATE', inplace=True)

# Extract the balance amount column
balance_amt = data["BALANCE_AMT"]
#extract the corrsponding account numbers
acc_no = data["Account No"]
# #droppying the NaN values
# balance_amt = balance_amt.interpolate()
#types
types=['IQR-AD','ThresholdAD','QuantileAD','Z-scoreAD','PersistAD [+ve]','PersistAD [-ve]',
       'LofAD','OneClassSVM','OneClassSVM [SGD]']
#provide an option to use different anomaly detection
ad=st.sidebar.selectbox("What type of anomaly detection is to be done:",types)

#function for plotting
def plot(anomalies):
    # Create a Plotly figure
    fig = go.Figure()

    # Add the line plot of balance amount
    fig.add_trace(go.Scatter(
        x=balance_amt.index,
        y=balance_amt.values,
        mode='lines',
        text=acc_no,
        hovertemplate='(%{x}, ₹%{y})<br><span style = "color: blue;">Acc No: %{text}<extra></extra>',
        name='Balance Amount'
    ))

    # Add markers for anomalies
    anomaly_indices = [balance_amt.index.get_loc(idx) for idx in balance_amt.index[anomalies]]
    anomaly_values = [balance_amt.values[idx] for idx in anomaly_indices]
    min_bal=balance_amt.values.min()
    max_bal=balance_amt.values.max()

    fig.add_trace(go.Scatter(
        x=balance_amt.index[anomalies],
        y=anomaly_values,
        mode='markers',
        marker=dict(
            color='#DBFF33',
            size=5.5,
            symbol='diamond',
        ),
        
        text=acc_no,
        hovertemplate='(%{x}, ₹%{y})<br><span style="color: red;">Acc No: %{text}<extra></extra>',
        name='Anomalies',
    ))
    
    # Create a list of shape objects for vertical lines
    shapes = [
        dict(
            type="line",
            x0=balance_amt.index[idx],
            y0=min_bal,
            x1=balance_amt.index[idx],
            y1=max_bal,
            line=dict(
                color="#F20D19",
                width=1,
                
            ),
        )
        for idx in anomaly_indices
    ]

    # Add the shapes to the layout
    fig.update_layout(shapes=shapes)

    # Set plot layout
    fig.update_layout(
        title='Balance Amount Anomaly Detection - Threshold ANOMALY DETECTION',
        xaxis_title='Date',
        yaxis_title='Balance Amount',
        width=1100,  # Set the width of the graph
        height=700 
        
    )
    #additional legend entry
    fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(
        color='red',
        size=20,
        symbol='line-ns-open'
    ),
    name='Anomalies'
))
    # Render the plot using Streamlit
    st.plotly_chart(fig)

# Apply anomaly detection using 

#based on range
if ad == "ThresholdAD":
    threshold_detector = ThresholdAD(low=300, high=900000000)
    anomalies = threshold_detector.detect(balance_amt)  
    plot(anomalies)

#The main idea behind Quantile AD is to identify observations that deviate significantly from the expected
if ad == "QuantileAD":
    quantile_detector = QuantileAD(low=0.05, high=0.80)
    anomalies = quantile_detector.fit_detect(balance_amt)
    plot(anomalies)
    
#based on mathematical formulae
if ad =="IQR-AD":
    iqr_detector = InterQuartileRangeAD(c=1.5)
    anomalies = iqr_detector.fit_detect(balance_amt)
    plot(anomalies)

#Robust Covariance:
#based on standard deviation
if ad == "Z-scoreAD":
    zscore_detector = zskorad(balance_amt)
    anomalies = zscore_detector
    
    plot(anomalies)
    
#based on the keeping records of each vaalues as fun traverses though it
if ad == "PersistAD [+ve]":
    perad_detector1 = PersistAD(c=4.0, side = "positive")
    perad_detector1.window=5
    anomalies = perad_detector1.fit_detect(balance_amt)
    anomalies[0] = False
    plot(anomalies)
    
if ad == "PersistAD [-ve]":
    perad_detector2 = PersistAD(c=4.0, side = "negative")
    anomalies = perad_detector2.fit_detect(balance_amt)
    anomalies[0] = False
    plot(anomalies)
    
#it uses a mathematical formulae to calculate the anomalies
if ad == "LofAD":
    LofAD_detector = lofad(balance_amt)
    anomalies = LofAD_detector
    anomalies[0] = False    
    plot(anomalies)
    
#it is a simple implementation of machine learning model
if ad == "OneClassSVM":
    ocSVM_detector = ocSVM(balance_amt)
    anomalies = ocSVM_detector
    plot(anomalies)

# During the training process, the SGD algorithm adjusts the model parameters 
# iteratively based on the gradients computed on the selected subset of training instances.
if ad == "OneClassSVM [SGD]":
    target = []
    sgd_detector = svm_sgd(balance_amt,acc_no)
    anomalies = sgd_detector
    plot(anomalies)














#printing result can help you understand the process more easily
# sgd_detector = svm_sgd(balance_amt,acc_no)
# anomalies = sgd_detector
# print(anomalies)
