# Save this script as app.py or any other name you prefer

import streamlit as st
from geo_location import *
from locations import *
import matplotlib.pyplot as plt
from plotgraph import *
import pandas as pd
from sklearn.cluster import KMeans
import time

# Function to perform the analysis
def analyze_coordinates(coordinates):
    start = time.time()
    org_coordinates = coordinates
    coordinates = coordinates.replace(" ", "")
    query = ['Hostel', 'PG', 'Dorm']
    DataFrames = []

    for i in query:
        data = getData(coordinates, i)
        DataFrames.append(data)

    residence = pd.concat(DataFrames, ignore_index=True)
    st.write(residence)

    try:
        query = ['Juice', 'coffee', 'gym', 'cafe']
        amenities = get_residence_data(residence, query)
    except Exception as e:
        st.error(e)
        st.stop()

    amenities_refined = [[b for a, b in x.items()] for i in amenities for x in i]
    columns = ['JUICE_SHOPS', 'COFFEE_SHOPS', 'GYMS', 'CAFE']
    df1 = pd.DataFrame(amenities_refined, columns=columns)

    df1['Latitude'] = residence['latitude']
    df1['Longitude'] = residence['longitude']
    df1 = df1.fillna(0)
    st.write(df1)

    # Clustering
    cluster_data = df1[['Latitude', 'Longitude']]
    num_clusters = 6
    classifier = KMeans(n_clusters=num_clusters, random_state=42)
    y_pred = classifier.fit_predict(cluster_data)

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(cluster_data['Latitude'], cluster_data['Longitude'], c=y_pred, cmap='rainbow')
    ax.set_title('K-Means Clustering over general population')
    ax.set_ylabel('Longitude')
    ax.set_xlabel('Latitude')
    st.pyplot(fig)

    clusters = pd.DataFrame(y_pred, columns=['Clusters'])
    df1 = pd.concat([df1, clusters], axis=1)
    df1.insert(0, "Name", residence['name'])
    df1.to_csv('locations.csv', index=False)

    map = get_graph(org_coordinates)
    end = time.time()
    st.success('Completed in {}'.format(end - start))

    return df1


# Streamlit app layout
st.title('Residence and Amenities Analyzer')

# Input field for coordinates
coordinates_input = st.text_input('Enter the coordinates:', '')

# Submit button
if st.button('Analyze'):
    if coordinates_input:
        result_df = analyze_coordinates(coordinates_input)
        
        # CSV Download link
        st.download_button(
            label="Download data as CSV",
            data=result_df.to_csv().encode('utf-8'),
            file_name='locations.csv',
            mime='text/csv',
        )

    else:
        st.error('Please enter the coordinates.')

# Run this with `streamlit run app.py` in your command line