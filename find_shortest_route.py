import pandas as pd
import networkx as nx
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('bus_routes_data.csv')
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Start Bus Stop'], row['End Bus Stop'],
                   distance=row['Distance (km)'],
                   travel_time=row['Travel Time (minutes)'])
    return G

def find_shortest_route(G, start, end, weight='Distance (km)'):
    try:
        # Compute the shortest path and its length based on the specified weight
        path = nx.shortest_path(G, source=start, target=end, weight=weight)
        path_length = nx.shortest_path_length(G, source=start, target=end, weight=weight)
        return path, path_length
    except nx.NetworkXNoPath:
        return None, float('inf')

# Streamlit interface
def main():
    st.title('Bus Route Finder')
    st.write('This application finds the shortest route between bus stops based on distance or travel time.')

    # Load data
    G = load_data()

    # User input
    start_stop = st.selectbox('Select Start Bus Stop', options=sorted(G.nodes))
    end_stop = st.selectbox('Select End Bus Stop', options=sorted(G.nodes))
    weight_choice = st.radio('Select parameter for shortest path calculation', ('distance', 'travel_time'))

    if st.button('Find Shortest Route'):
        path, path_length = find_shortest_route(G, start_stop, end_stop, weight=weight_choice)

        if path:
            st.write(f'Shortest route from {start_stop} to {end_stop}:')
            st.write(' -> '.join(path))
            st.write(f'Total {weight_choice}: {path_length:.2f}')
        else:
            st.write(f'No path found from {start_stop} to {end_stop}')

if __name__ == "__main__":
    main()
