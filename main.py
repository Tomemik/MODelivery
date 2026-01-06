import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from itertools import islice

SURFACE_COLORS = {
    # Dobre (Gładkie) -> Zielony/Niebieski
    'asphalt': '#2ecc71',  # Zielony
    'concrete': '#27ae60',  # Ciemny zielony
    'paved': '#1abc9c',  # Morski
    'metal': '#1abc9c',

    # Średnie (Twarde ale nierówne) -> Pomarańczowy/Żółty
    'paving_stones': '#f39c12',  # Pomarańczowy
    'sett': '#e67e22',  # Ciemny pomarańcz
    'concrete:plates': '#f1c40f',  # Żółty

    # Złe (Nierówne/Miękkie) -> Czerwony
    'cobblestone': '#c0392b',  # Czerwony (Kocie łby)
    'unpaved': '#e74c3c',  # Jasny czerwony
    'grass': '#d35400',  # Rdzawy
    'gravel': '#d35400',
    'dirt': '#d35400',
    'earth': '#d35400',
    'sand': '#d35400',

    # Nieznane -> Szary
    'unknown': '#7f8c8d'
}

st.set_page_config(page_title="MCDM Multi-Stop Robot", layout="wide")

st.title("📦 Autonomiczny Kurier: Trasa Wielopunktowa")
st.markdown("""
1. Klikaj na mapie, aby dodać kolejne punkty dostaw.
2. W tabeli po lewej **edytuj wagi paczek** dla każdego punktu.
3. Kliknij **Oblicz Trasę**, aby uruchomić algorytm TOPSIS dla każdego odcinka.
""")

if 'stops' not in st.session_state:
    st.session_state['stops'] = []


def reset_app():
    st.session_state['stops'] = []


@st.cache_resource
def load_graph(place_name):
    G_multi = ox.graph_from_address(place_name, dist=5000, network_type='walk')
    G_multi = ox.add_edge_speeds(G_multi)
    G_multi = ox.add_edge_travel_times(G_multi)
    G_simple = ox.convert.to_digraph(G_multi, weight='length')
    return G_simple


PLACE_NAME = "Rynek Główny, Kraków, Poland"
with st.spinner("Ładowanie mapy Krakowa..."):
    G = load_graph(PLACE_NAME)



def get_surface_score(edge_data):
    surface = edge_data.get('surface')
    highway = edge_data.get('highway')

    if isinstance(surface, list): surface = surface[0]
    if isinstance(highway, list): highway = highway[0]

    scores = {
        'asphalt': 10, 'concrete': 9, 'paved': 9, 'metal': 9,
        'paving_stones': 7, 'sett': 5, 'concrete:plates': 6,
        'cobblestone': 3, 'unpaved': 3, 'gravel': 2,
        'dirt': 2, 'grass': 1, 'sand': 1, 'earth': 2,
        'unknown': 5
    }

    if not surface or surface == 'unknown':
        if highway in ['primary', 'secondary', 'tertiary', 'residential', 'living_street']:
            surface = 'asphalt'
        elif highway in ['pedestrian', 'footway', 'cycleway', 'steps', 'platform']:
            surface = 'paved'
        elif highway == 'service':
            surface = 'concrete'
        elif highway in ['track', 'path']:
            surface = 'dirt'
        else:
            surface = 'unknown'

    return scores.get(surface, 5)


def calculate_battery_consumption(length_m, current_total_load, surface_score):
    base_robot_weight = 20.0
    total_mass = base_robot_weight + current_total_load

    friction_factor = 1 + (10 - surface_score) * 0.15

    energy_factor = 0.02

    consumption = length_m * total_mass * friction_factor * energy_factor
    return round(consumption, 2)


def run_topsis(df, weights):
    data = df.copy()
    denom = np.sqrt((data ** 2).sum())
    denom[denom == 0] = 1
    norm_data = data / denom

    norm_data['length'] *= weights['length']
    norm_data['battery'] *= weights['battery']
    norm_data['surface_quality'] *= weights['surface_quality']

    ideal = {'length': norm_data['length'].min(), 'battery': norm_data['battery'].min(),
             'surface_quality': norm_data['surface_quality'].max()}
    anti_ideal = {'length': norm_data['length'].max(), 'battery': norm_data['battery'].max(),
                  'surface_quality': norm_data['surface_quality'].min()}

    dist_pos = np.sqrt(((norm_data - pd.Series(ideal)) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((norm_data - pd.Series(anti_ideal)) ** 2).sum(axis=1))

    return dist_neg / (dist_pos + dist_neg + 1e-9)


with st.sidebar:
    st.header("⚙️ Konfiguracja")

    st.subheader("Lista Przystanków")
    if st.session_state['stops']:
        stops_df = pd.DataFrame(st.session_state['stops'])

        edited_df = st.data_editor(
            stops_df[['id', 'type', 'drop_weight']],
            column_config={
                "drop_weight": st.column_config.NumberColumn("Waga paczki (kg)", min_value=0.0, max_value=50.0,
                                                             step=0.5),
                "type": "Typ punktu"
            },
            disabled=["id", "type"],
            hide_index=True
        )

        for index, row in edited_df.iterrows():
            st.session_state['stops'][index]['drop_weight'] = row['drop_weight']

        total_cargo = sum(s['drop_weight'] for s in st.session_state['stops'])
        st.info(f"Całkowity ładunek na starcie: {total_cargo} kg")
    else:
        st.info("Kliknij na mapie, aby dodać punkty.")

    st.divider()
    st.subheader("Wagi Kryteriów (TOPSIS)")
    w_len = st.slider("Priorytet: Dystans", 0.0, 1.0, 0.4)
    w_bat = st.slider("Priorytet: Bateria", 0.0, 1.0, 0.4)
    w_surf = st.slider("Priorytet: Jakość drogi", 0.0, 1.0, 0.2)

    if st.button("🗑️ Resetuj wszystko"):
        reset_app()
        st.rerun()


center_lat = G.nodes[list(G.nodes)[0]]['y']
center_lon = G.nodes[list(G.nodes)[0]]['x']

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

for i, stop in enumerate(st.session_state['stops']):
    if i == 0:
        icon = folium.Icon(color="green", icon="play", prefix='fa')
        label = "BAZA (Start)"
    elif i == len(st.session_state['stops']) - 1:
        icon = folium.Icon(color="red", icon="flag", prefix='fa')
        label = f"STOP {i} (Koniec): -{stop['drop_weight']}kg"
    else:
        icon = folium.Icon(color="blue", icon="box", prefix='fa')
        label = f"STOP {i}: -{stop['drop_weight']}kg"

    folium.Marker([stop['lat'], stop['lon']], popup=label, icon=icon).add_to(m)

output = st_folium(m, width=1000, height=500, key="map_input")

if output['last_clicked']:
    new_lat = output['last_clicked']['lat']
    new_lon = output['last_clicked']['lng']

    is_duplicate = False
    if st.session_state['stops']:
        last_stop = st.session_state['stops'][-1]
        if last_stop['lat'] == new_lat and last_stop['lon'] == new_lon:
            is_duplicate = True

    if not is_duplicate:
        idx = len(st.session_state['stops'])
        pt_type = "BAZA" if idx == 0 else "DOSTAWA"
        default_weight = 0.0 if idx == 0 else 5.0

        st.session_state['stops'].append({
            'id': idx,
            'lat': new_lat,
            'lon': new_lon,
            'type': pt_type,
            'drop_weight': default_weight
        })
        st.rerun()

if st.button("🚀 Oblicz Trasę Wielokryterialną") and len(st.session_state['stops']) > 1:

    stops = st.session_state['stops']
    full_route_geometry = []
    route_details = []
    debug_data = []

    current_payload = sum(s['drop_weight'] for s in stops)

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        total_segments = len(stops) - 1

        for i in range(total_segments):
            start_pt = stops[i]
            end_pt = stops[i + 1]

            status_text.write(
                f"Analiza odcinka {i + 1}/{total_segments}: {start_pt['type']} -> {end_pt['type']} (Ładunek: {current_payload} kg)")

            node_a = ox.nearest_nodes(G, start_pt['lon'], start_pt['lat'])
            node_b = ox.nearest_nodes(G, end_pt['lon'], end_pt['lat'])

            K = 40
            k_paths = list(islice(nx.shortest_simple_paths(G, node_a, node_b, weight='length'), K))

            segment_candidates = []
            for pid, path in enumerate(k_paths):
                length = nx.path_weight(G, path, weight='length')

                surf_scores = []
                for u, v in zip(path[:-1], path[1:]):
                    d = G.get_edge_data(u, v)
                    if d: surf_scores.append(get_surface_score(d))

                avg_surf = sum(surf_scores) / len(surf_scores) if surf_scores else 5
                bat = calculate_battery_consumption(length, current_payload, avg_surf)

                segment_candidates.append({
                    'id': pid,
                    'length': length,
                    'battery': bat,
                    'surface_quality': avg_surf,
                    'path': path
                })

            df_seg = pd.DataFrame(segment_candidates)

            weights_dict = {'length': w_len, 'battery': w_bat, 'surface_quality': w_surf}
            w_s = sum(weights_dict.values()) or 1
            weights_dict = {k: v / w_s for k, v in weights_dict.items()}

            scores = run_topsis(df_seg[['length', 'battery', 'surface_quality']], weights_dict)
            df_seg['TOPSIS_SCORE'] = scores

            best_idx = scores.idxmax()
            best_variant = df_seg.iloc[best_idx]

            debug_info = {
                'segment_idx': i + 1,
                'start': start_pt['type'],
                'end': end_pt['type'],
                'df': df_seg[['id', 'length', 'battery', 'surface_quality', 'TOPSIS_SCORE']].copy(),
                'best_id': best_variant['id']
            }
            debug_data.append(debug_info)

            full_route_geometry.extend(best_variant['path'])
            route_details.append({
                'odcinek': f"{i}->{i + 1}",
                'ładunek_kg': current_payload,
                'długość_m': round(best_variant['length'], 0),
                'bateria_Wh': round(best_variant['battery'], 2),
                'nawierzchnia': round(best_variant['surface_quality'], 1)
            })

            current_payload -= end_pt['drop_weight']
            if current_payload < 0: current_payload = 0

            progress_bar.progress((i + 1) / total_segments)

        st.session_state['calculation_results'] = {
            'details': route_details,
            'geometry': full_route_geometry,
            'debug_data': debug_data
        }

    except nx.NetworkXNoPath:
        st.error("Nie znaleziono połączenia między którymiś punktami.")
    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

if 'calculation_results' in st.session_state:
    results = st.session_state['calculation_results']

    st.success("Trasa wyznaczona pomyślnie!")

    st.subheader("📋 Szczegóły Odcinków")
    st.dataframe(pd.DataFrame(results['details']))

    st.markdown("""
    **Legenda nawierzchni:**
    <span style='color:#2ecc71'>■</span> Asfalt/Beton (Super) &nbsp;|&nbsp; 
    <span style='color:#f39c12'>■</span> Kostka brukowa (Średnio) &nbsp;|&nbsp; 
    <span style='color:#c0392b'>■</span> Kocie łby/Szuter (Źle) &nbsp;|&nbsp; 
    <span style='color:#7f8c8d'>■</span> Nieznana (Szary)
    """, unsafe_allow_html=True)

    st.subheader("🗺️ Mapa Wynikowa z Nawierzchniami")

    if results['geometry']:
        first_node = results['geometry'][0]
        center_lat_res = G.nodes[first_node]['y']
        center_lon_res = G.nodes[first_node]['x']
    else:
        center_lat_res, center_lon_res = 50.06, 19.94

    m_final = folium.Map(location=[center_lat_res, center_lon_res], zoom_start=14)

    full_path_nodes = results['geometry']

    for u, v in zip(full_path_nodes[:-1], full_path_nodes[1:]):

        if u == v: continue

        edge_data = G.get_edge_data(u, v)
        if edge_data is None: continue

        surf = edge_data.get('surface')
        highway = edge_data.get('highway')

        if isinstance(surf, list): surf = surf[0]
        if isinstance(highway, list): highway = highway[0]

        if surf is None:
            if highway in ['primary', 'secondary', 'tertiary', 'residential', 'living_street']:
                surf = 'asphalt'
            elif highway in ['cycleway', 'footway', 'path', 'pedestrian']:
                surf = 'paved'
            elif highway == 'service':
                surf = 'concrete'
            elif highway == 'track':
                surf = 'dirt'
            else:
                surf = 'unknown'

        color = SURFACE_COLORS.get(surf, '#7f8c8d')

        coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]

        folium.PolyLine(
            coords,
            color=color,
            weight=5,
            opacity=0.8,
            tooltip=f"Typ: {highway} | Nawierzchnia: {surf}"
        ).add_to(m_final)

    for i, stop in enumerate(st.session_state['stops']):
        if i == 0:
            color, icon_name = "green", "play"
        elif i == len(st.session_state['stops']) - 1:
            color, icon_name = "red", "flag"
        else:
            color, icon_name = "blue", "box"

        folium.Marker(
            [stop['lat'], stop['lon']],
            popup=f"STOP {i}: {stop['drop_weight']}kg",
            icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
        ).add_to(m_final)

    st_folium(m_final, width=1000, height=500, key="final_map_display")

    st.divider()
    st.header("🔍 Analiza Decyzyjna TOPSIS (Krok po kroku)")

    debug_list = results.get('debug_data', [])

    for info in debug_list:
        with st.expander(f"Odcinek {info['segment_idx']}: {info['start']} -> {info['end']}"):

            df = info['df']
            best_id = info['best_id']

            diff_surf = df['surface_quality'].max() - df['surface_quality'].min()
            diff_len = df['length'].max() - df['length'].min()

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 kandydatów (wg TOPSIS):**")
                st.dataframe(
                    df.sort_values(by='TOPSIS_SCORE', ascending=False).head(5)
                    .style.background_gradient(subset=['TOPSIS_SCORE'], cmap='Greens')
                    .format("{:.4f}", subset=['TOPSIS_SCORE'])
                )

            with col2:
                st.write("**Wnioski algorytmu:**")
                if diff_surf < 0.5:
                    st.warning(
                        "⚠️ Brak różnorodności nawierzchni! Wszystkie znalezione trasy są takie same pod kątem jakości drogi.")
                else:
                    st.success(f"✅ Znaleziono alternatywne drogi. Różnica jakości: {diff_surf:.1f} pkt.")

                if diff_len < 10:
                    st.info("ℹ️ Trasy mają niemal identyczną długość.")

                winner = df[df['id'] == best_id].iloc[0]
                st.markdown(
                    f"**Wybrano wariant ID {best_id}**, ponieważ uzyskał wynik **{winner['TOPSIS_SCORE']:.4f}**.")