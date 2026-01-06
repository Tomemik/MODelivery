import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import BeautifyIcon
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from itertools import islice

st.set_page_config(page_title="MCDM Robot Delivery (TSP Battery)", layout="wide")

SURFACE_COLORS = {
    'asphalt': '#2ecc71',
    'concrete': '#27ae60',
    'paved': '#1abc9c',
    'metal': '#1abc9c',
    'paving_stones': '#f39c12',
    'sett': '#e67e22',
    'concrete:plates': '#f1c40f',
    'cobblestone': '#c0392b',
    'unpaved': '#e74c3c',
    'grass': '#d35400',
    'gravel': '#d35400',
    'dirt': '#d35400',
    'earth': '#d35400',
    'sand': '#d35400',
    'unknown': '#95a5a6'
}


@st.cache_resource
def load_graph(place_name):
    G = ox.graph_from_address(place_name, dist=3000, network_type='walk')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return ox.convert.to_digraph(G, weight='length')


def get_surface_details(edge_data):
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
        elif highway in ['pedestrian', 'footway', 'cycleway', 'steps']:
            surface = 'paved'
        elif highway == 'service':
            surface = 'concrete'
        elif highway in ['track', 'path']:
            surface = 'dirt'
        else:
            surface = 'unknown'

    return scores.get(surface, 5), surface, highway


def get_traffic_density(highway, hour):
    base_density = {
        'primary': 80, 'secondary': 60, 'tertiary': 40,
        'residential': 20, 'living_street': 10,
        'pedestrian': 50, 'footway': 30, 'steps': 10,
        'service': 5, 'path': 5, 'track': 0
    }.get(highway, 10)

    time_multiplier = 0.5
    if highway in ['primary', 'secondary', 'tertiary']:
        if 7 <= hour <= 9 or 15 <= hour <= 18:
            time_multiplier = 1.5
        elif 10 <= hour <= 14:
            time_multiplier = 1.0
    elif highway in ['pedestrian', 'footway', 'living_street']:
        if 11 <= hour <= 19:
            time_multiplier = 1.4
        elif 7 <= hour <= 9:
            time_multiplier = 1.2

    return min(max(base_density * time_multiplier, 0), 100)


def calculate_battery_consumption(length_m, current_total_load, avg_surface_score, avg_traffic):
    base_robot_weight = 25.0
    total_mass = base_robot_weight + current_total_load
    friction_factor = 1.0 + (10 - avg_surface_score) * 0.15
    traffic_penalty = 1.0 + (avg_traffic / 200.0)
    energy_per_meter_kg = 0.005
    return round(length_m * total_mass * friction_factor * traffic_penalty * energy_per_meter_kg, 2)


def add_energy_weights_to_graph(G, hour):
    for u, v, data in G.edges(data=True):
        length = data.get('length', 1.0)
        s_score, _, h_type = get_surface_details(data)
        traffic = get_traffic_density(h_type, hour)

        friction_factor = 1.0 + (10 - s_score) * 0.15
        traffic_penalty = 1.0 + (traffic / 200.0)
        energy_per_meter_kg = 0.005

        resistance = length * friction_factor * traffic_penalty * energy_per_meter_kg

        data['energy_resistance'] = resistance


def run_topsis(df, weights):
    data = df.copy()
    denom = np.sqrt((data ** 2).sum())
    denom[denom == 0] = 1
    norm_data = data / denom

    for col in weights:
        norm_data[col] *= weights[col]

    ideal = {
        'length': norm_data['length'].min(),
        'battery': norm_data['battery'].min(),
        'traffic_density': norm_data['traffic_density'].min(),
        'surface_quality': norm_data['surface_quality'].max()
    }
    anti_ideal = {
        'length': norm_data['length'].max(),
        'battery': norm_data['battery'].max(),
        'traffic_density': norm_data['traffic_density'].max(),
        'surface_quality': norm_data['surface_quality'].min()
    }

    dist_pos = np.sqrt(((norm_data - pd.Series(ideal)) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((norm_data - pd.Series(anti_ideal)) ** 2).sum(axis=1))
    return dist_neg / (dist_pos + dist_neg + 1e-9)


def solve_tsp_battery_greedy(G, stops):
    if len(stops) < 2:
        return stops

    nodes_info = []
    for s in stops:
        node = ox.nearest_nodes(G, s['lon'], s['lat'])
        nodes_info.append({'stop_data': s, 'osmid': node, 'original_id': s['id']})

    start_node = nodes_info[0]
    unvisited = nodes_info[1:]

    ordered_route = [start_node]
    current_node = start_node

    while unvisited:
        best_next = None
        min_cost = float('inf')
        best_idx = -1

        for idx, candidate in enumerate(unvisited):
            try:
                cost = nx.shortest_path_length(
                    G,
                    current_node['osmid'],
                    candidate['osmid'],
                    weight='energy_resistance'
                )

                if cost < min_cost:
                    min_cost = cost
                    best_next = candidate
                    best_idx = idx
            except nx.NetworkXNoPath:
                continue

        if best_next:
            ordered_route.append(best_next)
            current_node = best_next
            unvisited.pop(best_idx)
        else:
            break

    ordered_route.append(start_node)
    return [item['stop_data'] for item in ordered_route]


if 'stops' not in st.session_state: st.session_state['stops'] = []
if 'route_results' not in st.session_state: st.session_state['route_results'] = None

PLACE_NAME = "Rynek Główny, Kraków, Poland"
with st.spinner("Ładowanie grafu drogowego..."):
    G = load_graph(PLACE_NAME)

with st.sidebar:
    st.header("⚙️ Konfiguracja")
    delivery_hour = st.slider("🕒 Godzina rozpoczęcia", 0, 23, 8, format="%d:00")

    st.subheader("⚖️ Wagi Kryteriów")
    w_len = st.slider("Dystans", 0.0, 1.0, 0.3)
    w_bat = st.slider("Bateria", 0.0, 1.0, 0.3)
    w_surf = st.slider("Jakość Drogi", 0.0, 1.0, 0.2)
    w_traffic = st.slider("Unikanie Tłoku", 0.0, 1.0, 0.2)

    st.divider()
    if st.button("🗑️ Resetuj Trasę"):
        st.session_state['stops'] = []
        st.session_state['route_results'] = None
        st.rerun()

    if st.session_state['stops']:
        st.write("📦 **Edytuj wagi paczek (kg):**")
        stops_df = pd.DataFrame(st.session_state['stops'])
        if len(stops_df) > 1:
            edited_df = st.data_editor(
                stops_df[['id', 'drop_weight']],
                column_config={
                    "drop_weight": st.column_config.NumberColumn("Waga do zrzutu", min_value=0.0, max_value=20.0,
                                                                 step=0.5)},
                hide_index=True, disabled=['id']
            )
            for idx, row in edited_df.iterrows():
                real_idx = next((i for i, s in enumerate(st.session_state['stops']) if s['id'] == row['id']), None)
                if real_idx is not None:
                    st.session_state['stops'][real_idx]['drop_weight'] = row['drop_weight']

st.title("📦 Autonomiczny Kurier: Optymalizacja Baterii (TSP)")

col_map, col_data = st.columns([2, 1])

with col_map:
    start_node = list(G.nodes)[0]
    default_lat = G.nodes[start_node]['y']
    default_lon = G.nodes[start_node]['x']

    if 'map_center' not in st.session_state:
        st.session_state['map_center'] = [default_lat, default_lon]
    if 'map_zoom' not in st.session_state:
        st.session_state['map_zoom'] = 15

    m = folium.Map(location=st.session_state['map_center'], zoom_start=st.session_state['map_zoom'])

    for i, stop in enumerate(st.session_state['stops']):
        color = "green" if i == 0 else "blue"
        label = "BAZA" if i == 0 else f"Cel {stop['id']}"

        icon = BeautifyIcon(
            number=stop['id'],
            border_color=color,
            text_color=color,
            icon_shape='marker',
            border_width=2
        )
        folium.Marker(
            [stop['lat'], stop['lon']],
            popup=f"<b>{label}</b><br>Zrzut: {stop['drop_weight']}kg",
            icon=icon
        ).add_to(m)

    output = st_folium(
        m, width=800, height=500, key="input_map",
        returned_objects=["last_clicked", "zoom", "center"]
    )

    if output:
        if output.get('center'): st.session_state['map_center'] = [output['center']['lat'], output['center']['lng']]
        if output.get('zoom'): st.session_state['map_zoom'] = output['zoom']

    if output and output.get('last_clicked'):
        lc = output['last_clicked']
        is_new_click = True
        if st.session_state['stops']:
            last_stop = st.session_state['stops'][-1]
            if abs(last_stop['lat'] - lc['lat']) < 0.00001 and abs(last_stop['lon'] - lc['lng']) < 0.00001:
                is_new_click = False

        if is_new_click:
            new_id = len(st.session_state['stops'])
            st.session_state['stops'].append({
                'id': new_id,
                'lat': lc['lat'], 'lon': lc['lng'],
                'drop_weight': 5.0 if new_id > 0 else 0.0
            })
            st.rerun()

with col_data:
    st.info("Dodaj punkty na mapie. Punkt 0 to Baza (Start/Koniec).")

    if len(st.session_state['stops']) > 1:
        if st.button("🚀 Optymalizuj pod Baterię", type="primary"):

            with st.spinner("⚡ Aktualizacja wag energetycznych w grafie..."):
                add_energy_weights_to_graph(G, delivery_hour)

            with st.spinner("🔄 Wyznaczanie kolejności (Min. Zużycie Energii)..."):
                optimized_stops = solve_tsp_battery_greedy(G, st.session_state['stops'])

            st.success("Kolejność zoptymalizowana pod kątem baterii!")

            stops = optimized_stops
            full_geometry_data = []
            segment_details = []
            debug_data = []

            total_load = sum(s['drop_weight'] for s in stops)
            current_payload = total_load
            progress_bar = st.progress(0)

            try:
                for i in range(len(stops) - 1):
                    start_pt = stops[i]
                    end_pt = stops[i + 1]

                    orig_node = ox.nearest_nodes(G, start_pt['lon'], start_pt['lat'])
                    dest_node = ox.nearest_nodes(G, end_pt['lon'], end_pt['lat'])

                    k_paths = list(islice(nx.shortest_simple_paths(G, orig_node, dest_node, weight='length'), 15))

                    candidates = []
                    for idx, path in enumerate(k_paths):
                        length = nx.path_weight(G, path, weight='length')
                        path_metrics = {'surf': [], 'traffic': []}

                        for u, v in zip(path[:-1], path[1:]):
                            d = G.get_edge_data(u, v)
                            edge_d = d[0] if 0 in d else d
                            s_score, _, h_type = get_surface_details(edge_d)
                            t_score = get_traffic_density(h_type, delivery_hour)
                            path_metrics['surf'].append(s_score)
                            path_metrics['traffic'].append(t_score)

                        avg_surf = np.mean(path_metrics['surf']) if path_metrics['surf'] else 5
                        avg_traffic = np.mean(path_metrics['traffic']) if path_metrics['traffic'] else 0

                        bat = calculate_battery_consumption(length, current_payload, avg_surf, avg_traffic)

                        candidates.append({
                            'id': idx, 'path': path, 'length': length, 'battery': bat,
                            'surface_quality': avg_surf, 'traffic_density': avg_traffic
                        })

                    df_c = pd.DataFrame(candidates)
                    w_sum = sum([w_len, w_bat, w_surf, w_traffic]) or 1
                    norm_weights = {'length': w_len / w_sum, 'battery': w_bat / w_sum,
                                    'surface_quality': w_surf / w_sum, 'traffic_density': w_traffic / w_sum}

                    df_c['score'] = run_topsis(df_c[['length', 'battery', 'surface_quality', 'traffic_density']],
                                               norm_weights)
                    best = df_c.loc[df_c['score'].idxmax()]

                    debug_info = {
                        'segment_idx': i + 1,
                        'start_id': start_pt['id'],
                        'end_id': end_pt['id'],
                        'df': df_c[['id', 'length', 'battery', 'surface_quality', 'traffic_density', 'score']].copy(),
                        'best_id': df_c['score'].idxmax()
                    }
                    debug_data.append(debug_info)

                    path_nodes = best['path']
                    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                        d = G.get_edge_data(u, v)
                        edge_d = d[0] if 0 in d else d
                        _, surf_name, h_type = get_surface_details(edge_d)
                        t_perc = get_traffic_density(h_type, delivery_hour)
                        full_geometry_data.append({
                            'coords': [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]],
                            'surface': surf_name, 'highway': h_type, 'traffic': t_perc, 'leg_index': i
                        })

                    label_from = "BAZA" if start_pt['id'] == 0 else f"P{start_pt['id']}"
                    label_to = "BAZA" if end_pt['id'] == 0 else f"P{end_pt['id']}"

                    segment_details.append({
                        'Odcinek': f"{label_from} ➝ {label_to}",
                        'Dystans': f"{int(best['length'])} m",
                        'Bateria': f"{best['battery']:.2f} Wh",
                        'Ładunek': f"{current_payload} kg"
                    })

                    current_payload -= end_pt['drop_weight']
                    if current_payload < 0: current_payload = 0

                    progress_bar.progress((i + 1) / (len(stops) - 1))

                st.session_state['route_results'] = {
                    'geo_data': full_geometry_data,
                    'details': segment_details,
                    'debug_data': debug_data,
                    'optimized_stops': stops
                }

            except Exception as e:
                st.error(f"Błąd: {e}")

if st.session_state['route_results']:
    res = st.session_state['route_results']

    st.divider()
    st.subheader("📋 Plan Podróży (Zoptymalizowany)")
    st.dataframe(pd.DataFrame(res['details']))

    st.subheader("🗺️ Mapa Trasy")
    start_c = res['geo_data'][0]['coords'][0]
    m_res = folium.Map(location=start_c, zoom_start=15)

    base_features = []
    for item in res['geo_data']:
        color = SURFACE_COLORS.get(item['surface'], '#95a5a6')
        base_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[c[1], c[0]] for c in item['coords']]},
            "properties": {"style": {"color": color, "weight": 5, "opacity": 0.8}}
        })

    folium.GeoJson(
        data={"type": "FeatureCollection", "features": base_features},
        style_function=lambda x: x['properties']['style'],
        name="Nawierzchnie"
    ).add_to(m_res)

    legs_coords = {}
    for item in res['geo_data']:
        idx = item['leg_index']
        if idx not in legs_coords: legs_coords[idx] = []
        legs_coords[idx].append([[c[1], c[0]] for c in item['coords']])

    overlay_features = []
    for idx, coords_list in legs_coords.items():
        stops_opt = res['optimized_stops']
        f_id = stops_opt[idx]['id']
        t_id = stops_opt[idx + 1]['id']
        fname = "BAZA" if f_id == 0 else f"P{f_id}"
        tname = "BAZA" if t_id == 0 else f"P{t_id}"

        overlay_features.append({
            "type": "Feature",
            "geometry": {"type": "MultiLineString", "coordinates": coords_list},
            "properties": {"leg_name": f"Etap {idx + 1}: {fname} -> {tname}"}
        })

    folium.GeoJson(
        data={"type": "FeatureCollection", "features": overlay_features},
        style_function=lambda x: {"color": "transparent", "weight": 12, "opacity": 0},
        highlight_function=lambda x: {"color": "#f1c40f", "weight": 10, "opacity": 0.8},
        tooltip=folium.GeoJsonTooltip(fields=['leg_name'], labels=False, sticky=True),
        name="Etapy"
    ).add_to(m_res)

    opt_stops = res['optimized_stops']

    for order_idx, stop in enumerate(opt_stops[:-1]):
        is_base = (stop['id'] == 0)
        color = "green" if is_base else "red"

        icon = BeautifyIcon(number=stop['id'], border_color=color, text_color=color, icon_shape='marker',
                            border_width=2)

        folium.Marker(
            [stop['lat'], stop['lon']],
            tooltip=f"{'BAZA' if is_base else f'Punkt {stop['id']}'} (Odwiedziny #{order_idx})",
            icon=icon
        ).add_to(m_res)

    st_folium(m_res, width=1200, height=600, key="res_map")

    st.divider()
    st.header("🔍 Szczegóły Decyzji")

    debug_list = res.get('debug_data', [])
    for info in debug_list:
        start_label = "BAZA" if info['start_id'] == 0 else f"P{info['start_id']}"
        end_label = "BAZA" if info['end_id'] == 0 else f"P{info['end_id']}"

        with st.expander(f"Etap {info['segment_idx']}: {start_label} -> {end_label}"):
            df = info['df']
            best_score = df.loc[info['best_id'], 'score']
            st.write(f"Wybrano trasę ID **{info['best_id']}** z wynikiem TOPSIS: **{best_score:.4f}**")
            st.dataframe(df.sort_values(by='score', ascending=False).head(3))