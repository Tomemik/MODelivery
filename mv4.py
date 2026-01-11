import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import BeautifyIcon
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from itertools import islice

st.set_page_config(page_title="MCDM Robot Delivery: RSM & UTA", layout="wide")

SURFACE_COLORS = {
    'asphalt': '#2ecc71', 'concrete': '#27ae60', 'paved': '#1abc9c',
    'metal': '#1abc9c', 'paving_stones': '#f39c12', 'sett': '#e67e22',
    'concrete:plates': '#f1c40f', 'cobblestone': '#c0392b', 'unpaved': '#e74c3c',
    'grass': '#d35400', 'gravel': '#d35400', 'dirt': '#d35400',
    'earth': '#d35400', 'sand': '#d35400', 'unknown': '#95a5a6'
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
        'dirt': 2, 'grass': 1, 'sand': 1, 'earth': 2, 'unknown': 5
    }

    if not surface or surface == 'unknown':
        if highway in ['primary', 'secondary', 'tertiary', 'residential']:
            surface = 'asphalt'
        elif highway in ['pedestrian', 'footway', 'cycleway']:
            surface = 'paved'
        elif highway == 'service':
            surface = 'concrete'
        elif highway in ['track', 'path']:
            surface = 'dirt'
        else:
            surface = 'unknown'

    return scores.get(surface, 5), surface, highway

def get_traffic_density(highway, hour):
    base = {'primary': 80, 'secondary': 60, 'tertiary': 40, 'residential': 20, 'pedestrian': 50}.get(highway, 10)
    mult = 1.0
    if 7 <= hour <= 9 or 15 <= hour <= 18: mult = 1.5
    return min(max(base * mult, 0), 100)

def calculate_battery(length, load, surf, traffic):
    mass = 25.0 + load
    fric = 1.0 + (10 - surf) * 0.15
    pen = 1.0 + (traffic / 200.0)
    return length * mass * fric * pen * 0.005

def add_energy_weights(G, hour):
    for u, v, data in G.edges(data=True):
        l = data.get('length', 1.0)
        s, _, h = get_surface_details(data)
        t = get_traffic_density(h, hour)
        data['energy_resistance'] = calculate_battery(l, 0, s, t)

def naiwny_filtracja(X):
    P = []
    n = X.shape[0]
    local_X = X.copy()

    for i in range(n):
        if np.isnan(local_X[i]).any(): continue
        Y = local_X[i]
        y_index = i
        for j in range(i + 1, n):
            if np.isnan(local_X[j]).any(): continue
            if all(Y[2:] <= local_X[j][2:]):
                local_X[j] = np.nan
            elif all(local_X[j][2:] <= Y[2:]):
                Y = local_X[j]
                local_X[y_index] = np.nan
                y_index = j
        if not np.isnan(Y).any():
            P.append(Y.tolist())
            for k in range(n):
                if np.isnan(local_X[k]).any():
                    continue
                else:
                    if all(Y[2:] <= local_X[k][2:]):
                        local_X[k] = np.nan
            local_X[y_index] = np.nan
            eli = 0
            P0 = []
            for el in local_X:
                if (~np.isnan(el)).any():
                    eli = eli + 1
                    P0 = el
            if eli == 1:
                P.append(P0.tolist())
                break
    if len(P) == 0: return np.array([])
    return np.asarray(P)

def deleteSpecificRows(A, V):
    mask = np.zeros(A.shape[0]).astype(int)
    if V.shape[0] == 0: return A
    for it, row in enumerate(A[:, 2:]):
        for el in V[:, 2:]:
            if np.array_equal(row, el):
                mask[it] = 1
                break
    n_deleted = sum(mask)
    if n_deleted == A.shape[0]: return np.array([])
    Aout = np.zeros((A.shape[0] - n_deleted, A.shape[1]))
    i = 0
    for it, el in enumerate(A):
        if not mask[it]:
            Aout[i] = el
            i = i + 1
    return Aout

def getRanking(Aa, Ab, uVec):
    if Aa.shape[0] == 0 or Ab.shape[0] == 0: return np.zeros(uVec.shape[0])
    P_vol = []
    for elA in Aa:
        for elB in Ab:
            P_vol.append(np.prod(np.abs(elB[2:] - elA[2:])))
    P = np.array(P_vol)
    sumP = P.sum()
    if sumP == 0: sumP = 1.0
    w = P / sumP
    result = np.zeros(uVec.shape[0])
    for iter_idx, u in enumerate(uVec):
        f_score = 0
        i_pair = 0
        for elA in Aa:
            for elB in Ab:
                skipFlag = False
                for id_dim, elX in enumerate(u[2:]):
                    if not (min(elA[id_dim + 2], elB[id_dim + 2]) <= elX <= max(elA[id_dim + 2], elB[id_dim + 2])):
                        skipFlag = True
                        break
                if not skipFlag:
                    d1 = np.sqrt(sum(np.power(elA[2:] - u[2:], 2)))
                    d2 = np.sqrt(sum(np.power(elB[2:] - u[2:], 2)))
                    if (d1 + d2) > 0: f_score += w[i_pair] * (d2 / (d1 + d2))
                i_pair += 1
        result[iter_idx] = f_score
    return result

def run_advanced_rsm(df, weights):
    data_prep = df.copy()
    data_prep['surface_roughness'] = 10 - data_prep['surface_quality']

    crit_cols = ['length', 'battery', 'surface_roughness', 'traffic_density']

    norm_data = data_prep.copy()
    for col in crit_cols:
        _min, _max = data_prep[col].min(), data_prep[col].max()
        if _max > _min:
            norm_data[col] = (data_prep[col] - _min) / (_max - _min)
        else:
            norm_data[col] = 0.0
        w_key = 'surface_quality' if col == 'surface_roughness' else col
        norm_data[col] *= weights.get(w_key, 1.0)

    matrix = []
    for idx, row in norm_data.iterrows():
        vec = [row['id'], 0.0] + [row[c] for c in crit_cols]
        matrix.append(vec)

    X = np.array(matrix)

    A0 = naiwny_filtracja(X.copy())
    if A0.shape[0] == 0: return pd.Series(0, index=df.index)

    remaining = deleteSpecificRows(X, A0)
    if remaining.shape[0] > 0:
        A_worst = naiwny_filtracja(remaining.copy())
    else:
        nadir = np.max(X[:, 2:], axis=0)
        A_worst = np.array([[-1, 0.0] + nadir.tolist()])

    scores = getRanking(A0, A_worst, X)
    return pd.Series(scores, index=df.index)

def generate_uta_breakpoints(df, criteria_config):
    uta_functions = {}
    for crit, steps in criteria_config.items():
        min_val = df[crit].min()
        max_val = df[crit].max()
        x_vals = np.linspace(min_val, max_val, steps + 1)

        if crit == 'surface_quality':
            y_vals = np.linspace(0, 1, steps + 1)
        else:
            y_vals = np.linspace(1, 0, steps + 1)

        uta_functions[crit] = pd.DataFrame({'Breakpoint (x)': x_vals, 'Utility (u)': y_vals})
    return uta_functions

def calculate_uta_score(row, uta_funcs):
    total_utility = 0.0
    for crit, df_func in uta_funcs.items():
        val = row[crit]
        xs = df_func['Breakpoint (x)'].values
        ys = df_func['Utility (u)'].values
        u_val = np.interp(val, xs, ys)
        total_utility += u_val
    return total_utility

def run_topsis(df, weights):
    data = df.copy()
    norm = data / np.sqrt((data ** 2).sum())
    for c in weights: norm[c] *= weights[c]
    ideal = {'length': norm['length'].min(), 'battery': norm['battery'].min(),
             'traffic_density': norm['traffic_density'].min(), 'surface_quality': norm['surface_quality'].max()}
    anti = {'length': norm['length'].max(), 'battery': norm['battery'].max(),
            'traffic_density': norm['traffic_density'].max(), 'surface_quality': norm['surface_quality'].min()}
    d_pos = np.sqrt(((norm - pd.Series(ideal)) ** 2).sum(axis=1))
    d_neg = np.sqrt(((norm - pd.Series(anti)) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + 1e-9)

def solve_tsp(G, stops):
    if len(stops) < 2: return stops
    nodes = [ox.nearest_nodes(G, s['lon'], s['lat']) for s in stops]
    curr = nodes[0]
    route_nodes = [curr]
    unvisited = set(range(1, len(nodes)))
    path_indices = [0]
    while unvisited:
        nearest = min(unvisited, key=lambda x: nx.shortest_path_length(G, curr, nodes[x], weight='length'))
        route_nodes.append(nodes[nearest])
        path_indices.append(nearest)
        unvisited.remove(nearest)
        curr = nodes[nearest]
    path_indices.append(0)
    return [stops[i] for i in path_indices]

if 'stops' not in st.session_state: st.session_state['stops'] = []
if 'route_results' not in st.session_state: st.session_state['route_results'] = None
if 'uta_config' not in st.session_state: st.session_state['uta_config'] = None

PLACE = "Rynek Główny, Kraków, Poland"
G = load_graph(PLACE)

with st.sidebar:
    st.header("⚙️ Konfiguracja")
    hour = st.slider("Godzina", 0, 23, 8)

    st.subheader("⚖️ Wagi (TOPSIS & RSM)")
    w_len = st.slider("Dystans", 0.0, 1.0, 0.3)
    w_bat = st.slider("Bateria", 0.0, 1.0, 0.3)
    w_surf = st.slider("Nawierzchnia", 0.0, 1.0, 0.2)
    w_traf = st.slider("Ruch", 0.0, 1.0, 0.2)

    st.markdown("---")
    st.subheader("📈 Konfiguracja UTA*")
    uta_steps_len = st.number_input("Przedziały: Dystans", 1, 10, 3)
    uta_steps_bat = st.number_input("Przedziały: Bateria", 1, 10, 3)
    uta_steps_surf = st.number_input("Przedziały: Jakość", 1, 10, 2)
    uta_steps_traf = st.number_input("Przedziały: Ruch", 1, 10, 2)

    if st.button("🗑️ Reset"):
        st.session_state['stops'] = []
        st.session_state['route_results'] = None
        st.session_state['uta_config'] = None
        st.rerun()

st.title("📦 Optymalizacja trasy robota dostawczego (TOPSIS, RSM, UTA*)")

if st.session_state['route_results'] is None:
    col_ratio = [3, 2]
else:
    col_ratio = [2, 1]

col_map, col_data = st.columns(col_ratio)

with col_map:
    start_n = list(G.nodes)[0]
    m_loc = st.session_state.get('map_center', [G.nodes[start_n]['y'], G.nodes[start_n]['x']])
    m_zoom = st.session_state.get('map_zoom', 15)

    m = folium.Map(location=m_loc, zoom_start=m_zoom)
    for i, s in enumerate(st.session_state['stops']):
        c = "green" if i == 0 else "blue"
        # Fix: Pull text up with negative margin
        folium.Marker(
            [s['lat'], s['lon']],
            icon=BeautifyIcon(
                number=i,
                border_color=c,
                text_color=c,
                icon_shape='marker',
                inner_icon_style='margin-top: -1px; margin-left: -3px; font-size: 14px;'
            )
        ).add_to(m)

    out = st_folium(m, width=1400 if st.session_state['route_results'] is None else 800, height=1000 if st.session_state['route_results'] is None else 600, key="map_input",
                    returned_objects=["last_clicked", "zoom", "center"])

    if out:
        if out.get('center'): st.session_state['map_center'] = [out['center']['lat'], out['center']['lng']]
        if out.get('zoom'): st.session_state['map_zoom'] = out['zoom']
        if out.get('last_clicked'):
            lc = out['last_clicked']
            if not st.session_state['stops'] or (abs(st.session_state['stops'][-1]['lat'] - lc['lat']) > 0.0001):
                st.session_state['stops'].append(
                    {'id': len(st.session_state['stops']), 'lat': lc['lat'], 'lon': lc['lng'], 'drop_weight': 5.0})
                st.rerun()

with col_data:
    if len(st.session_state['stops']) < 2:
        st.info("Dodaj co najmniej 2 punkty na mapie.")
    else:
        st.write("### 📐 Edytor Funkcji Użyteczności (UTA*)")
        st.markdown("")

        dummy_df = pd.DataFrame(
            {'length': [0, 1000], 'battery': [0, 100], 'surface_quality': [1, 10], 'traffic_density': [0, 100]})
        steps_cfg = {'length': uta_steps_len, 'battery': uta_steps_bat, 'surface_quality': uta_steps_surf,
                     'traffic_density': uta_steps_traf}

        if st.session_state['uta_config'] is None:
            st.session_state['uta_config'] = generate_uta_breakpoints(dummy_df, steps_cfg)

        tabs = st.tabs(["Dystans", "Bateria", "Jakość", "Ruch"])
        edited_uta_funcs = {}
        criteria_map = {"Dystans": 'length', "Bateria": 'battery', "Jakość": 'surface_quality',
                        "Ruch": 'traffic_density'}

        for i, (label, key) in enumerate(criteria_map.items()):
            with tabs[i]:
                df_editor = st.data_editor(
                    st.session_state['uta_config'][key], key=f"editor_{key}",
                    column_config={"Breakpoint (x)": st.column_config.NumberColumn(disabled=True),
                                   "Utility (u)": st.column_config.NumberColumn(min_value=0.0, max_value=1.0)},
                    hide_index=True
                )
                edited_uta_funcs[key] = df_editor
                st.line_chart(df_editor.set_index('Breakpoint (x)'))

        if st.button("🚀 Oblicz Wszystkie Metody", type="primary"):
            st.session_state['uta_config'] = edited_uta_funcs

            with st.spinner("Analiza tras..."):
                add_energy_weights(G, hour)
                stops_ordered = solve_tsp(G, st.session_state['stops'])

                results = {'TOPSIS': [], 'RSM': [], 'UTA': []}
                full_geo = {'TOPSIS': [], 'RSM': [], 'UTA': []}
                payload = sum(s['drop_weight'] for s in stops_ordered)
                debug_list = []

                prog = st.progress(0)

                for i in range(len(stops_ordered) - 1):
                    s1, s2 = stops_ordered[i], stops_ordered[i + 1]
                    n1, n2 = ox.nearest_nodes(G, s1['lon'], s1['lat']), ox.nearest_nodes(G, s2['lon'], s2['lat'])
                    paths = list(islice(nx.shortest_simple_paths(G, n1, n2, weight='length'), 15))

                    cand_data = []
                    for idx, p in enumerate(paths):
                        length = nx.path_weight(G, p, weight='length')
                        surfs, trafs = [], []
                        for u, v in zip(p[:-1], p[1:]):
                            d = G.get_edge_data(u, v)
                            sc, _, h = get_surface_details(d)
                            surfs.append(sc)
                            trafs.append(get_traffic_density(h, hour))
                        avg_s = np.mean(surfs) if surfs else 5
                        avg_t = np.mean(trafs) if trafs else 0
                        bat = calculate_battery(length, payload, avg_s, avg_t)
                        cand_data.append(
                            {'id': idx, 'path': p, 'length': length, 'battery': bat, 'surface_quality': avg_s,
                             'traffic_density': avg_t})

                    df_c = pd.DataFrame(cand_data)
                    w_dict = {'length': w_len, 'battery': w_bat, 'surface_quality': w_surf, 'traffic_density': w_traf}

                    df_c['score_topsis'] = run_topsis(df_c[['length', 'battery', 'surface_quality', 'traffic_density']],
                                                      w_dict)
                    best_topsis = df_c.loc[df_c['score_topsis'].idxmax()]

                    df_c['score_rsm'] = run_advanced_rsm(df_c, w_dict)
                    if df_c['score_rsm'].sum() == 0:
                        best_rsm = df_c.iloc[0]
                    else:
                        best_rsm = df_c.loc[df_c['score_rsm'].idxmax()]

                    df_c['score_uta'] = df_c.apply(lambda row: calculate_uta_score(row, st.session_state['uta_config']),
                                                   axis=1)
                    best_uta = df_c.loc[df_c['score_uta'].idxmax()]

                    for m_name, best_row in [('TOPSIS', best_topsis), ('RSM', best_rsm), ('UTA', best_uta)]:
                        pts = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in best_row['path']]
                        full_geo[m_name].append(pts)
                        results[m_name].append(best_row)

                    debug_list.append({
                        'leg': i + 1,
                        'df': df_c[['id', 'score_topsis', 'score_rsm', 'score_uta']],
                        'winners': {'TOPSIS': best_topsis['id'], 'RSM': best_rsm['id'], 'UTA': best_uta['id']}
                    })

                    payload -= s2['drop_weight']
                    prog.progress((i + 1) / (len(stops_ordered) - 1))

                st.session_state['route_results'] = {'geo': full_geo, 'stats': results, 'debug': debug_list,
                                                     'stops': stops_ordered}
                st.rerun()

if st.session_state['route_results']:
    res = st.session_state['route_results']

    st.divider()
    st.header("🗺️ Mapa Wynikowa: Porównanie Metod")
    st.caption("Przełączaj warstwy (ikona w prawym górnym rogu), aby zobaczyć różnice.")

    m_res = folium.Map(location=[res['stops'][0]['lat'], res['stops'][0]['lon']], zoom_start=15)

    for i, s in enumerate(res['stops']):
        c = "black" if i == 0 else "gray"
        folium.Marker(
            [s['lat'], s['lon']],
            icon=BeautifyIcon(
                number=i,
                border_color=c,
                text_color=c,
                icon_shape='marker',
                inner_icon_style='margin-top: -1px; margin-left: -3px; font-size: 14px;'
            )
        ).add_to(m_res)

    fg_top = folium.FeatureGroup(name="TOPSIS (Czerwony)", show=True)
    for seg in res['geo']['TOPSIS']: folium.PolyLine(seg, color="#e74c3c", weight=6, opacity=0.7).add_to(fg_top)
    fg_top.add_to(m_res)

    fg_rsm = folium.FeatureGroup(name="RSM (Niebieski)", show=True)
    for seg in res['geo']['RSM']: folium.PolyLine(seg, color="#3498db", weight=6, opacity=0.7).add_to(fg_rsm)
    fg_rsm.add_to(m_res)

    fg_uta = folium.FeatureGroup(name="UTA* (Fioletowy)", show=True)
    for seg in res['geo']['UTA']: folium.PolyLine(seg, color="#9b59b6", weight=6, opacity=0.7).add_to(fg_uta)
    fg_uta.add_to(m_res)

    folium.LayerControl().add_to(m_res)
    st_folium(m_res, width=1800, height=1000, key="res_map_final")

    st.subheader("📊 Podsumowanie Liczbowe")
    summary = []
    for method in ['TOPSIS', 'RSM', 'UTA']:
        total_len = sum(x['length'] for x in res['stats'][method])
        total_bat = sum(x['battery'] for x in res['stats'][method])
        summary.append({'Metoda': method, 'Dystans [m]': int(total_len), 'Bateria [Wh]': round(total_bat, 2)})
    st.table(pd.DataFrame(summary))

    with st.expander("🔍 Szczegóły Decyzji (Logi)"):
        st.markdown("")
        for d in res['debug']:
            st.markdown(f"#### Etap {d['leg']}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("TOPSIS (Top 3)")
                st.dataframe(d['df'][['score_topsis']].sort_values('score_topsis', ascending=False).head(3))
            with c2:
                st.caption("RSM (Top 3)")
                st.dataframe(d['df'][['score_rsm']].sort_values('score_rsm', ascending=False).head(3))
            with c3:
                st.caption("UTA* (Top 3)")
                st.dataframe(d['df'][['score_uta']].sort_values('score_uta', ascending=False).head(3))