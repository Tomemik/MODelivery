import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import BeautifyIcon
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from itertools import islice

st.set_page_config(page_title="Robot Delivery: AHP, TOPSIS & UTA*", layout="wide")

# --- KONFIGURACJA POWIERZCHNI ---
SURFACE_SCORES = {
    'asphalt': 10, 'concrete': 9, 'paved': 9, 'metal': 9,
    'paving_stones': 7, 'sett': 5, 'concrete:plates': 6,
    'cobblestone': 3, 'unpaved': 3, 'gravel': 2,
    'dirt': 2, 'grass': 1, 'sand': 1, 'earth': 2, 'unknown': 5
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
    if not surface or surface == 'unknown':
        if highway in ['primary', 'secondary', 'tertiary', 'residential']:
            surface = 'asphalt'
        elif highway in ['pedestrian', 'footway', 'cycleway']:
            surface = 'paved'
        else:
            surface = 'unknown'
    return SURFACE_SCORES.get(surface, 5), surface, highway


def get_traffic_density(highway, hour):
    base = {'primary': 80, 'secondary': 60, 'tertiary': 40, 'residential': 20, 'pedestrian': 50}.get(highway, 10)
    mult = 1.5 if (7 <= hour <= 9 or 15 <= hour <= 18) else 1.0
    return min(max(base * mult, 0), 100)


def calculate_battery(length, load, surf, traffic):
    mass = 25.0 + load
    fric = 1.0 + (10 - surf) * 0.15
    pen = 1.0 + (traffic / 200.0)
    return length * mass * fric * pen * 0.005


# --- MCDM CORE ---
def calculate_ahp_weights(comparison_matrix):
    matrix = np.array(comparison_matrix)
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    max_eig = eig_vals.max().real
    weights = eig_vecs[:, eig_vals.argmax()].real
    weights = weights / weights.sum()
    n = len(weights)
    ci = (max_eig - n) / (n - 1) if n > 1 else 0
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12}
    cr = ci / ri[n] if ri[n] > 0 else 0
    return weights, cr


def run_ahp_ranking(df, weights_dict):
    temp = df.copy()
    for col in ['length', 'battery', 'traffic_density']:
        span = temp[col].max() - temp[col].min()
        temp[col] = (temp[col].max() - temp[col]) / span if span > 0 else 1.0
    span_s = temp['surface_quality'].max() - temp['surface_quality'].min()
    temp['surface_quality'] = (temp['surface_quality'] - temp['surface_quality'].min()) / span_s if span_s > 0 else 1.0
    return (temp['length'] * weights_dict['length'] + temp['battery'] * weights_dict['battery'] +
            temp['surface_quality'] * weights_dict['surface_quality'] + temp['traffic_density'] * weights_dict[
                'traffic_density'])


def run_topsis(df, weights):
    data = df[['length', 'battery', 'surface_quality', 'traffic_density']].copy()
    norm = data / np.sqrt((data ** 2).sum() + 1e-9)
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
    curr, route_nodes, unvisited, path_indices = nodes[0], [nodes[0]], set(range(1, len(nodes))), [0]
    while unvisited:
        nxt = min(unvisited, key=lambda x: nx.shortest_path_length(G, curr, nodes[x], weight='length'))
        route_nodes.append(nodes[nxt]);
        path_indices.append(nxt);
        unvisited.remove(nxt);
        curr = nodes[nxt]
    path_indices.append(0)  # Powrót do bazy
    return [stops[i] for i in path_indices]


# --- SESSION STATE ---
if 'stops' not in st.session_state: st.session_state['stops'] = []
if 'route_results' not in st.session_state: st.session_state['route_results'] = None
if 'uta_config' not in st.session_state: st.session_state['uta_config'] = None

PLACE = "Rynek Główny, Kraków, Poland"
G = load_graph(PLACE)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Parametry Misji")
    hour = st.slider("Godzina", 0, 23, 8)

    st.subheader("📦 Masa paczek [kg]")
    if len(st.session_state['stops']) > 0:
        stops_payload = []
        for i, s in enumerate(st.session_state['stops']):
            stops_payload.append({
                "ID": i,
                "Typ": "Baza" if i == 0 else "Dostawa",
                "Masa [kg]": s.get('drop_weight', 5.0) if i > 0 else 0.0
            })
        df_payload = pd.DataFrame(stops_payload)
        edited_df = st.data_editor(df_payload, disabled=["ID", "Typ"], hide_index=True)
        for i, row in edited_df.iterrows():
            st.session_state['stops'][i]['drop_weight'] = row['Masa [kg]']
    else:
        st.info("Dodaj bazę (0) na mapie.")

    st.subheader("⚖️ Wagi AHP (Saaty)")


    def get_saaty(val):
        opts = [9, 7, 5, 3, 1, 3, 5, 7, 9]
        s_val = opts[val + 4]
        return s_val if val >= 0 else 1 / s_val


    v1 = st.select_slider("Dystans vs Bateria", options=range(-4, 5), value=0)
    v2 = st.select_slider("Dystans vs Jakość", options=range(-4, 5), value=2)
    v3 = st.select_slider("Dystans vs Ruch", options=range(-4, 5), value=-1)
    v4 = st.select_slider("Bateria vs Jakość", options=range(-4, 5), value=1)
    v5 = st.select_slider("Bateria vs Ruch", options=range(-4, 5), value=-2)
    v6 = st.select_slider("Jakość vs Ruch", options=range(-4, 5), value=-3)

    m_ahp = np.ones((4, 4))
    m_ahp[0, 1] = get_saaty(v1);
    m_ahp[1, 0] = 1 / m_ahp[0, 1]
    m_ahp[0, 2] = get_saaty(v2);
    m_ahp[2, 0] = 1 / m_ahp[0, 2]
    m_ahp[0, 3] = get_saaty(v3);
    m_ahp[3, 0] = 1 / m_ahp[0, 3]
    m_ahp[1, 2] = get_saaty(v4);
    m_ahp[2, 1] = 1 / m_ahp[1, 2]
    m_ahp[1, 3] = get_saaty(v5);
    m_ahp[3, 1] = 1 / m_ahp[1, 3]
    m_ahp[2, 3] = get_saaty(v6);
    m_ahp[3, 2] = 1 / m_ahp[2, 3]
    ahp_w, cr = calculate_ahp_weights(m_ahp)
    w_dict = {'length': ahp_w[0], 'battery': ahp_w[1], 'surface_quality': ahp_w[2], 'traffic_density': ahp_w[3]}
    st.caption(f"Spójność (CR): {cr:.3f}")

    st.subheader("📈 Konfiguracja UTA*")
    uta_steps_len = st.number_input("Przedziały: Dystans", 1, 10, 3)
    uta_steps_bat = st.number_input("Przedziały: Bateria", 1, 10, 3)
    uta_steps_surf = st.number_input("Przedziały: Jakość", 1, 10, 2)
    uta_steps_traf = st.number_input("Przedziały: Ruch", 1, 10, 2)

    if st.button("🗑️ Resetuj wszystko"):
        st.session_state['stops'] = [];
        st.session_state['route_results'] = None;
        st.session_state['uta_config'] = None;
        st.rerun()

st.title("📦 Optymalizacja trasy robota (AHP, TOPSIS, UTA*)")

col_map, col_uta = st.columns([3, 2])

with col_map:
    m = folium.Map(location=[50.0617, 19.9373], zoom_start=15)
    for i, s in enumerate(st.session_state['stops']):
        color = "green" if i == 0 else "blue"
        folium.Marker(
            [s['lat'], s['lon']],
            icon=BeautifyIcon(
                number=i,
                border_color=color,
                text_color=color,
                icon_shape='marker',
                inner_icon_style='margin-top: -1px; margin-left: -3px; font-size: 14px;'
            )
        ).add_to(m)

    out = st_folium(m, width="100%", height=500, key="delivery_map")
    if out and out.get('last_clicked'):
        lc = out['last_clicked']
        if not st.session_state['stops'] or (abs(st.session_state['stops'][-1]['lat'] - lc['lat']) > 0.0001):
            st.session_state['stops'].append(
                {'lat': lc['lat'], 'lon': lc['lng'], 'drop_weight': 5.0 if st.session_state['stops'] else 0.0})
            st.rerun()

with col_uta:
    st.subheader("📐 Edytor Funkcji Użyteczności (UTA*)")
    if st.session_state['uta_config'] is None:
        dummy_df = pd.DataFrame(
            {'length': [0, 2000], 'battery': [0, 500], 'surface_quality': [1, 10], 'traffic_density': [0, 100]})
        uta_funcs = {}
        for k, steps in [('length', uta_steps_len), ('battery', uta_steps_bat), ('surface_quality', uta_steps_surf),
                         ('traffic_density', uta_steps_traf)]:
            x = np.linspace(dummy_df[k].min(), dummy_df[k].max(), steps + 1)
            y = np.linspace(0, 1, steps + 1) if k == 'surface_quality' else np.linspace(1, 0, steps + 1)
            uta_funcs[k] = pd.DataFrame({'Breakpoint (x)': x, 'Utility (u)': y})
        st.session_state['uta_config'] = uta_funcs

    tabs = st.tabs(["Dystans", "Bateria", "Jakość", "Ruch"])
    for i, k in enumerate(st.session_state['uta_config'].keys()):
        with tabs[i]:
            st.session_state['uta_config'][k] = st.data_editor(st.session_state['uta_config'][k], key=f"uta_ed_{k}",
                                                               hide_index=True)
            st.line_chart(st.session_state['uta_config'][k].set_index('Breakpoint (x)'))

if st.button("🚀 Oblicz optymalną trasę", type="primary"):
    if len(st.session_state['stops']) < 2:
        st.error("Dodaj bazę (0) i punkty dostaw.")
    else:
        with st.spinner("Przetwarzanie grafu i rankingów..."):
            ordered = solve_tsp(G, st.session_state['stops'])
            res_stats, res_geo, res_debug = {'TOPSIS': [], 'AHP': [], 'UTA': []}, {'TOPSIS': [], 'AHP': [],
                                                                                   'UTA': []}, []
            payload = sum(s['drop_weight'] for s in st.session_state['stops'])

            for i in range(len(ordered) - 1):
                n1, n2 = ox.nearest_nodes(G, ordered[i]['lon'], ordered[i]['lat']), ox.nearest_nodes(G, ordered[i + 1][
                    'lon'], ordered[i + 1]['lat'])
                paths = list(islice(nx.shortest_simple_paths(G, n1, n2, weight='length'), 15))
                c_data = []
                for idx, p in enumerate(paths):
                    l = nx.path_weight(G, p, weight='length')
                    stats = [get_surface_details(G.get_edge_data(u, v)) for u, v in zip(p[:-1], p[1:])]
                    avg_s = np.mean([s[0] for s in stats])
                    avg_t = np.mean([get_traffic_density(s[2], hour) for s in stats])
                    bat = calculate_battery(l, payload, avg_s, avg_t)
                    c_data.append({'id': idx, 'path': p, 'length': l, 'battery': bat, 'surface_quality': avg_s,
                                   'traffic_density': avg_t})

                df_c = pd.DataFrame(c_data)
                df_c['s_topsis'] = run_topsis(df_c, w_dict)
                df_c['s_ahp'] = run_ahp_ranking(df_c, w_dict)
                df_c['s_uta'] = df_c.apply(lambda r: np.sum([np.interp(r[k], st.session_state['uta_config'][k][
                    'Breakpoint (x)'], st.session_state['uta_config'][k]['Utility (u)']) for k in
                                                             st.session_state['uta_config']]), axis=1)

                for m, col in [('TOPSIS', 's_topsis'), ('AHP', 's_ahp'), ('UTA', 's_uta')]:
                    best = df_c.loc[df_c[col].idxmax()]
                    res_stats[m].append(best)
                    res_geo[m].append([[G.nodes[n]['y'], G.nodes[n]['x']] for n in best['path']])

                res_debug.append({'leg': i + 1, 'df': df_c[
                    ['id', 'length', 'battery', 'surface_quality', 's_topsis', 's_ahp', 's_uta']]})
                payload -= ordered[i + 1]['drop_weight']

            st.session_state['route_results'] = {'geo': res_geo, 'stats': res_stats, 'stops': ordered,
                                                 'debug': res_debug}
            st.rerun()

if st.session_state['route_results']:
    rr = st.session_state['route_results']
    st.divider()

    st.header("🗺️ Wynik Optymalizacji")
    m_res = folium.Map(location=[rr['stops'][0]['lat'], rr['stops'][0]['lon']], zoom_start=15)

    for i, s in enumerate(rr['stops']):
        color = "green" if i == 0 or i == len(rr['stops']) - 1 else "blue"
        label = f"Punkt {i}" if i < len(rr['stops']) - 1 else "Powrót do Bazy"
        is_base = (i == 0 or i == len(rr['stops']) - 1)
        number = 0 if is_base else i
        folium.Marker([s['lat'], s['lon']], popup=label,
                      icon=BeautifyIcon(number=number, border_color=color, text_color=color, icon_shape='marker', inner_icon_style='margin-top: -1px; margin-left: -3px; font-size: 14px;'
        )).add_to(m_res)

    colors = {"TOPSIS": "red", "AHP": "blue", "UTA": "purple"}
    for m, clr in colors.items():
        fg = folium.FeatureGroup(name=f"Metoda {m}").add_to(m_res)
        for seg in rr['geo'][m]: folium.PolyLine(seg, color=clr, weight=5, opacity=0.8).add_to(fg)
    folium.LayerControl().add_to(m_res)
    st_folium(m_res, width=1200, height=500, key="res_map")

    summary = []
    for m in ['TOPSIS', 'AHP', 'UTA']:
        summary.append({
            'Metoda': m,
            'Dystans [m]': int(sum(x['length'] for x in rr['stats'][m])),
            'Bateria [Wh]': round(sum(x['battery'] for x in rr['stats'][m]), 2)
        })
    st.table(pd.DataFrame(summary))

    with st.expander("🔍 Szczegóły Decyzji (Logi Rankingów)"):
        for d in rr['debug']:
            st.markdown(f"**Etap {d['leg']}**")
            st.dataframe(d['df'].sort_values('s_ahp', ascending=False), hide_index=True)