"""
proprietà Istituto di Analisi dei Sistemi ed Informatica "Antonio
Ruberti" del Consiglio Nazionale delle Ricerche

10/12/2023   versione v. 1.0

Autori:
Diego Maria Pinto
Marco Boresta
Giuseppe Stecca
Giovanni Felici
"""

import os
import json
import warnings
warnings.simplefilter("ignore")
#################################
import pandas as pd
#################################
import streamlit as st
import folium
from streamlit_folium import folium_static
import warnings
warnings.simplefilter("ignore")
from folium import plugins
from App_function_file import load_preprocessed_trajs_data
import numpy as np
import geojson

def create_map(truck_id, serial_path, origin, destination):
    tiles = None
    m = folium.Map(location=[46.09338, 13.44000], tiles=tiles, zoom_start=7)

    # aggiungo un layer "piastrella" (tyle) come specificato di seguito
    tiles_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    tile_layer = folium.TileLayer(
        tiles=tiles_url,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        max_zoom=19,
        name='Positron',
        control=False,
        opacity=0.7
    )
    tile_layer.add_to(m)

    data_path = os.getcwd() + '/Data/'
    geojson_filename = 'CNTR_BN_10M_2020_4326.geojson'
    with open(data_path + geojson_filename, 'r') as geojson_file:
        EU_countries_borders_layer = json.load(geojson_file)

    style = {'fillColor': '#00000000', 'linecolor': 'grey'}
    folium.GeoJson(EU_countries_borders_layer, style_function=lambda x: style).add_to(m)

    timestamped_geojson_name = 'truck_id_' + str(truck_id) + '_serial_' + str(serial_path) + '_from_' + origin + '_to_' + destination + '.geojson'
    file_path = os.getcwd() + '/Data/Geojson_paths/'
    if os.path.exists(file_path + timestamped_geojson_name):
        with open(file_path + timestamped_geojson_name) as file:
            timestamped_geojson = geojson.load(file)

    plugins.TimestampedGeoJson({'type': 'FeatureCollection',
                                'features': timestamped_geojson['data']['features']},
                               period='PT1M',
                               add_last_point=True,
                               transition_time=30,
                               auto_play=True,
                               loop=False,
                               max_speed=40,
                               loop_button=True,
                               time_slider_drag_update=True,
                               ).add_to(m)

    m.save(os.path.join('map_with_gps_locations.html'))

    return m

def get_traj_datetimelist(truck_id, serial_path, origin, destination):
    timestamped_geojson_name = 'truck_id_' + str(truck_id) + '_serial_' + str(serial_path) + '_from_' + origin + '_to_' + destination + '.geojson'
    file_path = os.getcwd() + '/Data/Geojson_paths/'
    if os.path.exists(file_path + timestamped_geojson_name):
        with open(file_path + timestamped_geojson_name) as file:
            timestamped_geojson = geojson.load(file)
    traj_datetime_list = timestamped_geojson['data']['features'][0]['properties']['times']
    return traj_datetime_list

def NormalizeData(data):
    return list((data - np.min(data)) / (np.max(data) - np.min(data)))


def main():
    cwd = os.getcwd()
    st.set_page_config(layout="wide")

    # LOAD DATA
    ### Load preprocessed data of all trajectories ###
    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    file_name = 'preprocessed_data_22_23.xlsx'
    df_trajs = load_preprocessed_trajs_data(data_path, file_name)

    ### Load statistics of processed data ###
    data_path = cwd
    file_name = 'ARPA_logistics_metrics_v2.0.xlsx'
    df_base = pd.read_excel(data_path + '/' + file_name)

    ######### FILTRO A MONTE ##########
    df_base = df_base[df_base['from'] == 'Samsung']
    df_base = df_base[df_base['to'] == 'Gerbole']
    ######################## ##########

    df_scenario_1 = pd.read_excel(data_path + '/' + 'ARPA_logistics_metrics_v2.0_Scenario_1_non_critico.xlsx')
    df_scenario_2 = pd.read_excel(data_path + '/' + 'ARPA_logistics_metrics_v2.0_Scenario_2_al_limite.xlsx')
    df_scenario_3 = pd.read_excel(data_path + '/' + 'ARPA_logistics_metrics_v2.0_Scenario_3_critico.xlsx')
    ###################################################

    # DISPLAY FILTERS AND MAP
    with st.container():
        col1, col2, col3 = st.columns([2.5, 2.5, 5], gap="small")

        with col1:
            st.image(os.getcwd() + '/Data/Loghi/' + "immagine_CNR_iasi_logo.png", width = 160)
        with col2:
            st.image(os.getcwd() + '/Data/Loghi/' + "immagine_CRF-logo.png", width = 130)
        with col3:
            st.image(os.getcwd() + '/Data/Loghi/' + "immagine_logo_EU_PON.png", width = 600)

    with st.container():
        col1, col2, col3, col4 = st.columns([0.5, 1.5, 6,2], gap="small")

        with col1:
            # col1.header('Filtri Scenario')
            scenario_list = ['Non critico', 'Al limite', 'Critico']
            scenario = st.sidebar.selectbox('Seleziona uno scenario', options=scenario_list, index = 0)
            if scenario == 'Non critico':
                df = df_scenario_1
                soglia_perc_convogli_in_ritatdo = 10

            elif scenario == 'Al limite':
                df = df_scenario_2
                df_input = pd.read_excel(os.getcwd() + '/Data/' + 'File_HMI_Scenari_Limite_e_Critico.xlsx', sheet_name='Scenario_Limite')
                soglia_perc_convogli_in_ritatdo = df_input['Convogli in ritardo limite %'].iloc[0]

            else:
                df = df_scenario_3
                df_input = pd.read_excel(os.getcwd() + '/Data/' + 'File_HMI_Scenari_Limite_e_Critico.xlsx', sheet_name='Scenario_Critico')
                soglia_perc_convogli_in_ritatdo = df_input['Convogli in ritardo limite %'].iloc[0]
                df_input_critico_2 = pd.read_excel(os.getcwd() + '/Data/' + 'File_HMI_Scenari_Limite_e_Critico.xlsx', sheet_name='Scenario_Critico_1')
                soglia_perc_convogli_in_ritatdo_2 = df_input_critico_2['Convogli in ritardo limite %'].iloc[0]

            truck_id_list = list(df['truck_id'].unique())
            truck_id_list.sort()
            truck_id = st.sidebar.selectbox('Seleziona il Truck ID', options=truck_id_list)
            serial_path_list = list(df[df['truck_id'] == truck_id]['path_serial'].unique())
            serial_path_list.sort()
            serial_path = st.sidebar.selectbox('Seleziona il path serial', options=serial_path_list)
            origin_list = list(df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path)]['from'].unique())
            origin = st.sidebar.selectbox('Seleziona orgine del percorso', options=origin_list)
            destination_list = list(df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path) & (df['from'] == origin)]['to'].unique())
            destination = st.sidebar.selectbox('Seleziona destinazione del percorso', options=destination_list)

        with col2:
            col2.header('KPI di traccia')

            df_traj = df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path) & (df['from'] == origin) & (df['to'] == destination)]

            traj_datetime_list = get_traj_datetimelist(truck_id, serial_path, origin, destination)
            traj_datetime_0_1_list = NormalizeData(np.arange(0, len(traj_datetime_list)))

            traj_datetime_0_1 = st.select_slider('Seleziona una percentuale di avanzamento della spedizione', options=traj_datetime_0_1_list, value=0)
            timebar_list_index = traj_datetime_0_1_list.index(traj_datetime_0_1)

            tot_num_stops = int(df_traj['# stops'].iloc[0])
            tot_total_dur = float(df_traj['total_duration [hours]'].iloc[0])
            tot_driving_dur = int(df_traj['total_driving [hours]'].iloc[0])
            tot_stop_dur = float(df_traj['overall stops duration [hours]'].iloc[0])
            tot_delay_time = float(df_traj['dalay_time'].iloc[0])

            stops_list = list(np.arange(0, tot_num_stops + 1))
            dur_list = list(np.arange(0, tot_total_dur + 1))
            driving_dur_list = list(np.arange(0, tot_driving_dur + 1))
            delay_time_list =  list(np.arange(0, tot_delay_time + 1))

            num_stops = 0
            stop_dur = 0
            if traj_datetime_0_1 > 0.2 and traj_datetime_0_1 < 0.6 :
                num_stops = stops_list[1]
                stop_dur = float(df_traj['stop_1_time'])
            elif traj_datetime_0_1 > 0.6:
                num_stops = stops_list[-1]
                stop_dur = tot_stop_dur

            total_dur = round(dur_list[timebar_list_index] + stop_dur,2)
            driving_dur = driving_dur_list[timebar_list_index]

            delay_time = 0
            if traj_datetime_0_1 == 1:
                delay_time = tot_delay_time

            if num_stops == 1:
                st.metric(label="Fermate effettuate", value=str(num_stops) + ' fermata')
            else:
                st.metric(label="Fermate effettuate", value = str(num_stops) + ' fermate')
            st.metric(label="Durata complessiva", value = str(total_dur) + ' ore')
            st.metric(label="Tempo in movimento", value= str(driving_dur) + ' ore')
            st.metric(label="Tempo in sosta", value = str(stop_dur) + ' ore')
            st.metric(label="Ritardo accumulato all'arrivo", value = str(round(delay_time,2)) + ' ore')

            CW_target_time = 13.22
            if driving_dur > CW_target_time:
                text = 'CheckPoint Warning: in ritardo!'
                st.markdown(f'<h1 style="color:#e60707;font-size:18px;">{text}</h1>', unsafe_allow_html=True)
                dalay_time_CW = round(float(df_traj['dalay_time_CW'].iloc[0]),2)
                st.metric(label="Ritardo sul CheckPoint Warning", value=str(dalay_time_CW) + ' ore')
            else:
                text = 'CheckPoint Warning: OK'
                st.markdown(f'<h1 style="color:#00ff26;font-size:18px;">{text}</h1>', unsafe_allow_html=True)

            CA_target_time = 19.65
            if driving_dur > CA_target_time:
                text = 'CheckPoint Alert: in ritardo!'
                st.markdown(f'<h1 style="color:#e60707;font-size:18px;">{text}</h1>', unsafe_allow_html=True)
                dalay_time_CA = round(float(df_traj['dalay_time_CA'].iloc[0]),2)
                st.metric(label="Ritardo sul CheckPoint Alert", value=str(dalay_time_CA) + ' ore')
            else:
                text = 'CheckPoint Alert: OK'
                st.markdown(f'<h1 style="color:#00ff26;font-size:18px;">{text}</h1>', unsafe_allow_html=True)

        with col3:

            st.title('ARPA - Inbound Weekly Logistics')
            st.caption('Battery Supply Logistic Monotoring - KPIs - Scenarios and Track Visualization')

            scenario_start = df['start_daytime'].min()
            scenario_end = df['end_daytime'].max()
            timebar_list = pd.date_range(start=scenario_start, end=scenario_end, freq='2H', inclusive="both").to_list()
            timebar_list += [scenario_end]

            timebar_0_1_list = NormalizeData(np.arange(0, len(timebar_list)))

            datetime_0_1 = st.select_slider('Seleziona una percentuale di avanzamento della settimana', options=timebar_0_1_list, value=0)
            timebar_list_index = timebar_0_1_list.index(datetime_0_1)
            datetime = timebar_list[timebar_list_index]

            # col2.header(f'Scenario {scenario} - truck_id {truck_id} - serial path {serial_path} - from {origin} to {destination}')
            m = create_map(truck_id, serial_path, origin, destination)
            folium_static(m)

        with col4:

            perc_spedizioni_partite = round(df[df['start_daytime'] <= datetime].shape[0] / len(df),2)*100
            perc_spedizioni_in_corso = round(df[(df['start_daytime'] < datetime) & (df['end_daytime'] > datetime)].shape[0] / len(df),2)*100
            spedizioni_arrivate = df[df['end_daytime'] <= datetime]
            perc_spedizioni_arrivate = round(spedizioni_arrivate.shape[0] / len(df),2)*100

            perc_spedizioni_arrivate_in_ritardo = round(spedizioni_arrivate[spedizioni_arrivate['dalay_time'] > 0].shape[0] / len(df),2)*100

            col4.header('KPI di scenario')

            KP1_s = st.metric(label="% Spedizioni partite", value = round(perc_spedizioni_partite,2))
            KP2_s = st.metric(label="% Spedizioni in corso", value = round(perc_spedizioni_in_corso,2))
            KP3_s = st.metric(label="% Spedizioni arrivate", value = round(perc_spedizioni_arrivate,2))
            KP4_s = st.metric(label="% Spedizioni arrivate in ritardo", value = round(perc_spedizioni_arrivate_in_ritardo,2))

            simulazione_attivata = False
            button_enabler = True
            if perc_spedizioni_arrivate_in_ritardo > soglia_perc_convogli_in_ritatdo:
                button_enabler = False
            if button_enabler == True:
                st.write('Simulazione non necessaria')
            else:
                st.write('Simulazione necessaria!')

            if st.button("Avvia la simulazione", type="primary", disabled = button_enabler):
                simulazione_attivata = True
                st.write('Simulazione completata')

                if 'input' not in st.session_state:
                    st.session_state.input = True

                def reset():
                    st.session_state.input = True
                    simulazione_attivata = False

                if st.session_state.input:
                    # Show input widgets if in input mode
                    st.write(f'Osserva in basso i risultati della simulazione')
                    st.button('Resetta i risultati', on_click=reset)
                else:
                    # Otherwise, not in input mode, so show result
                    st.write(f'Osserva in basso i risultati della simulazione')
                    st.button('Resetta i risultati', on_click=reset)  # Callback changes it to input mode

    if simulazione_attivata == True:
        if scenario == 'Critico' and perc_spedizioni_arrivate_in_ritardo > soglia_perc_convogli_in_ritatdo_2:
            df_input = df_input_critico_2

        ########## print e plots risultati simulazione #####################
        warning_text = df_input['Messaggi HMI - Warning'].iloc[0]
        action_text = df_input['Messaggi HMI - Azione'].iloc[0]
        prod_sett_schedulata = df_input['produzione sett. schedulata'].iloc[0]
        prod_sett_simulata = df_input['produzione sett. Simulata'].iloc[0]
        perdita_produttiva = df_input['perdita produttiva'].iloc[0]
        scorta_limite = df_input['Livello scorta limite'].iloc[0]
        scorta_simulato = df_input['Livello scorta simulato'].iloc[0]
        ####################################################################
        with st.container():
            col1, col2, col3, col4 = st.columns([3.5, 3.5, 1.5, 1.5], gap="small")

            with col1:
                st.write('Produzione Settimanale')
                chart_data = pd.DataFrame([['perdita produttiva', perdita_produttiva, "#969391"],
                                           ['prod. sett. Simulata',  prod_sett_simulata,  "#df6b04"],
                                           ['prod. sett. schedulata',prod_sett_schedulata, "#2600fc"]], columns = ['Week','Produzione','colore'])

                st.bar_chart(chart_data, x='Week', y='Produzione', color='colore', use_container_width=True)
            with col2:
                st.write('Scorta Magazzino Gerbole')

                chart_data = pd.DataFrame([['Liv. Scorta Limite', scorta_limite, "#2600fc"],
                                           ['Liv. Scorta Simulato',  scorta_simulato,  "#df6b04"]],
                                          columns = ['Week','Produzione','colore'])
                st.bar_chart(chart_data, x='Week', y='Produzione', color='colore', use_container_width=True)

            with col3:
                st.markdown(f'<h1 style="color:#ff3333;font-size:18px;">{warning_text}</h1>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<h1 style="color:#2600fc;font-size:18px;">{action_text}</h1>', unsafe_allow_html=True)

if __name__ == '__main__':

    main()

    ### Load statistics of processed data ###
    data_path = os.getcwd()
    file_name = 'ARPA_logistics_metrics_v2.0.xlsx'
    df = pd.read_excel(data_path + '/' + file_name)
    ###################################################

    df_scenario_1 = pd.read_excel(data_path + '/' + 'ARPA_logistics_metrics_v2.0_Scenario_1_non_critico.xlsx')
    df_scenario_2 = pd.read_excel(data_path + '/' + 'ARPA_logistics_metrics_v2.0_Scenario_2_al_limite.xlsx')
    df_scenario_3 = pd.read_excel(data_path + '/' + 'ARPA_logistics_metrics_v2.0_Scenario_3_critico.xlsx')

    scenario_start = df['start_daytime'].min()
    scenario_end = df['end_daytime'].max()
    timebar_list = pd.date_range(start=scenario_start, end=scenario_end, freq='2H', inclusive="both").to_list()
    timebar_list += [scenario_end]

    truck_id = 3
    serial_path = 0
    origin = 'Samsung'
    destination = 'Gerbole'
    df_traj = df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path) & (df['from'] == origin) & (df['to'] == destination)]
    num_stops = int(df_traj['# stops'])
    total_dur = float(df_traj['total_duration [hours]'])
    driving_dur = int(df_traj['total_driving [hours]'])
    stop_dur = float(df_traj['overall stops duration [hours]'])
    delay_time = float(df_traj['dalay_time'])

    delay_time = float(df_traj['dalay_time'])

    timestamped_geojson_name = 'truck_id_' + str(truck_id) + '_serial_' + str(serial_path) + '_from_' + origin + '_to_' + destination + '.geojson'
    file_path = os.getcwd() + '/Data/Geojson_paths/'
    if os.path.exists(file_path + timestamped_geojson_name):
        with open(file_path + timestamped_geojson_name) as file:
            timestamped_geojson = geojson.load(file)




