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
import random
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.features import DivIcon
from streamlit_folium import folium_static
import warnings
warnings.simplefilter("ignore")
from folium import plugins
from App_function_file import load_preprocessed_trajs_data
import numpy as np
import geojson

def initialize_session_state():
    if "center" not in st.session_state:
        st.session_state["center"] = [46.09338, 13.44000]
    if "zoom" not in st.session_state:
        st.session_state["zoom"] = 7
    if "week_0_1_slider" not in st.session_state:
        st.session_state["week_0_1_slider"] = 0
    if "traj_0_1_slider" not in st.session_state:
        st.session_state["traj_0_1_slider"] = 0


def reset_session_state():
    # Delete all the items in Session state besides center and zoom
    for key in st.session_state.keys():
        if key in ["center", "zoom"]:
            continue
        del st.session_state[key]
    initialize_session_state()

def initialize_map(truck_id, serial_path, origin, destination, traj_0_1_index):
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


    long_CW = 13.775175598014489  # CheckPoint Warning - Trieste
    long_CA = 9.690197660608838  # CheckPoint Alert - Piacenza / Bergamo

    CW_points = [(46.5,long_CW), (45.65 ,long_CW)]
    folium.PolyLine(CW_points, color="orange").add_to(m)

    CA_points = [(46.3, long_CA), (44.115, long_CA)]
    folium.PolyLine(CA_points, color="orange").add_to(m)

    folium.map.Marker(
        [46.5, long_CW],
        icon=DivIcon(
            icon_size=(10, 10),
            icon_anchor=(0, 0),
            html='<div style="font-size: 10pt; color:darkorange;"><b>CheckPoint Warning</b></div>',
        )
    ).add_to(m)

    folium.map.Marker(
        [46.3, long_CA],
        icon=DivIcon(
            icon_size=(30, 30),
            icon_anchor=(0, 0),
            html='<div style="font-size: 10pt; color:darkorange;"><b>CheckPoint Alert</b></div>',
        )
    ).add_to(m)

    if traj_0_1_index == 0:
        traj_0_1_index = 1

    timestamped_geojson['data']['features'][0]['properties']['times'] = timestamped_geojson['data']['features'][0]['properties']['times'][:traj_0_1_index]
    timestamped_geojson['data']['features'][0]['geometry']['coordinates'] = timestamped_geojson['data']['features'][0]['geometry']['coordinates'][:traj_0_1_index]

    # plugins.TimestampedGeoJson({'type': 'FeatureCollection',
    #                             'features': timestamped_geojson['data']['features']},
    #                            period='PT1M',
    #                            add_last_point=True,
    #                            auto_play=False,
    #                            loop=False,
    #                            loop_button=False,
    #                            time_slider_drag_update=False,
    #                            ).add_to(m)

    traj_coordinates = timestamped_geojson['data']['features'][0]['geometry']['coordinates'][:traj_0_1_index]
    traj_times = timestamped_geojson['data']['features'][0]['properties']['times'][:traj_0_1_index]

    def reverse_lon_lat(x):
        a = [[p[1], p[0]] for p in x]
        return a

    traj_coordinates = reverse_lon_lat(traj_coordinates)

    folium.PolyLine(
        locations=traj_coordinates,
        color="#FF0000",
        weight=5,
    ).add_to(m)

    folium.map.Marker(location=traj_coordinates[-1], popup="Test", icon=folium.Icon(icon='truck', prefix='fa', color="#FF0000")).add_to(m)

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

def reset_sliders():
    st.session_state.traj_0_1_slider = 0
    st.session_state.week_0_1_slider = 0
    return

def reset_traj_slider():
    st.session_state.traj_0_1_slider = 0
    return

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

    df_scenario_1.loc[df_scenario_1['from'] == 'Samsung', 'from'] = 'Stab. Pacchi Batterie'
    df_scenario_2.loc[df_scenario_2['from'] == 'Samsung', 'from'] = 'Stab. Pacchi Batterie'
    df_scenario_3.loc[df_scenario_3['from'] == 'Samsung', 'from'] = 'Stab. Pacchi Batterie'
    df_scenario_1.loc[df_scenario_1['to'] == 'Gerbole', 'to'] = 'Magazzino Principale'
    df_scenario_2.loc[df_scenario_2['to'] == 'Gerbole', 'to'] = 'Magazzino Principale'
    df_scenario_3.loc[df_scenario_3['to'] == 'Gerbole', 'to'] = 'Magazzino Principale'
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
        col1, col2, col3, col4 = st.columns([0.5, 1.5, 6, 2], gap="small")

        initialize_session_state()

        with col1:
            scenario_list = ['Non critico', 'Al limite', 'Critico']

            scenario = st.sidebar.selectbox('Seleziona uno scenario', options=scenario_list, index = 0, key = 'scenario', on_change=reset_sliders)
            if scenario == 'Non critico':
                df = df_scenario_1
                soglia_perc_convogli_in_ritatdo = 20

            elif scenario == 'Al limite':
                df = df_scenario_2
                df_input = pd.read_excel(os.getcwd() + '/Data/' + 'File_HMI_Scenari_Limite_e_Critico_rev1.xlsx', sheet_name='Scenario_Limite')
                soglia_perc_convogli_in_ritatdo = df_input['Convogli in ritardo limite %'].iloc[0]

            else:
                df = df_scenario_3
                df_input_critico_0 = pd.read_excel(os.getcwd() + '/Data/' + 'File_HMI_Scenari_Limite_e_Critico_rev1.xlsx', sheet_name='Scenario_Limite')
                soglia_perc_convogli_in_ritatdo = df_input_critico_0['Convogli in ritardo limite %'].iloc[0]
                df_input_critico_1 = pd.read_excel(os.getcwd() + '/Data/' + 'File_HMI_Scenari_Limite_e_Critico_rev1.xlsx', sheet_name='Scenario_Critico')
                soglia_perc_convogli_in_ritatdo_1 = df_input_critico_1['Convogli in ritardo limite %'].iloc[0]
                df_input_critico_2 = pd.read_excel(os.getcwd() + '/Data/' + 'File_HMI_Scenari_Limite_e_Critico_rev1.xlsx', sheet_name='Scenario_Critico_1')
                soglia_perc_convogli_in_ritatdo_2 = df_input_critico_2['Convogli in ritardo limite %'].iloc[0]


            ##########################################
            # reset_session_state()
            ##########################################
            truck_id_list = list(df['truck_id'].unique())
            truck_id_list.sort()
            truck_id = st.sidebar.selectbox('Seleziona il Truck ID', options=truck_id_list, key='truck_id', on_change=reset_traj_slider)
            serial_path_list = list(df[df['truck_id'] == truck_id]['path_serial'].unique())
            serial_path_list.sort()
            serial_path = st.sidebar.selectbox('Seleziona il path serial', options=serial_path_list, key='serial_path', on_change=reset_traj_slider)
            origin_list = list(df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path)]['from'].unique())
            origin = st.sidebar.selectbox('Seleziona orgine del percorso', options=origin_list, key='origin', on_change=reset_traj_slider)
            destination_list = list(df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path) & (df['from'] == origin)]['to'].unique())
            destination = st.sidebar.selectbox('Seleziona destinazione del percorso', options=destination_list, key='destination', on_change=reset_traj_slider)

        with col2:
            col2.header('KPI di traccia')

            df_traj = df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path) & (df['from'] == origin) & (df['to'] == destination)]

            if origin == 'Stab. Pacchi Batterie':
                origin = 'Samsung'
            if destination == 'Magazzino Principale':
                destination = 'Gerbole'

            traj_datetime_list = get_traj_datetimelist(truck_id, serial_path, origin, destination)

            timebar_0_1_list = NormalizeData(np.arange(0, 21))
            traj_0_1_slider = st.select_slider('Seleziona una percentuale di avanzamento della spedizione', options=timebar_0_1_list,  key='traj_0_1_slider')
            if traj_0_1_slider == 1:
                traj_0_1_index = len(traj_datetime_list) - 1
            else:
                traj_0_1_index = int(traj_0_1_slider * len(traj_datetime_list))

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
            if traj_0_1_slider > 0.2 and traj_0_1_slider <= 0.6 :
                num_stops = stops_list[1]
                stop_dur = float(df_traj['stop_1_time'].iloc[0])
            elif traj_0_1_slider > 0.6:
                num_stops = stops_list[-1]
                stop_dur = tot_stop_dur

            if traj_0_1_slider != 1:
                driving_dur = driving_dur_list[int(traj_0_1_slider * len(driving_dur_list))]
            else:
                driving_dur = driving_dur_list[-1]

            total_dur = driving_dur + stop_dur

            # if traj_0_1_slider != 1:
            #     total_dur = round(dur_list[int(traj_0_1_slider * len(dur_list))] + stop_dur, 2)
            #     # total_dur = round(dur_list[traj_0_1_index] + stop_dur,2)
            # else:
            #     total_dur = driving_dur + stop_dur

            delay_time = 0
            if traj_0_1_slider == 1:
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
            if (driving_dur > CW_target_time) and (df_traj['dalay_time_CW'].iloc[0] > 0):
                text = 'CheckPoint Warning: in ritardo!'
                st.markdown(f'<h1 style="color:#e60707;font-size:18px;">{text}</h1>', unsafe_allow_html=True)
                dalay_time_CW = round(float(df_traj['dalay_time_CW'].iloc[0]),2)
                st.metric(label="Ritardo sul CheckPoint Warning", value=str(dalay_time_CW) + ' ore')
            else:
                text = 'CheckPoint Warning: OK'
                st.markdown(f'<h1 style="color:#00ff26;font-size:18px;">{text}</h1>', unsafe_allow_html=True)

            CA_target_time = 19.65
            if (driving_dur > CA_target_time) and (df_traj['dalay_time_CA'].iloc[0] > 0):
                text = 'CheckPoint Alert: in ritardo!'
                st.markdown(f'<h1 style="color:#e60707;font-size:18px;">{text}</h1>', unsafe_allow_html=True)
                dalay_time_CA = round(float(df_traj['dalay_time_CA'].iloc[0]),2)
                st.metric(label="Ritardo sul CheckPoint Alert", value=str(dalay_time_CA) + ' ore')
            else:
                text = 'CheckPoint Alert: OK'
                st.markdown(f'<h1 style="color:#00ff26;font-size:18px;">{text}</h1>', unsafe_allow_html=True)

        with col3:

            st.title('ARPA - Inbound Weekly Logistics')
            st.caption('Battery Supply Logistic Monitoring - KPIs - Scenarios and Track Visualization')

            scenario_start = df['start_daytime'].min()
            scenario_end = df['end_daytime'].max()
            timebar_list = pd.date_range(start=scenario_start, end=scenario_end, freq='12H', inclusive="both").to_list()
            timebar_list += [scenario_end]

            timebar_0_1_list = NormalizeData(np.arange(0, 21))

            week_0_1_slider = st.select_slider('Seleziona una percentuale di avanzamento della settimana', options=timebar_0_1_list, key = 'week_0_1_slider', on_change=reset_traj_slider)

            if week_0_1_slider == 1:
                week_0_1_index = len(timebar_list) - 1
            else:
                week_0_1_index = int(week_0_1_slider * len(timebar_list))

            datetime = timebar_list[week_0_1_index]

            initialize_session_state()

            m = initialize_map(truck_id, serial_path, origin, destination, traj_0_1_index)
            folium_static(m)


        with col4:

            perc_spedizioni_partite = round(df[df['start_daytime'] <= datetime].shape[0] / len(df),2)*100
            perc_spedizioni_in_corso = round(df[(df['start_daytime'] < datetime) & (df['end_daytime'] > datetime)].shape[0] / len(df),2)*100
            spedizioni_arrivate = df[df['end_daytime'] <= datetime]
            perc_spedizioni_arrivate = round(spedizioni_arrivate.shape[0] / len(df),2)*100

            perc_spedizioni_arrivate_in_ritardo = round(spedizioni_arrivate[spedizioni_arrivate['dalay_time'] > 0].shape[0] / len(df),2)*100

            col4.header('KPI di scenario')

            if week_0_1_slider == 0:
                perc_spedizioni_partite = 0

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

                def reset_simulation():
                    st.session_state.input = True
                    simulazione_attivata = False

                if st.session_state.input:
                    # Show input widgets if in input mode
                    st.write(f'Osserva in basso i risultati della simulazione')
                    st.button('Resetta i risultati', on_click=reset_simulation)
                else:
                    # Otherwise, not in input mode, so show result
                    st.write(f'Osserva in basso i risultati della simulazione')
                    st.button('Resetta i risultati', on_click=reset_simulation)  # Callback changes it to input mode

    if simulazione_attivata == True:

        if st.session_state.scenario == 'Critico':
            if perc_spedizioni_arrivate_in_ritardo > soglia_perc_convogli_in_ritatdo:
                df_input = df_input_critico_0
            if perc_spedizioni_arrivate_in_ritardo > soglia_perc_convogli_in_ritatdo_1:
                df_input = df_input_critico_1
            if perc_spedizioni_arrivate_in_ritardo > soglia_perc_convogli_in_ritatdo_2:
                df_input = df_input_critico_2


        ########## print e plots risultati simulazione #####################
        warning_text = df_input['Messaggi HMI - Warning'].iloc[0]
        action_text = df_input['Messaggi HMI - Azione'].iloc[0]
        prod_sett_schedulata = df_input['produzione sett. Schedulata %'].iloc[0]
        prod_sett_simulata = df_input['produzione sett. Simulata %'].iloc[0]
        perdita_produttiva = df_input['perdita produttiva %'].iloc[0]
        scorta_limite = df_input['Livello scorta limite %'].iloc[0]
        scorta_simulato = df_input['Livello scorta simulato %'].iloc[0]
        ####################################################################
        with st.container():
            col1, col2, col3, col4 = st.columns([3.5, 3.5, 1.5, 1.5], gap="small")

            with col1:
                st.write('Produzione Settimanale')
                chart_data = pd.DataFrame([['Perdita Produtt. %', perdita_produttiva, "#969391"],
                                           ['Prod.Sett. Simulata %',  prod_sett_simulata,  "#df6b04"],
                                           ['Prod.Sett. Schedulata %',prod_sett_schedulata, "#2600fc"]], columns = ['Week','Produzione %','colore'])

                st.bar_chart(chart_data, x='Week', y='Produzione %', color='colore', use_container_width=True)
            with col2:
                st.write('Scorta Magazzino Ottimizzato')

                chart_data = pd.DataFrame([['Liv.Scorta Limite %', scorta_limite, "#2600fc"],
                                           ['Liv.Scorta Simulato %',  scorta_simulato,  "#df6b04"]],
                                          columns = ['Week','Giacenza %','colore'])
                st.bar_chart(chart_data, x='Week', y='Giacenza %', color='colore', use_container_width=True)

            with col3:
                st.markdown(f'<h1 style="color:#ff3333;font-size:18px;">{warning_text}</h1>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<h1 style="color:#2600fc;font-size:18px;">{action_text}</h1>', unsafe_allow_html=True)

if __name__ == '__main__':


    # entra sul terminale git bash, apri la cartella arpacnriasi, quindi fai git pull
    main()

    #######################################################################################################################
    ############################################# MAIN for debug ##########################################################
    #######################################################################################################################

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

    truck_id = 26
    serial_path = 0
    origin = 'Samsung'
    destination = 'Gerbole'
    df_traj = df[(df['truck_id'] == truck_id) & (df['path_serial'] == serial_path) & (df['from'] == origin) & (df['to'] == destination)]

    tot_num_stops = int(df_traj['# stops'].iloc[0])
    tot_total_dur = float(df_traj['total_duration [hours]'].iloc[0])
    tot_driving_dur = int(df_traj['total_driving [hours]'].iloc[0])
    tot_stop_dur = float(df_traj['overall stops duration [hours]'].iloc[0])
    tot_delay_time = float(df_traj['dalay_time'].iloc[0])

    stops_list = list(np.arange(0, tot_num_stops + 1))
    dur_list = list(np.arange(0, tot_total_dur + 1))
    driving_dur_list = list(np.arange(0, tot_driving_dur + 1))
    delay_time_list = list(np.arange(0, tot_delay_time + 1))

    num_stops = 0
    stop_dur = 0
    traj_0_1_slider = 0.5

    traj_datetime_list = get_traj_datetimelist(truck_id, serial_path, origin, destination)
    if traj_0_1_slider == 1:
        traj_0_1_index = len(traj_datetime_list) - 1
    else:
        traj_0_1_index = int(traj_0_1_slider * len(traj_datetime_list))

    if traj_0_1_slider > 0.2 and traj_0_1_slider <= 0.6:
        num_stops = stops_list[1]
        stop_dur = float(df_traj['stop_1_time'].iloc[0])
    elif traj_0_1_slider > 0.6:
        num_stops = stops_list[-1]
        stop_dur = tot_stop_dur

    total_dur = round(dur_list[traj_0_1_index] + stop_dur, 2)

    if traj_0_1_slider != 1:
        driving_dur = driving_dur_list[int(traj_0_1_slider * len(driving_dur_list))]
    else:
        driving_dur = driving_dur_list[-1]

    total_dur = driving_dur + stop_dur

    delay_time = 0
    if traj_0_1_slider == 1:
        delay_time = tot_delay_time


    timestamped_geojson_name = 'truck_id_' + str(truck_id) + '_serial_' + str(serial_path) + '_from_' + origin + '_to_' + destination + '.geojson'
    file_path = os.getcwd() + '/Data/Geojson_paths/'
    if os.path.exists(file_path + timestamped_geojson_name):
        with open(file_path + timestamped_geojson_name) as file:
            timestamped_geojson = geojson.load(file)

    scenario_start = df_scenario_1['start_daytime'].min()
    scenario_end = df_scenario_1['end_daytime'].max()
    timebar_list = pd.date_range(start=scenario_start, end=scenario_end, freq='12H', inclusive="both").to_list()
    timebar_list += [scenario_end]

    timebar_0_1_list = NormalizeData(np.arange(0, len(timebar_list)))
    timebar_0_1_list_to_visualize = NormalizeData(np.arange(0, 21))

    #bypass_0_1_slider   = st.select_slider('Seleziona una percentuale di avanzamento della settimana',options=timebar_0_1_list_to_visualize, value=0)

    bypass_0_1_slider = 1
    if bypass_0_1_slider == 1:
        bypass_index = len(timebar_list) - 1
    else:
        bypass_index = int(bypass_0_1_slider*len(timebar_list))

    datetime = timebar_list[bypass_index]

    traj_lines = timestamped_geojson['data']['features'][0]['geometry']['coordinates']

    traj_lines_new = [tuple(x) for x in traj_lines]

    m = initialize_map(truck_id, serial_path, origin, destination, 6)

    # output_file = "map_temp.html"
    # m.save(output_file)








