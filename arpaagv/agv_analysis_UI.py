"""
proprietà Istituto di Analisi dei Sistemi ed Informatica "Antonio
Ruberti" del Consiglio Nazionale delle Ricerche

07/12/2023   versione v. 1.0

Autori:
Marco Boresta
Diego Maria Pinto
Giuseppe Stecca
Giovanni Felici
"""



from operator import itemgetter
from itertools import groupby
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch import nn, optim
import torch
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    image_width = 30  # You can adjust this value to your desired width
    # st.image("immagine_CNR_iasi_logo.png", use_column_width=False, width=image_width)
    st.image("immagine_CNR_iasi_logo.png")  # Replace with the path to your first logo

with col2:
    
    st.title("Dimostratore Progetto ARPA")

with col3:
    # st.image("immagine_CRF-logo.png", use_column_width=False, width=image_width)
    st.image("immagine_CRF-logo.png")  # Replace with the path to your second logo

# test_mission_indices = st.multiselect(
#     "Select the mission indices for testing:", list(range(0, 9)))
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 'Real Time'

tab_choice = st.selectbox("Choose Tab", ["Real Time", "Statistical analysis", "Machine Learning"])
st.session_state['active_tab'] = tab_choice

tab1, tab2, tab3 = st.tabs(["Real Time", "Statistical analysis", "Machine Learning"])


if st.session_state['active_tab'] == 'Real Time':

    with tab1:

        np.random.seed(8)

        # Define constants
        time_series_length = 3 * 60  # 5 minutes
        frequency_seconds = 1  # frequency of updates in seconds
        time_to_zero = 120  # Time after which values should go near zero for mission type 1
        period_outside_upper_confidence = 10  # Maximum period to be outside upper confidence for mission type 2
        update_period = 10  # Period to check for the condition to go outside the confidence interval

        # Define mission types and confidence intervals
        mission_type = st.sidebar.radio(
            "Che tipo di missione vuoi mostrare?",
            [
                (1, 'Funzionamento normale'),
                (2, 'Funzionamento al limite'),
                (3, 'Funzionamento critico')
            ],
            format_func=lambda x: x[1]
        )
        # Mission type and confidence intervals setup
        # mission_number = st.sidebar.selectbox("What type of missions are you doing?", (1, 2, 3, 4))
        mission_options = {
    1: "Missione con carico 0 Kg",
    2: "Missione con carico 10 Kg",
    3: "Missione con carico 20 Kg"}

        st.sidebar.markdown("Qual è il carico in questa missione? <br> (Bande di funzionamento aggiornate)", unsafe_allow_html=True)
        mission_number = st.sidebar.selectbox("", mission_options.keys(), format_func=lambda x: mission_options[x])


        confidence_intervals_group1 = {
            1: [(0.67, 1.78), (0.70, 1.70), (0.76, 1.5), (0.92, 1.7)],
            2: [(0.65, 1.73), (0.65, 1.74), (0.66, 1.58), (0.81, 1.68)],
            3: [(0.68, 1.75), (0.68, 1.75), (0.64, 1.54), (0.80, 1.71)],
            4: [(1.3, 2.1), (1.5, 2.0), (1.2, 2.9), (1.4, 2.2)]
        }
        confidence_intervals_group2 = {
            1: [(14.74, 21), (21.5, 34.6), (17.31, 21.75), (24.56, 34.68)],
            2: [(13.35, 23.17), (19.43, 36.70), (14.52, 23.42), (20.9, 35.9)],
            3: [(13.5, 24), (20.3, 36.9), (14.63, 23.34), (21.05, 35.62)],
            4: [(23, 32), (25, 30), (22, 39), (24, 33)]
        }


        y_lim_lb_1 = y_lim_lb_2 = 0
        y_lim_ub_1 = max([xx[1] for xx in confidence_intervals_group1[mission_number]])
        y_lim_ub_2 = max([xx[1] for xx in confidence_intervals_group2[mission_number]])

        # Initialize the data arrays
        x = np.linspace(0, time_series_length, time_series_length)
        y_group1 = np.full((4, time_series_length), np.nan)
        y_group2 = np.full((4, time_series_length), np.nan)






        # Colors for different plots
        colors = ['red', 'orange', 'green', 'blue']

        # Titles for the motors
        motor_titles = ["Motore 1", "Motore 2", "Motore 3", "Motore 4"]

        # Function to create and update plots
        def create_and_update_plots(axs, motor_titles, y_values, confidence_intervals, group):
            for i, ax in enumerate(axs):
                lower_confidence, upper_confidence = confidence_intervals[i]
                ax.clear()
                ax.set_title(motor_titles[i])
                ax.plot(x, y_values[i], color=colors[i])
                ax.axhline(y=lower_confidence, color='gray', linestyle='--')
                ax.axhline(y=upper_confidence, color='gray', linestyle='--')
                ax.fill_between(x, lower_confidence, upper_confidence, color='gray', alpha=0.5)
                ax.set_ylim([y_lim_lb_1-0.1 if group == 1 else y_lim_lb_2-1, y_lim_ub_1+1 if group == 1 else y_lim_ub_2+1])
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Value (mg)" if group == 1 else "Value (mm/s)")
            plt.tight_layout()

        time_to_zero = 3 * 30  # 2 minutes expressed in seconds


        motor_flags = {}
        for motor in ['Motore 1', 'Motore 2', 'Motore 3', 'Motore 4']:
            motor_flags[motor] = True






        # Start button and real-time updating logic

        left_column, right_column = st.columns([2.5, 4])  # Adjust the ratio if needed


        with left_column:
            
            agv_status_placeholder = st.empty()
            agv_status = "🟢 In movimento"  # Initialize with the AGV moving status
            agv_status_placeholder.markdown(f"### STATO AGV\n{agv_status}")


            st.markdown("### STATO MOTORI") 
        # Placeholders for each motor status
            motor_status_placeholders = {
                "Motore 1": st.empty(),
                "Motore 2": st.empty(),
                "Motore 3": st.empty(),
                "Motore 4": st.empty()
            }
            
            ## Initialize the motor statuses to green (moving)
            for motor, placeholder in motor_status_placeholders.items():
                placeholder.markdown(f"- {motor}: 🟢")

            st.image("immagine_motori.png", use_column_width=True)


        with right_column:
            
            

            # Set up placeholders for the plots
            plot_placeholder_group1 = st.empty()
            plot_placeholder_group2 = st.empty()

            # Set up the figure and axes for the plots
            fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
            fig2, axs2 = plt.subplots(2, 2, figsize=(10, 8))
            fig1.subplots_adjust(hspace=0.3, wspace=0.2)
            fig2.subplots_adjust(hspace=0.3, wspace=0.2)



        if st.button('Start'):
            # Initialize the last value to be in the middle of the confidence intervals
            last_values_group1 = [(ci[1] + ci[0]) / 2.0 for ci in confidence_intervals_group1[mission_number]]
            last_values_group2 = [(ci[1] + ci[0]) / 2.0 for ci in confidence_intervals_group2[mission_number]]









            if mission_type[0] == 1:
                # Loop to update the plots in real-time
                for i in range(time_series_length):
                    current_time = i


                    # Simulate new data for both groups
                    for motor_index in range(4):
                        if current_time < time_to_zero:
                        # Generate random step within a range and add to last value
                            step = np.random.uniform(-0.1, 0.1)
                            y_group1[motor_index, i] = np.clip(last_values_group1[motor_index] + step,
                                                            *confidence_intervals_group1[mission_number][motor_index])
                            last_values_group1[motor_index] = y_group1[motor_index, i]

                            step = np.random.uniform(-0.1, 0.1)*10
                            y_group2[motor_index, i] = np.clip(last_values_group2[motor_index] + step,
                                                            *confidence_intervals_group2[mission_number][motor_index])
                            last_values_group2[motor_index] = y_group2[motor_index, i]
                        else:
                            y_group1[motor_index, i]= np.random.uniform(0, 0.1)
                            y_group2[motor_index, i]= np.random.uniform(0, 0.1)*10
                            
                            if current_time < time_to_zero + 30:
                                agv_status = "🔴 Fermo. Possibile problema di collisione"  # Update to the AGV stopped status

                            else:
                                agv_status = "🔴 Fermo. Verificare alimentazione AGV"  # Update to the AGV stopped status
                            agv_status_placeholder.markdown(f"### STATO AGV\n{agv_status}")



                            for motor, placeholder in motor_status_placeholders.items():
                                flag = motor_flags[motor]
                                color = "🟢" if flag else f'🟠 Potenziale problema sul '+str(motor)
                                placeholder.markdown(f'<span style="color: lightgrey;">- {motor}: {color}</span>', unsafe_allow_html=True)





            # Update the plots for group 1
                    create_and_update_plots(axs1.flatten(), motor_titles, y_group1, confidence_intervals_group1[mission_number], 1)
                    plot_placeholder_group1.pyplot(fig1)

                    # Update the plots for group 2
                    create_and_update_plots(axs2.flatten(), motor_titles, y_group2, confidence_intervals_group2[mission_number], 2)
                    plot_placeholder_group2.pyplot(fig2)

                    # Update lightstop status if needed...
                    # Wait for the specified frequency
                    time.sleep(frequency_seconds/3)



            elif mission_type[0] == 2:
                # Loop to update the plots in real-time

                target_motor = np.random.randint(0,4)

                start_out_of_bounds = 10+np.random.randint(10)
                end_out_of_bounds = start_out_of_bounds+np.random.randint(20,30)


                for i in range(time_series_length):
                    current_time = i
                    

                    # Simulate new data for both groups
                    for motor_index in range(4):
                        if current_time < time_to_zero:

                            if not (start_out_of_bounds <= current_time<= end_out_of_bounds and motor_index == target_motor):
                            # Generate random step within a range and add to last value
                                step = np.random.uniform(-0.1, 0.1)
                                y_group1[motor_index, i] = np.clip(last_values_group1[motor_index] + step,
                                                                *confidence_intervals_group1[mission_number][motor_index])
                                last_values_group1[motor_index] = y_group1[motor_index, i]

                                step = np.random.uniform(-0.1, 0.1)*10
                                y_group2[motor_index, i] = np.clip(last_values_group2[motor_index] + step,
                                                                *confidence_intervals_group2[mission_number][motor_index])
                                last_values_group2[motor_index] = y_group2[motor_index, i]
                        
                        
                        
                            else:
                                if current_time == start_out_of_bounds:
                                    upper_limit = confidence_intervals_group1[mission_number][motor_index][1]
                                    y_group1[motor_index, i] = upper_limit + 0.1*upper_limit  # You can adjust the range as needed
                                    last_values_group1[motor_index] = y_group1[motor_index, i]    
                                else:
                                    step = np.random.uniform(-0.1, 0.1)
                                    y_group1[motor_index, i] = np.clip(last_values_group1[motor_index] + step,
                                                                    upper_limit, upper_limit*1.2)
                                    last_values_group1[motor_index] = y_group1[motor_index, i]           

                                if current_time >= start_out_of_bounds + 10:

                                    motor_flags[motor_titles[target_motor]] = False
                                    for motor, flag in motor_flags.items():
                                        color = "🟢" if flag else "🟠 Potenziale problema sul " + str(motor)
                                        motor_status_placeholders[motor].markdown(f"- {motor}: {color}")
                                
                                
                                ## accelerazione comunque ok
                                step = np.random.uniform(-0.1, 0.1)*10
                                y_group2[motor_index, i] = np.clip(last_values_group2[motor_index] + step,
                                                                *confidence_intervals_group2[mission_number][motor_index])
                                last_values_group2[motor_index] = y_group2[motor_index, i]
                        else:
                            y_group1[motor_index, i]= np.random.uniform(0, 0.1)
                            y_group2[motor_index, i]= np.random.uniform(0, 0.1)*10
                            
                            if current_time < time_to_zero + 30:
                                agv_status = "🔴 Fermo. Possibile problema di collisione"  # Update to the AGV stopped status
                            else:
                                agv_status = "🔴 Fermo. Verificare alimentazione AGV"  # Update to the AGV stopped status
                            agv_status_placeholder.markdown(f"### STATO AGV\n{agv_status}")



                            for motor, placeholder in motor_status_placeholders.items():
                                flag = motor_flags[motor]
                                if motor!= target_motor:
                                    color = "🟢" if flag else "🟠 Potenziale problema sul " + str(motor)
                                else:
                                    color = "🟢" if flag else "🟠 Potenziale problema sul " + str(motor)
                                
                                placeholder.markdown(f'<span style="color: lightgrey;">- {motor}: {color}</span>', unsafe_allow_html=True)




                            



        # Update the plots for group 1
                    create_and_update_plots(axs1.flatten(), motor_titles, y_group1, confidence_intervals_group1[mission_number], 1)
                    plot_placeholder_group1.pyplot(fig1)

                    # Update the plots for group 2
                    create_and_update_plots(axs2.flatten(), motor_titles, y_group2, confidence_intervals_group2[mission_number], 2)
                    plot_placeholder_group2.pyplot(fig2)

                    # Update lightstop status if needed...
                    # Wait for the specified frequency
                    time.sleep(frequency_seconds/3)


            elif mission_type[0] == 3:
                # Loop to update the plots in real-time

                target_motor = np.random.randint(0,4)
                target_motor = np.random.randint(0,4)

                start_out_of_bounds = 10+np.random.randint(10)
                end_out_of_bounds = start_out_of_bounds+np.random.randint(30,45)


                for i in range(time_series_length):
                    current_time = i
                    

                    # Simulate new data for both groups
                    for motor_index in range(4):
                        if current_time < time_to_zero:

                            if not (start_out_of_bounds <= current_time<= end_out_of_bounds and motor_index == target_motor):
                            # Generate random step within a range and add to last value
                                step = np.random.uniform(-0.1, 0.1)
                                y_group1[motor_index, i] = np.clip(last_values_group1[motor_index] + step,
                                                                *confidence_intervals_group1[mission_number][motor_index])
                                last_values_group1[motor_index] = y_group1[motor_index, i]

                                step = np.random.uniform(-0.1, 0.1)*10
                                y_group2[motor_index, i] = np.clip(last_values_group2[motor_index] + step,
                                                                *confidence_intervals_group2[mission_number][motor_index])
                                last_values_group2[motor_index] = y_group2[motor_index, i]
                        
                        
                        
                            else:
                                if current_time == start_out_of_bounds:
                                    upper_limit = confidence_intervals_group1[mission_number][motor_index][1]
                                    y_group1[motor_index, i] = upper_limit + 0.1*upper_limit  # You can adjust the range as needed
                                    last_values_group1[motor_index] = y_group1[motor_index, i]    
                                else:
                                    step = np.random.uniform(-0.1, 0.1)
                                    y_group1[motor_index, i] = np.clip(last_values_group1[motor_index] + step,
                                                                    upper_limit, upper_limit*1.2)
                                    last_values_group1[motor_index] = y_group1[motor_index, i]           

                                if start_out_of_bounds + 10 <= current_time <= start_out_of_bounds + 20:

                                    motor_flags[motor_titles[target_motor]] = False
                                    for motor, flag in motor_flags.items():
                                        color = "🟢" if flag else "🟠 Potenziale problema sul " + str(motor)
                                        motor_status_placeholders[motor].markdown(f"- {motor}: {color}")
                                
                                elif current_time >= start_out_of_bounds + 25:

                                    motor_flags[motor_titles[target_motor]] = False
                                    for motor, flag in motor_flags.items():
                                        color = "🟢" if flag else "🔴 Richiesta attività di ispezione sul " + str(motor)
                                        motor_status_placeholders[motor].markdown(f"- {motor}: {color}")
                            
                                
                                ## accelerazione comunque ok
                                step = np.random.uniform(-0.1, 0.1)*10
                                y_group2[motor_index, i] = np.clip(last_values_group2[motor_index] + step,
                                                                *confidence_intervals_group2[mission_number][motor_index])
                                last_values_group2[motor_index] = y_group2[motor_index, i]
                        else:
                            y_group1[motor_index, i]= np.random.uniform(0, 0.1)
                            y_group2[motor_index, i]= np.random.uniform(0, 0.1)*10
                            
                            if current_time < time_to_zero + 30:
                                agv_status = "🔴 Fermo. Possibile problema di collisione"  # Update to the AGV stopped status
                            else:
                                agv_status = "🔴 Fermo. Verificare alimentazione AGV"  # Update to the AGV stopped status
                            agv_status_placeholder.markdown(f"### STATO AGV\n{agv_status}")

                            for motor, placeholder in motor_status_placeholders.items():
                                flag = motor_flags[motor]
                                if motor!= target_motor:
                                    color = "🟢" if flag else "🔴 Richiesta attività di ispezione sul " + str(motor)
                                else:
                                    color = "🟢" if flag else "🔴 Richiesta attività di ispezione sul " + str(motor)
                                
                                placeholder.markdown(f'<span style="color: lightgrey;">- {motor}: {color}</span>', unsafe_allow_html=True)



        # Update the plots for group 1
                    create_and_update_plots(axs1.flatten(), motor_titles, y_group1, confidence_intervals_group1[mission_number], 1)
                    plot_placeholder_group1.pyplot(fig1)

                    # Update the plots for group 2
                    create_and_update_plots(axs2.flatten(), motor_titles, y_group2, confidence_intervals_group2[mission_number], 2)
                    plot_placeholder_group2.pyplot(fig2)

                    # Update lightstop status if needed...
                    # Wait for the specified frequency
                    time.sleep(frequency_seconds/3)


elif st.session_state['active_tab'] == 'Statistical analysis':

    with tab2:

    # Sidebar to choose a single mission index for Statistical Analysis
        # mission_index_stat = st.sidebar.selectbox(
        #     "Select a mission index for Statistical Analysis:", list(range(0, 9))
        # )

        mission_descriptions = {
            1:"Missione con carico 0 Kg",
            2:"Missione con carico 10 Kg",
            3:"Missione con carico 20 Kg",
        }

        # mission_index_stat = st.sidebar.selectbox("Quali sono le caratteristiche di questa missione?:", mission_descriptions)

        mission_index_stat = st.sidebar.selectbox("Quali sono le caratteristiche di questa missione?:", mission_descriptions.keys(), format_func=lambda x: mission_descriptions[x])



    # Sidebar to choose confidence level
        confidence_level = st.sidebar.slider(
            "Seleziona la confidenza richiesta:", min_value=75, max_value=99, value=95, step=1)



        def preprocess_data(data_path):
            import pandas as pd
            from datetime import datetime
            import numpy as np

            # Function to convert comma-separated floats to period-separated floats
            def convert_comma_separated_float(value):
                return float(value.replace(',', '.'))

            # Read the csv file
            data = pd.read_csv(data_path, sep=';', parse_dates=['Timestamp'], converters={
                            'Value': convert_comma_separated_float})

            # Extract the date from the Timestamp column
            date = data['Timestamp'].iloc[0].date()

            # Define mission phases
            missions = [
                ('T01', '10:35', '10:45'),
                ('T02', '10:45', '11:55'),
                ('T03', '11:58', '12:10'),
                ('T04', '12:10', '13:10'),
                ('T05', '13:15', '13:25'),
                ('T06', '13:25', '14:25'),
                ('T07', '15:00', '15:30'),
                ('T08', '15:35', '16:05'),
                ('T09', '16:15', '17:05'),
                ('T10', '17:06', '17:20'),
            ]

            missions = [
                ('T02', '10:45', '11:55'),
                ('T04', '12:10', '13:10'),
                ('T06', '13:25', '14:25'),
                ('T07', '15:00', '15:30')

            ]


            # Convert mission times to Timestamp objects with the extracted date
            missions = [(name, pd.Timestamp(f"{date} {start_time}"), pd.Timestamp(
                f"{date} {end_time}")) for name, start_time, end_time in missions]

            # Define mission types
            missions_fermi = set(["T01", "T03", "T05"])
            missions_movement = set(["T02", "T04", "T06", "T07", "T08", "T09", "T10"])

            # Define motor groups
            motor_group1 = [182, 185, 188, 191]
            motor_group2 = [183, 186, 189, 192]

            # Round the timestamps down to the nearest second
            data['Timestamp'] = data['Timestamp'].dt.floor('S')

            # Group by timestamp and motor identifier, calculate the mean value
            grouped_data = data.groupby(
                ['Timestamp', 'LayoutObject_Ident']).mean().reset_index()

            # Reshape the data so each row corresponds to a unique second and each column corresponds to a motor
            reshaped_data = grouped_data.pivot(
                index='Timestamp', columns='LayoutObject_Ident', values='Value').reset_index()

            # Create a column for the mission label, initially set to None
            reshaped_data['Label'] = None

            # Label the data based on mission type
            for mission in missions:
                mission_name, start_time, end_time = mission
                mask = (reshaped_data['Timestamp'] >= start_time) & (
                    reshaped_data['Timestamp'] <= end_time)
                if mission_name in missions_fermi:
                    reshaped_data.loc[mask, 'Label'] = 0  # idle
                elif mission_name in missions_movement:
                    reshaped_data.loc[mask, 'Label'] = 1  # movement

            # Remove rows where Label is still None
            reshaped_data = reshaped_data.dropna(subset=['Label'])

            # Apply rolling mean with a window of 5 seconds
            reshaped_data.set_index('Timestamp', inplace=True)
            reshaped_data[motor_group1 + motor_group2] = reshaped_data[motor_group1 +
                                                                    motor_group2].rolling('5s').mean()

            reshaped_data = reshaped_data.dropna()
            reshaped_data.reset_index(inplace=True)

           

            return data, missions, motor_group1, motor_group2
        # Function for LightGBM predictions (Replace with your actual prediction code)


        def compute_confidence_interval(mission_data, motor_group, group_number, z_score, confidence_level):
            local_confidence_intervals = {}
            for motor in motor_group:
                motor_data = mission_data[mission_data['LayoutObject_Ident'] == motor]
                mean_value = motor_data['Value'].mean()
                std_dev_value = motor_data['Value'].std()

                # Use the Z-score in the confidence interval calculation
                lower_bound = mean_value - z_score * std_dev_value
                upper_bound = mean_value + z_score * std_dev_value

                local_confidence_intervals[motor] = (lower_bound, upper_bound)
            return local_confidence_intervals, confidence_level


        def plot_motor_group_2x2_subplots(mission_name, start_time, end_time, motor_group, group_number, confidence_intervals, confidence_level):
            unit = ["mm/s", "mg"]

            mask = (data['Timestamp'] >= start_time +
                    timedelta(seconds=45)) & (data['Timestamp'] <= end_time)
            mission_data = data.loc[mask]

            group_data = mission_data[mission_data['LayoutObject_Ident'].isin(
                motor_group)]
            global_min = group_data['Value'].min()
            global_max = group_data['Value'].max()

            colors = ['red', 'orange', 'green', 'blue']
            fig, axes = plt.subplots(
                nrows=2, ncols=2, figsize=(24, 18), facecolor='white')
            fig.suptitle(f"Group {group_number} - dati relativi alle {'velocità' if group_number==1 else 'accelerazioni'}", fontsize=26)

            metrics = []
            for index, motor in enumerate(motor_group):
                row = index // 2
                col = index % 2
                ax = axes[row, col]
                motor_data = mission_data[mission_data['LayoutObject_Ident'] == motor]

                ax.plot(motor_data['Timestamp'], motor_data['Value'],
                        label=f'Motor {motor}', color=colors[index])

                lower_bound, upper_bound = confidence_intervals[motor]
                ax.fill_between(motor_data['Timestamp'], lower_bound,
                                upper_bound, color='gray', alpha=0.2)

                longest_sequence_outside = find_longest_sequence_outside(
                    motor_data, lower_bound, upper_bound)
                print(f"Longest Sequence Outside: {longest_sequence_outside}")

                # if longest_sequence_outside > sensitivity:
                #     st.warning(
                #         f"WARNING: Motor {motor} {'acceleration' if group_number == 2 else 'speed'} was out of the expected range for more than {sensitivity} seconds. You should check it out.")

                motor_to_int_dic = {182: 1, 185:2, 188:3, 191:4, 183:1, 186:2, 189:3, 192:4}

                outside_mask = (motor_data['Value'] < lower_bound) | (
                    motor_data['Value'] > upper_bound)
                outside_timestamps = motor_data[outside_mask]['Timestamp']

                outside_seconds = len(outside_timestamps.dt.round('S').unique())
                print(
                    f"Total timestamps outside confidence interval: {outside_seconds}")

                ax.set_title(f'Motor {motor_to_int_dic[motor]}', fontsize=20)
                ax.set_xlabel('Timestamp', fontsize=16)
                ax.set_ylabel(unit[group_number-1], fontsize=16)
                ax.tick_params(axis='x', labelsize=14, rotation=45)
                ax.tick_params(axis='y', labelsize=14)

                ax.set_ylim(global_min, global_max)
                metrics.append({
                    "Motor": motor_to_int_dic[motor],
                    "LB": lower_bound,
                    "UB": upper_bound,
                    "Confidence": confidence_level,
                    "Seconds Outside Bands": outside_seconds,
                    "Longest Sequence Outside": longest_sequence_outside,
                    "Unity of measure": "mg" if group_number == 2 else "mm/s"
                })

            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            return metrics, fig


        def find_longest_sequence_outside(data, lower_bound, upper_bound, tolerance=10):
            outside_mask = (data['Value'] < lower_bound) | (
                data['Value'] > upper_bound)
            outside_data = data[outside_mask]

            # Convert datetime to UNIX timestamp (seconds since 1970-01-01)
            rounded_timestamps = (
                outside_data['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

            sorted_timestamps = np.sort(rounded_timestamps.unique())
            diffs = np.diff(sorted_timestamps)

            sequences = []
            current_sequence = [sorted_timestamps[0]] if len(
                sorted_timestamps) > 0 else []

            for i in range(1, len(sorted_timestamps)):
                if diffs[i-1] <= tolerance:
                    current_sequence.append(sorted_timestamps[i])
                else:
                    sequences.append(current_sequence)
                    current_sequence = [sorted_timestamps[i]]
            if current_sequence:
                sequences.append(current_sequence)

            longest_sequence = max((len(s) for s in sequences), default=0)
            return longest_sequence


        def print_metrics(metrics):
            """
            Print the metrics in a structured format.
            """
            motor_to_int_dic = {182: 1, 185:2, 188:3, 191:4, 183:1, 186:2, 189:3, 192:4, "182": "1", "185":"2", "188":"3", "191":"4", "183":"1", "186":"2", "189":"3", "192":"4"}

            for metric in metrics:
                print("----------------------")
                print(f"Motor: {motor_to_int_dic[metric['Motor']]}")
                print(f"LB: {round(metric['LB'], 4)}")
                print(f"UB: {round(metric['UB'], 4)}")
                print(f"Confidence: {metric['Confidence']}")
                print(f"Seconds Outside Bands: {metric['Seconds Outside Bands']}")
                print(f"Unity of Measure: {metric['Unity of measure']}")
            print("----------------------")


        # Streamlit UI
        st.title("ARPA: analysis of AGV movements")



        import scipy.stats as stats

    
        # Convert the confidence level to Z-score
        confidence_level = confidence_level / 100.0  # Convert to decimal
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        data_path = "data_AGV2.csv"

        # Data Preprocessing
        data, missions, motor_group1, motor_group2 = preprocess_data(
            data_path)

        mission_name, start_time, end_time = missions[mission_index_stat-1]
        # Display some sample data (Optional)

        start_time += timedelta(seconds=15)

        # Extracting data for the current mission
        mask = (data['Timestamp'] >= start_time) & (
            data['Timestamp'] <= end_time)
        

        mission_data = data[mask]

        mission_print_data = mission_data.copy()
        mission_print_data['Timestamp'] = pd.to_datetime(mission_print_data['Timestamp'])
        dic_convert =  {182: 1, 185:2, 188:3, 191:4, 183:1, 186:2, 189:3, 192:4, "182": "1", "185":"2", "188":"3", "191":"4", "183":"1", "186":"2", "189":"3", "192":"4"}
        mission_print_data = mission_print_data.replace({"LayoutObject_Ident": dic_convert})                    


        # Now, format the 'Timestamp' column to only show hh:mm:ss
        mission_print_data['Timestamp'] = mission_print_data['Timestamp'].dt.strftime('%d-%m %H:%M:%S')

        

        # Computing confidence intervals for the current mission
        confidence_intervals_group1, confidence_level = compute_confidence_interval(
            mission_data, motor_group1, 1, z_score, confidence_level)
        confidence_intervals_group2, confidence_level = compute_confidence_interval(
            mission_data, motor_group2, 2, z_score, confidence_level)
        confidence_intervals = {
            **confidence_intervals_group1, **confidence_intervals_group2}

        # Initialize a dictionary to store confidence intervals for each motor

        # Plotting motor group time series in 2x2 subplots and calculating metrics for the first mission
        metrics_group1_2x2, fig1 = plot_motor_group_2x2_subplots(
            mission_name, start_time, end_time, motor_group1, 1, confidence_intervals, confidence_level)
        st.pyplot(fig1)
        st.write("Metrics for Motor Group 1:")
        st.write(metrics_group1_2x2)

        metrics_group2_2x2, fig2 = plot_motor_group_2x2_subplots(
            mission_name, start_time, end_time, motor_group2, 2, confidence_intervals, confidence_level)
        st.pyplot(fig2)
        st.write("Metrics for Motor Group 2:")
        st.write(metrics_group2_2x2)


        st.write("Sample Test Data")

        st.write(mission_print_data.head())





        


else:

    with tab3:

  

        # Create a mapping of mission names to indices
        missions = {
            "1: fermo ADDESTRAMENTO": 0,
            "2: movimento ADDESTRAMENTO": 1,
            "3: fermo": 2,
            "4: movimento": 3,
            "5: fermo": 4,
            "6: movimento": 5,
            # "7: movimento": 6,
            # "8: movimento": 7,
            # "9: movimento": 8,
            # "10: movimento 2x10": 9
        }

        # Display the multiselect widget with mission names
        selected_missions = st.sidebar.multiselect(
            "Select mission indices for Machine Learning:",
            options=list(missions.keys()),
            default=["1: fermo ADDESTRAMENTO", "2: movimento ADDESTRAMENTO"]
        )

        # Convert selected mission names back to their indices
        mission_indices_ml = [missions[mission] for mission in selected_missions]


      


        def preprocess_data(data_path):
            import pandas as pd
            from datetime import datetime
            import numpy as np

            # Function to convert comma-separated floats to period-separated floats
            def convert_comma_separated_float(value):
                return float(value.replace(',', '.'))

            # Read the csv file
            data = pd.read_csv(data_path, sep=';', parse_dates=['Timestamp'], converters={
                            'Value': convert_comma_separated_float})

            # Extract the date from the Timestamp column
            date = data['Timestamp'].iloc[0].date()

            # Define mission phases
            missions = [
                ('T01', '10:35', '10:45'),
                ('T02', '10:45', '11:55'),
                ('T03', '11:58', '12:10'),
                ('T04', '12:10', '13:10'),
                ('T05', '13:15', '13:25'),
                ('T06', '13:25', '14:25'),
                ('T07', '15:00', '15:30'),
                ('T08', '15:35', '16:05'),
                ('T09', '16:15', '17:05'),
                ('T10', '17:06', '17:20'),
            ]

            # Convert mission times to Timestamp objects with the extracted date
            missions = [(name, pd.Timestamp(f"{date} {start_time}"), pd.Timestamp(
                f"{date} {end_time}")) for name, start_time, end_time in missions]

            # Define mission types
            missions_fermi = set(["T01", "T03", "T05"])
            missions_movement = set(["T02", "T04", "T06", "T07", "T08", "T09", "T10"])

            # Define motor groups
            motor_group1 = [182, 185, 188, 191]
            motor_group2 = [183, 186, 189, 192]

            # Round the timestamps down to the nearest second
            data['Timestamp'] = data['Timestamp'].dt.floor('S')

            # Group by timestamp and motor identifier, calculate the mean value
            grouped_data = data.groupby(
                ['Timestamp', 'LayoutObject_Ident']).mean().reset_index()

            # Reshape the data so each row corresponds to a unique second and each column corresponds to a motor
            reshaped_data = grouped_data.pivot(
                index='Timestamp', columns='LayoutObject_Ident', values='Value').reset_index()

            # Create a column for the mission label, initially set to None
            reshaped_data['Label'] = None

            # Label the data based on mission type
            for mission in missions:
                mission_name, start_time, end_time = mission
                mask = (reshaped_data['Timestamp'] >= start_time) & (
                    reshaped_data['Timestamp'] <= end_time)
                if mission_name in missions_fermi:
                    reshaped_data.loc[mask, 'Label'] = 0  # idle
                elif mission_name in missions_movement:
                    reshaped_data.loc[mask, 'Label'] = 1  # movement

            # Remove rows where Label is still None
            reshaped_data = reshaped_data.dropna(subset=['Label'])

            # Apply rolling mean with a window of 5 seconds
            reshaped_data.set_index('Timestamp', inplace=True)
            reshaped_data[motor_group1 + motor_group2] = reshaped_data[motor_group1 +
                                                                    motor_group2].rolling('5s').mean()

            reshaped_data = reshaped_data.dropna()
            reshaped_data.reset_index(inplace=True)

            # Generate list of start and end times for selected missions
            test_start_times = [missions[i][1] for i in mission_indices_ml]
            test_end_times = [missions[i][2] for i in mission_indices_ml]

            # Create mask to filter test set based on multiple missions
            test_mask = False
            for start, end in zip(test_start_times, test_end_times):
                test_mask |= (reshaped_data['Timestamp'] >= start) & (
                    reshaped_data['Timestamp'] <= end)

            # Create test set based on mask
            test_data = reshaped_data[test_mask]

            # Define columns to drop
            X_columns_to_drop = ['Timestamp', 'Label']

            # Create X and y sets for training, test, and extra

            X_test, y_test = test_data.drop(
                columns=X_columns_to_drop), test_data['Label'].astype('int8')

            return X_test, y_test, data, missions, motor_group1, motor_group2
        # Function for LightGBM predictions (Replace with your actual prediction code)


        def make_lgbm_predictions(model_path, X_test, y_test):
            import pickle
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

            # Load the trained LightGBM model
            with open(model_path, 'rb') as f:
                lgbm_model = pickle.load(f)

            # Make predictions for test and extra datasets using LightGBM
            y_pred_test_lgbm = lgbm_model.predict(X_test)

            # Convert predictions to binary class labels
            y_pred_test_lgbm = np.round(y_pred_test_lgbm).astype(int)

            # Compute evaluation metrics for LightGBM
            metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred_test_lgbm),
                'test_f1': f1_score(y_test, y_pred_test_lgbm),
                'test_confusion_matrix': confusion_matrix(y_test, y_pred_test_lgbm),
            }

            return y_pred_test_lgbm, metrics

        # Redefining the compute_confidence_interval function to return the computed confidence intervals



        # Streamlit UI
        st.title("ARPA: analysis of AGV movements")


        data_path = "data_AGV2.csv"
        # Data Preprocessing
        X_test, y_test, data, missions, motor_group1, motor_group2 = preprocess_data(
            data_path)

        


        st.markdown("## Legenda:")
        st.markdown("- ### 0: AGV in stato di Fermo")
        st.markdown("- ### 1: AGV in movimento")
        

        st.markdown("# LightGBM:")

        # File upload widget for LightGBM model
        uploaded_model_file = st.file_uploader(
            "Choose a LightGBM model file", type=["pkl"])

        if uploaded_model_file is not None:
            # Assuming you've saved the uploaded model to a temporary location
            if uploaded_model_file is not None:
                # Save the uploaded model to a temporary location
                model_path = "temp_model.pkl"
                with open(model_path, "wb") as f:
                    f.write(uploaded_model_file.getbuffer())

            # Make Predictions
            y_pred_test_lgbm, metrics = make_lgbm_predictions(
                model_path, X_test, y_test)



            # Display Metrics and Predictions
            st.subheader("LightGBM Evaluation Metrics")

            # Display metrics for test data
            st.write(f"Test Accuracy: {metrics['test_accuracy']}")
            st.write(f"Test F1 Score: {metrics['test_f1']}")
            st.write("Test Confusion Matrix:")
            st.write(metrics['test_confusion_matrix'])

            # Import matplotlib for plotting

            # Plotting real vs. predicted target variable for test set
            st.subheader("Test Set - LightGBM")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(X_test.index, y_test, label='Real')
            ax.scatter(X_test.index, y_pred_test_lgbm,
                    color='r', label='Predicted')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Target Variable')
            plt.legend()
            st.pyplot(fig)


        st.markdown("# Neural Network:")

        # Streamlit UI
        uploaded_scaler_file = st.file_uploader(
            "Choose the scaler file", type=["pkl"])
        uploaded_model_file = st.file_uploader(
            "Choose the model file", type=["pth"])

        if uploaded_scaler_file is not None and uploaded_model_file is not None:
            # Load scaler
            loaded_scaler = pickle.load(uploaded_scaler_file)

            # NeuralNet definition (replace sizes with your actual sizes)
            class NeuralNet(nn.Module):
                def __init__(self, input_size, hidden_size, num_classes):
                    super(NeuralNet, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_size, num_classes)

                def forward(self, x):
                    out = self.fc1(x)
                    out = self.relu(out)
                    out = self.fc2(out)
                    return out

            # Initialize model
            input_size = 8  # replace with your input size
            hidden_size = 64
            num_classes = 1
            model = NeuralNet(input_size, hidden_size, num_classes)

            # Load model
            model.load_state_dict(torch.load(uploaded_model_file))

            # Preprocess your data (replace 'your_data_path' with your actual data path)

            # Scaling
            X_test_scaled = loaded_scaler.transform(X_test)

            # Convert to tensor
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

            # Model inference
            model.eval()
            with torch.inference_mode():
                test_logits = model(X_test_tensor)
                test_pred = torch.round(torch.sigmoid(test_logits))

            # Evaluation
            y_test_np = y_test_tensor.cpu().numpy()
            y_pred_test_np = test_pred.detach().cpu().numpy()
            accuracy_test = accuracy_score(y_test_np, y_pred_test_np)
            f1_test = f1_score(y_test_np, y_pred_test_np)

            # Output metrics and plots
            st.write(f"Accuracy: {accuracy_test}")
            st.write(f"F1 Score: {f1_test}")

            # Plotting
            timestamp_test = X_test.index
            # Create a figure and axes object
            fig, ax = plt.subplots(figsize=(14, 6))
            # Use axes object for plotting
            ax.plot(timestamp_test, y_test_np, label='Real')
            ax.scatter(timestamp_test, y_pred_test_np,
                    color='r', label='Predicted')
            ax.set_xlabel('Timestamp')  # Set x-axis label using axes object
            ax.set_ylabel('Target Variable')  # Set y-axis label using axes object
            ax.set_title('Test Set - NeuralNet')  # Set title using axes object
            ax.legend()  # Show legend using axes object
            st.pyplot(fig)  # Pass the figure object to Streamlit's pyplot function
        
        # Display some sample data (Optional)
        st.write("Sample Test Data")
        st.write(X_test.head())


st.image("immagine_logo_EU_PON.png")