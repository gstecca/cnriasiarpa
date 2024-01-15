import os
import movingpandas as mpd
mpd.show_versions()
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import movingpandas as mpd

def load_data(filepath, filename):
    file = filepath + filename
    data = pd.read_csv(file)
    return data

def plot_routes(df):
    hvplot_defaults = {'tiles':'CartoLight','frame_height':400, 'frame_width': 700, 'cmap': 'Viridis','colorbar': True}
    traj_collection = mpd.TrajectoryCollection(df, 'node_sn', t='timestamp', x='GPS_lon', y='GPS_lat')
    traj_collection.hvplot(title=str(traj_collection), line_width=5, **hvplot_defaults)

class bbox:
    def __init__(self, location_name, north, west, south, east):
        self.location_name = location_name
        self.north = north
        self.west = west
        self.south = south
        self.east = east

    def is_in_bbox(self, lat, long):
        if ((lat > self.south) and (lat < self.north) and (long > self.west) and (long < self.east)):
                return True
        else:
            return False

# loc is expected to be a tuple (y,x) i.e. (lat,long)
def check_in_bbox_list(bbox_list, location_tuple):
    location = None
    for bbox in bbox_list:
        if bbox.is_in_bbox(location_tuple[0], location_tuple[1]) == True:
            location = bbox.location_name
    return location


def get_truck_KPIs(df, truck_ID):

    df_k = df[df['node_sn'] == truck_ID]
    df_gerbole = df_k[(df_k['GPS_lat'] > bbox_Gerbole.south) & (df_k['GPS_lat'] < bbox_Gerbole.north) & (df_k['GPS_lon'] > bbox_Gerbole.west) & (df_k['GPS_lon'] < bbox_Gerbole.east)]
    df_gerbole.sort_values(['date', 'time'], ascending=[True, True], inplace=True)

    return 100

def inout_from_bbox(df, bbox):
    #############
    recording_days = df['date'].unique().tolist()
    #############
    df_in_bbox = df[(df['GPS_lat'] > bbox.south) & (df['GPS_lat'] < bbox.north) &
                    (df['GPS_lon'] > bbox.west) & (df['GPS_lon'] < bbox.east)]
    df_in_bbox.sort_values(['date', 'time'], ascending=[True, True], inplace=True)
    days_in_bbox = df_in_bbox['date'].unique().tolist()
    #############
    days_out_of_bbox = list(set(recording_days) - set(days_in_bbox))
    days_out_of_bbox.sort()
    #############
    leaving_datatimes  = []
    arriving_datatimes = []
    for day in days_in_bbox:
        if days_in_bbox.index(day) == len(days_in_bbox) - 1: #i.e. if the day is the last element of the list, then it MUST be the last ore the only leaving DAY
            # --> the thesis here is that if this is the case, then there would not be any return to this bbox
            leaving_day = day
            leaving_time = df_in_bbox[df_in_bbox['date'] == leaving_day]['time'].values[-1]
            leaving_datatimes.append([leaving_day,leaving_time])
            break
        following_day = days_in_bbox[days_in_bbox.index(day) + 1] # considero il giorno in bbox e quello successivo in bbbox
        delta = following_day - day
        if delta.days > 1: #i.e. these two days are non-consecutive in time
            if recording_days.index(following_day) != recording_days.index(day) + 1: #i.e. these two days are non-consecutive in recording days list
                leaving_day = day
                leaving_time = df_in_bbox[df_in_bbox['date'] == leaving_day]['time'].values[-1]
                leaving_datatimes.append([leaving_day, leaving_time])
                #############
                arriving_day = following_day
                arriving_time = df_in_bbox[df_in_bbox['date'] == arriving_day]['time'].values[1]
                arriving_datatimes.append([arriving_day, arriving_time])
                break

    return {'leaving datetimes': leaving_datatimes, 'arriving_datatimes': arriving_datatimes}


if __name__ == '__main__':

    cwd = os.getcwd()
    data_path = cwd + '/Data/'
    file_name_1 = 'cyrene_logistics_incremental 22-03-2023.csv'
    file_name_2 = 'cyrene_logistics_incremental 05-05-2023.csv'

    df1 = load_data(data_path, file_name_1)
    df2 = load_data(data_path, file_name_2)
    df = pd.concat([df1,df2])
    print('data is loaded')


    bbox_Gerbole = bbox('Gerbole', 45.00608379933475, 7.494028665630488, 44.9789664976549, 7.541781329617419)
    bbox_Samsung = bbox('Samsung', 47.45, 19.12, 47.40, 19.19)
    bbox_Mirafiori = bbox('Mirafiori',45.03783123341941, 7.61185459392937,   45.01659162577987, 7.637021253139538)

    bbox_logistic_graph = bbox('logistic_network',48.313817888801935, 6.408918345461935, 44.21152525718785, 19.296717017916514)
    # G = get_network_graph(bbox_logistic_graph)
    #################################################################################
    # cleaning dataset
    df = df.dropna(axis=1, how='all')
    df = df[df['GPS_lat'] > 0.1] # clening dataset GPS no-values (i.e GPS = 0.0000)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['time'] = pd.to_datetime(df['timestamp']).dt.time
    #################################################################################

    #################################################################################
    # Plotting
    # plot_routes(df)
    #################################################################################

    truck_list = df['node_sn'].unique().tolist()
    truck_list.sort()
    for k in truck_list[:1]:
        df_k = df[df['node_sn'] == k]
        df_k.sort_values(['date','time'], ascending=[True, True], inplace = True)

        Gerbole_inout_datetimes = inout_from_bbox(df_k, bbox_Gerbole)

        lat = 45.0
        long = 7.52
        temp = bbox_Gerbole.is_in_bbox(lat,long)

        main_bbox_list = [bbox_Gerbole, bbox_Samsung, bbox_Mirafiori]
        location_is_in = check_in_bbox_list(main_bbox_list, (lat,long))

        print(location_is_in)






    # dato un camion k ho ordinato il percorso dall'inizio alla fine del tracking GPS.
    # @TODO:
    # 1) individurare km percorsi nel chekpoint posizione-tempo ogni 4 ore dalla partenza.




    # TruckTurning: This KPI measures the truck’s turntable, that is, the average time spent between the exit for
    # collect/delivery and the return of the vehicle to the company.



