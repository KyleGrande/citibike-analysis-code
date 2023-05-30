"""
Name:       Kyle Grande
Email:      kyle.grande72@myhunter.cuny.edu
Resources:  Used the following resources to help with this project:
            pandas documentation: https://pandas.pydata.org/docs/
            matplotlib documentation: https://matplotlib.org/stable/contents.html
            seaborn documentation: https://seaborn.pydata.org/tutorial.html
            numpy documentation: https://numpy.org/doc/stable/
            folium documentation: https://python-visualization.github.io/folium/
            prophet documentation: https://facebook.github.io/prophet/docs/quick_start.html
            https://facebook.github.io/prophet/docs/quick_start.html
            https://facebook.github.io/prophet/docs/diagnostics.html
            https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
            https://facebook.github.io/prophet/docs/non-daily_data.html
            https://facebook.github.io/prophet/docs/uncertainty_intervals.html
            https://facebook.github.io/prophet/docs/multiplicative_seasonality.html

Title:      Exploring Trends in the Growing NYC Citi-Bike Sharing Program
URL:        https://www.kylegrande.github.io/citi-bike-sharing-analysis
"""
import pandas as pd
from pandas.errors import ParserError
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import folium
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

def read_data(csv_files):
    """
    Reads bike sharing data from CSV files, processes the data, and returns a DataFrame.

    Args:
    csv_files (list): A list of CSV files containing bike sharing data.

    Returns:
    DataFrame: A DataFrame containing processed bike sharing data.
    """
    # Read data from CSV files from folder Data
    dfs = []
    dtypes = {'start_station_id': 'object', 'end_station_id': 'object'}

    for file in csv_files:
        try:
            for chunk in pd.read_csv(file,
                                     dtype=dtypes,
                                     chunksize=1000000,
                                     on_bad_lines='skip'):
                dfs.append(chunk)
        except ParserError as error:
            print(f"Error parsing file {file}: {error}")
            continue

    data = pd.concat(dfs, ignore_index=True)
    return data

def process_data(data):
    '''
    Process the data to add new columns and convert columns to the correct data types.

    Args:
    data (DataFrame): A DataFrame containing bike sharing data.

    Returns:
    DataFrame: A DataFrame containing processed bike sharing data.
    '''
    # Convert the 'started_at' and 'ended_at' columns to datetime objects
    data['started_at'] = pd.to_datetime(data['started_at'],
                                        errors='coerce',
                                        infer_datetime_format=True)
    data['ended_at'] = pd.to_datetime(data['ended_at'],
                                      errors='coerce',
                                      infer_datetime_format=True)

    # Drop rows with NaT values
    data.dropna(subset=['started_at', 'ended_at'], inplace=True)
    # data['started_at'] = pd.to_datetime(data['started_at'],
    #                                     errors='coerce',
    #                                     infer_datetime_format=True)
    # data['ended_at'] = pd.to_datetime(data['ended_at'],
    #                                   errors='coerce',
    #                                   infer_datetime_format=True)

    # Extract features from the datetime columns
    data['date'] = data['started_at'].dt.date
    # data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
    # data['date'] = data['date'].dt.date
    data['duration'] = (data['ended_at'] - data['started_at']).dt.total_seconds() / 60
    data['day_of_week'] = data['started_at'].dt.dayofweek
    data['hour_of_day'] = data['started_at'].dt.hour
    data['is_weekend'] = data['day_of_week'].isin([5, 6])

    return data

def plot_duration_histogram(data):
    '''
    Plots a histogram of the bike trip durations.

    Args:
    data (DataFrame): A DataFrame containing bike sharing data.

    Returns:
    None
    '''

    plt.figure(figsize=(10, 5))
    # Define bins to include every minute up to 60 minutes
    bins = np.arange(0, 61, 1)
    # Count the number of trips in each bin
    value_counts = data['duration'].value_counts(bins=bins, sort=False)
    # Plot the counts for each bin
    plt.bar(bins[:-1], value_counts, width=1, edgecolor='black')
    plt.title('Bike Trip Duration Histogram')
    plt.xlabel('Trip Duration (minutes)')
    plt.ylabel('Number of Trips')
    plt.savefig('duration_histogram.png', bbox_inches='tight')
    plt.show()

def plot_trips_by_user_type(data):
    '''
    Plots a bar chart of the number of bike trips by user type.

    Args:
    data (DataFrame): A DataFrame containing bike sharing data.

    Returns:
    None
    '''

    # Count the number of trips for each user type
    user_type_counts = data['member_casual'].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    # Plot the counts for each user type
    sns.barplot(x=user_type_counts.index, y=user_type_counts.values, palette='viridis')
    plt.title('Bike Trips by User Type')
    plt.xlabel('User Type')
    plt.ylabel('Number of Trips')
    # ensures y-axis starts at 0
    plt.ylim(bottom=0)

    # Format y-axis tick labels to display with commas
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    # Save the figure as a PNG image
    plt.savefig('trips_by_user_type.png', bbox_inches='tight')
    plt.show()

def plot_median_ride_duration_by_user_type(data):
    '''
    Plots a bar chart of the median ride duration by user type.

    Args:
    data (DataFrame): A DataFrame containing bike sharing data.

    Returns:
    None
    '''

    # Calculate the median ride duration for each user type
    duration_data = data.groupby('member_casual')['duration'].median().sort_index()
    # Plot the median ride duration for each user type
    sns.barplot(x=duration_data.index, y=duration_data.values, palette='viridis')
    plt.title('Median Ride Duration by Rider Type')
    plt.xlabel('Rider Type')
    plt.ylabel('Duration (minutes)')
    plt.savefig('median_ride_duration_by_user_type.png', bbox_inches='tight')
    plt.show()

def plot_daily_trips_over_time(data):
    '''
    Plots a line chart of the number of bike trips over time.

    Args:
    data (DataFrame): A DataFrame containing bike sharing data.

    Returns:
    None
    '''
    # Extract the date from the 'started_at' column
    data['date'] = data['started_at'].dt.date
    # Count the number of trips per day and sort by date
    daily_trips = data['date'].value_counts().sort_index()

    plt.figure(figsize=(15, 5))
    # Plot the number of trips per day
    plt.plot(daily_trips.index, daily_trips.values)
    plt.title('Bike Trips Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Trips')
    # ensures y-axis starts at 0
    plt.ylim(bottom=0)

    # Format y-axis tick labels to display with commas
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.savefig('daily_trips_over_time.png', bbox_inches='tight')
    plt.show()
def plot_trips_by_day_of_week(data):
    '''
    Plots the number of bike trips by day of the week.

    Args:
    data (DataFrame): A DataFrame containing bike trip data.

    Returns:
    None
    '''
    # Sort the index to ensure days are plotted in order
    day_of_week_counts = data['day_of_week'].value_counts().sort_index()
    # Define the day of week names for the x-axis labels
    day_of_week_names = ['Monday',
                         'Tuesday',
                         'Wednesday',
                         'Thursday',
                         'Friday',
                         'Saturday',
                         'Sunday']

    plt.figure(figsize=(8, 4))
    # Create the bar plot
    sns.barplot(x=day_of_week_counts.index, y=day_of_week_counts.values)
    plt.title('Bike Trips by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Trips')
    plt.xticks(day_of_week_counts.index, day_of_week_names)
     # ensures y-axis starts at 0
    plt.ylim(bottom=0)

    # Format y-axis tick labels to display with commas
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.savefig('trips_by_day_of_week.png', bbox_inches='tight')  #
    plt.show()

def plot_trips_by_hour_of_day(data):
    '''
    Plots the number of bike trips by hour of the day.

    Args:
    data (DataFrame): A DataFrame containing bike trip data.

    Returns:
    None
    '''

    # Sort the index to ensure hours are plotted in order
    hour_of_day_counts = data['hour_of_day'].value_counts().sort_index()
    # Set the figure size
    plt.figure(figsize=(10, 4))
    # Create the bar plot
    sns.barplot(x=hour_of_day_counts.index, y=hour_of_day_counts.values)
    plt.title('Bike Trips by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Trips')
    # ensures y-axis starts at 0
    plt.ylim(bottom=0)

    # Format y-axis tick labels to display with commas
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.savefig('trips_by_hour_of_day.png', bbox_inches='tight')
    plt.show()

def plot_pop_routes_map_all(data, num_routes=10, save_file='pop_routes_map_all.html'):
    """
    Creates a Folium map of the most pop bike routes in the data, including all pop routes.
    Saves the map as an HTML file.

    Args:
    data (DataFrame): A DataFrame containing bike trip data with start and end station
    latitude and longitude coordinates.
    num_routes (int): The number of most pop routes to display on the map.
    save_file (str): The filename to save the HTML map as.
    """
    # Aggregate bike trip data by start and end
    # station pairs and count the number of trips between each pair
    routes = data.groupby(['start_station_id',
                           'end_station_id']).size().reset_index(name='trip_count')

    # Sort routes by trip count in descending order
    pop_routes_all = routes.sort_values(by='trip_count', ascending=False)

    # Select the top num_routes routes for the map that includes all pop routes
    pop_routes_all = pop_routes_all.head(num_routes)

    # Generate a color palette with the number of colors equal to the number of routes
    colors = sns.color_palette("husl", num_routes).as_hex()

    # Create the first map (includes all pop routes)
    map_all = folium.Map(location=[data['start_lat'].mean(),
                                   data['start_lng'].mean()],
                                   zoom_start=12)
    for idx, (_, row) in enumerate(pop_routes_all.iterrows()):
        start_lat, start_lng = data.loc[data['start_station_id'] == row['start_station_id'],
                                        ['start_lat', 'start_lng']].iloc[0]
        end_lat, end_lng = data.loc[data['end_station_id'] == row['end_station_id'],
                                    ['end_lat', 'end_lng']].iloc[0]
        line_color = colors[idx]
        folium.PolyLine(locations=[[start_lat, start_lng], [end_lat, end_lng]],
                        color=line_color,
                        weight=20,
                        tooltip=f'Trip count: {row["trip_count"]}').add_to(map_all)

    map_all.save(save_file)

def plot_pop_routes_map_exclude_same(data,
                                         num_routes=10,
                                         save_file='pop_routes_map_exclude_same.html'):
    """
    Creates a Folium map of the most pop bike routes in the data,
    excluding routes where the start and end stations are the same.
    Saves the map as an HTML file.

    Args:
    data (DataFrame): A DataFrame containing bike trip data with start and end station
    latitude and longitude coordinates.
    num_routes (int): The number of most pop routes to display on the map.
    save_file (str): The filename to save the HTML map as.
    """
    # Aggregate bike trip data by start and end
    # station pairs and count the number of trips between each pair
    routes = data.groupby(['start_station_id',
                           'end_station_id']).size().reset_index(name='trip_count')

    # Sort routes by trip count in descending order
    pop_routes_all = routes.sort_values(by='trip_count', ascending=False)

    # Filter out routes where the start and end
    # stations are the same and select the top num_routes routes
    pop_routes_exclude = pop_routes_all[pop_routes_all['start_station_id'] !=
                                        pop_routes_all['end_station_id']].head(num_routes)

    # Generate a color palette with the number of colors equal to the number of routes
    colors = sns.color_palette("husl", num_routes).as_hex()

    # Create the second map (excludes routes where the start and end stations are the same)
    map_exclude_same = folium.Map(location=[data['start_lat'].mean(),
                                            data['start_lng'].mean()],
                                            zoom_start=12)
    for idx, (_, row) in enumerate(pop_routes_exclude.iterrows()):
        start_lat, start_lng = data.loc[data['start_station_id'] == row['start_station_id'],
                                        ['start_lat', 'start_lng']].iloc[0]
        end_lat, end_lng = data.loc[data['end_station_id'] == row['end_station_id'],
                                    ['end_lat', 'end_lng']].iloc[0]
        line_color = colors[idx]
        folium.PolyLine(locations=[[start_lat, start_lng], [end_lat, end_lng]],
                        color=line_color,
                        weight=10,
                        tooltip=f'Trip count: {row["trip_count"]}').add_to(map_exclude_same)

    map_exclude_same.save(save_file)

def prepare_data_for_prediction(data):
    """
    Prepares the data for predicting the number of rides on a given day.

    Args:
    data (DataFrame): A DataFrame containing bike trip data.

    Returns:
    daily_data (DataFrame): A DataFrame containing the target variable,
    number of rides per day, and the date.
    """
    daily_data = data.groupby('date').size().reset_index(name='num_rides')
    # Rename the columns to 'ds' and 'y' as required by Prophet
    daily_data.columns = ['ds', 'y']

    return daily_data

def train_and_evaluate_model(daily_data):
    """
    Trains and evaluates a time series forecasting model using the given data.

    Args:
    daily_data (DataFrame): The daily data with target variable and date.

    Returns:
    model (Prophet): The trained forecasting model.
    """

    # Define holidays and special events
    holidays = pd.DataFrame({
        'holiday': 'special_event',
        'ds': pd.to_datetime([
            '2022-01-01',  # New Year's Day
            '2022-07-04',  # Independence Day
            '2022-12-25',  # Christmas Day
            '2022-05-31',  # Memorial Day
            '2022-09-06',  # Labor Day
            '2022-11-25',  # Thanksgiving Day
            # Add more holidays
        ]),
        'lower_window': 0,
        'upper_window': 1,
    })

    # Initialize the Prophet model with specified parameters
    model = Prophet(
        interval_width=0.95,  # 95% uncertainty interval
        changepoint_prior_scale=0.1,  # Flexibility for trend changes
        holidays=holidays  # Include holidays and special events
    )

    # Add daily, weekly, and yearly seasonality
    model.add_seasonality(name='daily', period=1, fourier_order=3)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    # Fit the model
    model.fit(daily_data)

    # Perform cross-validation
    df_cv = cross_validation(model, initial='180 days', period='30 days', horizon='7 days')

    # Calculate performance metrics
    df_p = performance_metrics(df_cv)
    print(df_p.head())

    # Create a dataframe for the next 30 days
    future = model.make_future_dataframe(periods=30)

    # Make predictions for the next 30 days
    forecast = model.predict(future)

    # Plot the forecast
    model.plot(forecast)
    plt.savefig('forecast.png', bbox_inches='tight')
    plt.show()

    return model

def main():
    '''
    Main function for the program.
    '''

    csv_files = ['Data/202201.csv',
                 'Data/202202.csv',
                 'Data/202203.csv',
                 'Data/202204.csv',
                 'Data/202205.csv',
                 'Data/202206.csv',
                 'Data/202207.csv',
                 'Data/202208.csv',
                 'Data/202209.csv',
                 'Data/202210.csv',
                 'Data/202211.csv',
                 'Data/202212.csv']

    # Read the data from the CSV files
    data = read_data(csv_files)

    # Process the data
    data = process_data(data)

    # Plot the data
    plot_duration_histogram(data)

    plot_trips_by_user_type(data)

    plot_median_ride_duration_by_user_type(data)

    plot_daily_trips_over_time(data)

    plot_trips_by_day_of_week(data)

    plot_trips_by_hour_of_day(data)

    plot_pop_routes_map_all(data)

    plot_pop_routes_map_exclude_same(data)

    plot_pop_routes_map_all(data, 100)

    plot_pop_routes_map_exclude_same(data, 100)

    # Prepare the data for prediction
    daily_data = prepare_data_for_prediction(data)

    # Train and evaluate the model
    train_and_evaluate_model(daily_data)

if __name__ == '__main__':
    main()
