import pickle
from meteostat import Hourly, Point
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
import calendar

# Initialize a global cache to store weather data
weather_cache = {}

# === Save Cache to File ===
def save_cache_to_file():
    with open('weather_cache.pkl', 'wb') as f:
        pickle.dump(weather_cache, f)
    print("Cache saved to file.")

# === Load Cache from File ===
def load_cache_from_file():
    global weather_cache  # Make sure to access the global cache variable
    try:
        with open('weather_cache.pkl', 'rb') as f:
            weather_cache = pickle.load(f)
        print("Cache loaded from file.")
    except FileNotFoundError:
        print("No cache file found. Starting fresh.")
        weather_cache = {}

# === Base Class for Activity Setting ===
class SettingBehavior:
    def __init__(self, setting_id):
        self.setting_id = setting_id
        self.transitions = {}

    def initialize_state(self):
        return None

    def get_next_activity(self, current_activity, state):
        raise NotImplementedError("Implement in subclass")


# === Setting 1: TV + Cat ===
class SettingTVandCat(SettingBehavior):
    def __init__(self):
        super().__init__(1)
        self.transitions = {
            'ComingHome': ['Cooking'],
            'Cooking': ['Eating'],
            'Eating': ['Cat-cuddles', 'WatchingTV'],
            'Cat-cuddles': ['WatchingTV', 'Sleeping'],
            'WatchingTV': ['Cat-cuddles', 'Sleeping'],
            'Sleeping': []
        }

    def initialize_state(self):
        return -1  # couch_task_finished

    def get_next_activity(self, current_activity, state):
        if state == 1:
            return 'Cat-cuddles', 0
        elif state == 2:
            return 'WatchingTV', 0
        elif state == 0:
            return 'Sleeping', -1
        elif state == -1:
            possible_next = self.transitions.get(current_activity, ['Sleeping'])
            next_activity = np.random.choice(possible_next)
        else:
            next_activity = 'Sleeping'

        if current_activity == 'Eating':
            state = {'Cat-cuddles': 2, 'WatchingTV': 1}.get(next_activity, -1)
        elif current_activity in ['Cat-cuddles', 'WatchingTV'] and next_activity in ['WatchingTV', 'Cat-cuddles']:
            state = 0
        elif current_activity == 'Sleeping':
            state = -1

        return next_activity, state


# === Setting 2: Reading + Cat ===
class SettingReadingCat(SettingBehavior):
    def __init__(self):
        super().__init__(2)
        self.transitions = {
            'ComingHome': ['Cooking'],
            'Cooking': ['Eating'],
            'Eating': ['Reading(Book)'],
            'Reading(Book)': ['Cat-cuddles'],
            'Cat-cuddles': ['Reading(Book)']
        }

    def initialize_state(self):
        return -1

    def get_next_activity(self, current_activity, state):
        if state == 0:
            return 'Sleeping', -1
        possible_next = self.transitions.get(current_activity, ['Sleeping'])
        return np.random.choice(possible_next), state


# === Setting 3: Gaming ===
class SettingGaming(SettingBehavior):
    def __init__(self):
        super().__init__(3)
        self.transitions = {
            'ComingHome': ['Cooking'],
            'Cooking': ['Eating'],
            'Eating': ['ComputerGames'],
            'ComputerGames': ['Reading(Book)', 'Cat-cuddles', 'Sleeping'],
            'Reading(Book)': ['Sleeping'],
            'Cat-cuddles': ['Sleeping']
        }

    def initialize_state(self):
        return -1

    def get_next_activity(self, current_activity, state):
        if state == 0:
            return 'Sleeping', -1
        possible_next = self.transitions.get(current_activity, ['Sleeping'])
        return np.random.choice(possible_next), state


# === Case Sequence Generator ===
def generate_case(case_id, setting_behavior, num_events, start_time, weather_label):
    data = []
    default_start = start_time
    case_date = default_start#default_start + timedelta(days=case_id)

    current_activity = 'ComingHome'
    current_time = start_time
    state = setting_behavior.initialize_state()

    def additional_features(ts):
        weekday = ts.weekday()
        return {
            'Date': ts.date(),
            'Weekday': calendar.day_name[weekday],
            'IsWeekend': weekday >= 5,
            'Setting': setting_behavior.setting_id,
            'Mood': np.random.choice(['Happy', 'Tired', 'Neutral']),
            'Weather': weather_label  # Use actual weather label
            # Further features as 'PlanWithFriends' or 'OnVacation' are possible
        }

    data.append({
        'Case ID': case_date.strftime('%Y-%m-%d'),
        'Activity': current_activity,
        'Timestamp': current_time,
        **additional_features(current_time)
    })

    for i in range(num_events - 1):
        next_activity, state = setting_behavior.get_next_activity(current_activity, state)
        current_time += timedelta(minutes=np.random.randint(10, 45))
        data.append({
            'Case ID': case_date.strftime('%Y-%m-%d'),
            'Activity': next_activity,
            'Timestamp': current_time,
            **additional_features(current_time)
        })
        current_activity = next_activity
        if current_activity == 'Sleeping':
            break

        if i == num_events - 3 and isinstance(setting_behavior, (SettingReadingCat, SettingGaming)):
            state = 0

    return data

def decide_behavior_from_weather(last_time, location, min_intervals=4):
    # Use the date as a key for caching (we ignore time of the day here)
    date_key = last_time.date()

    # Check if weather data is already cached for this date
    if date_key in weather_cache:
        data = weather_cache[date_key]
        print(f"Using cached weather data for {date_key}")
    else:
        start = last_time
        end = start + timedelta(hours=4)
        data = Hourly(location, start, end).fetch()

        # Check if the data is empty
        if data.empty:
            print("Warning: No data available for the given time period.")
            return 1, 'Unknown'

        # Handle missing data by filling with reasonable defaults
        data.fillna({
            'temp': data['temp'].mean(),
            'prcp': 0,
            'rhum': data['rhum'].mean()  # Filling humidity with the mean value
        }, inplace=True)

        # Ensure the relevant columns are numeric
        data['temp'] = pd.to_numeric(data['temp'], errors='coerce')
        data['prcp'] = pd.to_numeric(data['prcp'], errors='coerce')
        data['rhum'] = pd.to_numeric(data['rhum'], errors='coerce')

        # Store the fetched data in the cache for future use
        weather_cache[date_key] = data

    # Evaluating outdoor conditions
    good_weather = data[
        (data['temp'] > 10) & (data['temp'] < 30) &  # Comfortable outside temperature
        (data['prcp'] <= 1) &  # Allow light rain
        (data['rhum'] < 80)   # Low humidity
    ]

    # Check if good_weather DataFrame is empty
    if good_weather.empty:
        good_count = 0
    else:
        good_count = len(good_weather)

    # Determine behavior based on number of "good" intervals
    if good_count >= min_intervals:
        behavior = 3  # Gaming (good weather)
    elif good_count > 0:
        behavior = 2  # Mixed weather
    else:
        behavior = 1  # TV + Cat (indoor)

    # Determine simplified weather label
    avg_prcp = data['prcp'].mean()
    avg_rhum = data['rhum'].mean()

    if avg_prcp > 0.1:
        weather_label = 'Rainy'
    elif avg_rhum > 70:
        weather_label = 'Cloudy'
    else:
        weather_label = 'Sunny'

    return behavior, weather_label

def generate_log_weather_dependent(num_cases=20, location=Point(52.52, 13.4050)):  # Berlin default
    behaviors = {
        1: SettingTVandCat(),
        2: SettingReadingCat(),
        3: SettingGaming()
    }

    full_log = []

    # Calculate default_start as today minus the number of cases
    today = datetime.now()
    default_start = today - timedelta(days=num_cases)

    for i in range(1, num_cases + 1):
        # Set the day
        case_date = default_start + timedelta(days=i)

        # Generate realistic start time in the evening
        start_time = datetime.combine(
            case_date,
            time(np.random.randint(18, 22), np.random.randint(0, 60))
        )

        # Get weather-dependent behavior using the actual start_time
        setting_choice, weather_label = decide_behavior_from_weather(start_time, location)
        setting_behavior = behaviors[setting_choice]

        # Choose event count
        num_events = {
            1: 6,
            2: np.random.randint(4, 11),
            3: np.random.randint(5, 7)
        }[setting_choice]

        # Generate case
        case_log = generate_case(i, setting_behavior, num_events, start_time, weather_label)
        full_log.extend(case_log)

    # Save the cache after the log generation
    save_cache_to_file()

    return pd.DataFrame(full_log)

# Load the cache at the start of the script
load_cache_from_file()
