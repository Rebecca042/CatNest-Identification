import pandas as pd
import numpy as np
import datetime
import calendar

from meteostat import Point

import GenerateProcessDataWeather as processW


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
def generate_case(case_id, setting_behavior, num_events, start_time):
    data = []
    case_date = start_time.date()  # Using the start_time's date

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
            'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'])
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
        current_time += datetime.timedelta(minutes=np.random.randint(10, 45))
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


# === Central Log Generation ===
def generate_log(num_cases=15):
    behaviors = {
        1: SettingTVandCat(),
        2: SettingReadingCat(),
        3: SettingGaming()
    }

    full_log = []
    # Define probabilities for each setting: 40% TV+Cat, 30% Reading+Cat, 30% Gaming
    setting_probs = [0.4, 0.3, 0.3]
    today = datetime.datetime.now()
    default_start = today - datetime.timedelta(days=num_cases)

    for i in range(1, num_cases + 1):
        case_date = default_start + datetime.timedelta(days=i)

        # Generate a realistic start time in the evening (from 6 PM to 10 PM)
        start_time = datetime.datetime.combine(
            case_date,
            datetime.time(np.random.randint(18, 22), np.random.randint(0, 60))
        )

        setting_choice = np.random.choice([1, 2, 3], p=setting_probs)
        setting_behavior = behaviors[setting_choice]
        num_events = {
            1: 6,
            2: np.random.randint(4, 11),
            3: np.random.randint(5, 7)
        }[setting_choice]
        case_log = generate_case(i, setting_behavior, num_events, start_time)
        full_log.extend(case_log)

    return pd.DataFrame(full_log)

np.random.seed(42)
# === Generate Log and Save ===
df = generate_log(50000)
munich = Point(48.1351, 11.5820)
#df = processW.generate_log_weather_dependent(num_cases=20, location=munich)
df = df.sort_values(by=['Case ID', 'Timestamp']).reset_index(drop=True)
print(df.head())
today = datetime.date.today().isoformat()
df.to_csv(f'cat_activity_log_fixed_rng_{today}.csv', index=False)

