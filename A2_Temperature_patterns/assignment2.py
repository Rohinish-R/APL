import csv

# from plotting_lib import *

### Any object with records in the name is of type {date:temp}
# Extraction functions -> interface with external files
# Modifier functions -> Chainable operations on records
# Analysis functions -> Provides insights on data
# API calls -> Functions linking the above in a meaningful context


# =============================================================================
# Extraction functions
# =============================================================================
def get_city_temperatures(filename, city_name, start_window="0", end_window="9999"):
    """File Wrapper for Filter function"""

    start_window = str(start_window)
    end_window = str(end_window)
    temperature_data = {}

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)

        temperature_data = filter(reader, city_name, start_window, end_window)

    return temperature_data


def get_city_averages(filename, cities_list, window_start="0"):
    cities = {
        record[0]: {"country": record[1], "sum_temp": 0, "points": 0}
        for record in cities_list
    }

    with open(filename, "r") as temperature_file:
        reader = csv.reader(temperature_file)
        next(reader)
        for row in reader:
            if row[0] < window_start or row[0] == "City":
                continue
            try:
                cities[row[3]]["sum_temp"] += float(row[1])
                cities[row[3]]["points"] += 1
            except ValueError:
                continue

    for city in cities:
        try:
            cities[city]["average"] = cities[city]["sum_temp"] / cities[city]["points"]
        except ZeroDivisionError:
            cities[city]["average"] = None

    return cities


# =============================================================================
# Modifier functions
# =============================================================================


def filter(reader, city_name, start_window="0", end_window="9999"):
    """
    Extract temperature data for a specific city in a specific time period from given data
    Time period = [start_window, end_window)

    Parameters: data (list): list(reader) object
                city_name (str): Name of the city to extract data for
                [start_window, end_window)

    Raw data:
    dt, AverageTemperature, AverageTemperatureUncertainty, City, Country, Latitude, Longitude

    Returns:
    dict: Dictionary mapping 'YYYY-MM' to temperature (float)
          Returns empty dict if city not found
    """
    data = list(reader)

    temperature_data = {}
    for row in data[::-1]:
        # Check if this row matches our city
        if row[3] == city_name:
            if row[0] < start_window:
                continue
            if row[0] < end_window:
                # Extract year-month from date (format: 1849-01-01 -> 1849-01)
                date_str = row[0]
                year_month = date_str[:7]  # Take first 7 characters (YYYY-MM)

                # Get temperature, handle missing values
                temp_str = row[1]
                if temp_str and temp_str.strip():  # Check if not empty
                    try:
                        temperature = float(temp_str)
                        temperature_data[year_month] = temperature
                    except ValueError:
                        # Skip rows with invalid temperature data
                        continue

    return temperature_data


def get_annual_average(records):
    """Takes an average of temperatures in a year
    If no readings in a year, year is skipped
    input is dict of format {date:temp}"""

    log = {}
    prev_year = "9999"
    for data in records:
        year = data[0:4]
        try:
            temp = float(records[data])
        except ValueError:
            continue

        if year < prev_year:
            log[year] = {"sum": temp, "points": 1}
            prev_year = year
        else:
            log[year]["sum"] += temp
            log[year]["points"] += 1

    results = {}
    for year in log:
        results[year] = log[year]["sum"] / log[year]["points"]

    return results


def get_window_average(window_size, annual_data_records):
    """Performs a uniformly weighted centered moving average on the given data
    Input -> window_size: number of elements
             annual_data_records: dict of date:temp values
    Output -> window_avg_records: dict of (date, window_avg_temp) values
    """
    annual_data = list(annual_data_records.items())
    window_avg_records = {}

    if (window_size == 0) or (window_size >= len(annual_data)):
        for i in range(len(annual_data)):
            window_avg_records[annual_data[i][0]] = annual_data[i][1]

    lower_bound = window_size // 2
    upper_bound = window_size - lower_bound
    marginal_sum = 0

    # Centered moving average
    for i in range(lower_bound, len(annual_data) - upper_bound):
        date = annual_data[i][0]

        # Filling initial
        if i == lower_bound:
            for k in range(window_size):
                marginal_sum += annual_data[k][1]
            window_avg_records[date] = marginal_sum / window_size
            continue

        # Updating
        marginal_sum -= annual_data[i - lower_bound][1]
        marginal_sum += annual_data[i + upper_bound][1]
        window_avg_records[date] = marginal_sum / window_size

    return window_avg_records


# =============================================================================
# Analysis functions
# =============================================================================


def get_available_cities(filename, limit=None):
    """
    Get list of unique cities in the dataset.

    Parameters:
    filename (str): Path to the CSV file
    limit (int): Maximum number of cities to return (None for all)

    Returns:
    list: List of unique city names
    """
    cities = set()

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            cities.add((row["City"], row["Country"], row["Latitude"]))
            if limit and len(cities) >= limit:
                break

    def lat_func(rec):
        return rec[2]

    return sorted(list(cities), key=lat_func)


def get_period_average(record, mode="object"):
    """Flat average
    Input: record -> dataset
           mode ->
            if records is a list of (date, temp) pairs then mode = list
            else mode = "object"

    Output: Average_temp -> float
            data_points -> int

    if no data_points then Average_temp = None
    """
    sum_temps = 0
    data_points = 0

    if mode == "list":
        record = record
        for data in record:
            try:
                sum_temps += float(data[1])
                data_points += 1
            except:
                continue
    elif mode == "object":
        record = record.values()
        for data in record:
            try:
                sum_temps += float(data)
                data_points += 1
            except:
                continue

    if data_points == 0:
        return None, 0
    average_temp = sum_temps / data_points
    return average_temp, data_points


def get_slope(records):
    """Slope of the least squares line for the dataset
    Input -> dict of date:temp values
    Output -> float of slope
    """
    x = list(records.keys())
    y = list(records.values())
    starting_x = int(x[0])
    starting_y = float(records[x[0]])
    normal_x = [int(date) - starting_x for date in x]
    normal_y = [float(temp) - starting_y for temp in y]

    n = len(x)

    sigma_x = sum(normal_x)
    sigma_y = sum(normal_y)
    sigma_xy = sum(normal_x[i] * normal_y[i] for i in range(n))
    sigma_x2 = sum(xval * xval for xval in normal_x)

    try:
        slope = (n * sigma_xy - sigma_x * sigma_y) / (n * sigma_x2 - sigma_x * sigma_x)
    except ZeroDivisionError:
        slope = None

    intercept = (sigma_y - slope * sigma_x) / n

    return slope, intercept


def build_trends(records):
    """Returns periods of warming and cooling
    Input - dict of date:temp
    Output -> warming_period: [{'start': year, 'end': year, 'rate': float}],
              cooling_period: [{'start': year, 'end': year, 'rate': float}],
    """

    class State:
        def __init__(self, date, temp):
            self.trend = None
            self.start_date = date
            self.end_date = date
            self.start_temp = temp
            self.end_temp = temp

        def update(self, date, temp):
            self.end_date = date
            self.end_temp = temp

        def analysis(self):
            rate = (self.end_temp - self.start_temp) / (
                int(self.end_date) - int(self.start_date)
            )
            period_analysis = {
                "start": self.start_date,
                "end": self.end_date,
                "rate": rate,
            }
            return period_analysis

        def break_trend(self, status, date, temp):
            self.trend = status
            self.start_date = date
            self.end_date = date
            self.start_temp = temp
            self.end_temp = temp

    data = list(records.items())[::-1]
    warming_periods, cooling_periods = [], []

    init = data[0]
    year, prev_temp = init  # Data is (date, temp)

    system = State(year, prev_temp)
    status = None

    i = 0
    end_year = data[-1][0]
    while int(year) < int(end_year) and i < len(data):
        if data[i][0] == year:
            date, temp = data[i]
            i += 1
        else:
            date, temp = year, None

        if (temp == prev_temp) or (temp is None) or (prev_temp is None):
            status = None
        elif temp > prev_temp:
            status = "warming"
        elif temp < prev_temp:
            status = "cooling"

        if (status == None) or (status == system.trend):
            prev_temp = temp
        elif status != system.trend:
            system.update(date, temp)
            period_analysis = system.analysis()
            system.break_trend(status, date, temp)

            if period_analysis["rate"] >= 0:
                warming_periods.append(period_analysis)
            else:
                cooling_periods.append(period_analysis)

        year = str(int(year) + 1)

    return warming_periods, cooling_periods


# =============================================================================
# API functions
# =============================================================================


def find_temperature_extremes(filename, city_name):
    """Find the hottest and coldest months on record for a city.
    Params:
    filename(str) -> name of .csv file
    city_name(str) -> city name (case sensitive)

    Output:
    { { "hottest": {"date": str, "temperature": float} }, { "coldest": {"date": str, "temperature": float} }}
    """
    results = {
        "hottest": {"date": None, "temperature": float("nan")},
        "coldest": {"date": None, "temperature": float("nan")},
    }

    records = get_city_temperatures(filename, city_name)
    if records == {}:
        return results

    dates = records.keys()
    results["hottest"]["temperature"] = max(records.values())
    results["coldest"]["temperature"] = min(records.values())

    for date in dates:
        temp = records[date]
        if temp == results["hottest"]["temperature"]:
            results["hottest"]["date"] = date
        if temp == results["coldest"]["temperature"]:
            results["coldest"]["date"] = date

    # Plotting
    # simple_plotter("Madras", records, form="loose", invert=False, label="minimal")

    return results


def get_seasonal_averages(filename, city_name, season):
    """
    Calculate average temperature for a specific season across all years.

    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city
    season (str): 'spring', 'summer', 'fall', or 'winter'

    Returns:
    dict: {
        'city': str,
        'season': str,
        'average_temperature': float
    }

    Assume: Spring = Mar,Apr,May; Summer = Jun,Jul,Aug; Fall = Sep,Oct,Nov; Winter = Dec,Jan,Feb
    """
    result = {
        "city": city_name,
        "season": season,
        "average_temperature": float("nan"),
    }

    season_map = {
        "spring": ("03", "04", "05"),
        "summer": ("06", "07", "08"),
        "fall": ("09", "10", "11"),
        "winter": ("12", "01", "02"),
    }

    records = get_city_temperatures(filename, city_name)
    if records == {}:
        return result

    sum_temps = 0
    months_recorded = 0
    required_months = season_map[season]

    for date in records:
        month = date[5:7]
        if month in required_months:
            months_recorded += 1
            sum_temps += records[date]

    try:
        seasonal_average = sum_temps / months_recorded
        result["average_temperature"] = seasonal_average
    except ZeroDivisionError:
        seasonal_average = float("nan")

    return result


def compare_decades(filename, city_name, decade1, decade2):
    """
    Compare average temperatures between two decades for a city.

    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city
    decade1 (int): First decade (e.g., 1980 for 1980s)
    decade2 (int): Second decade (e.g., 2000 for 2000s)

    Returns:
    dict: {
        'city': str,
        'decade1': {'period': '1980s', 'avg_temp': float, 'data_points': int},
        'decade2': {'period': '2000s', 'avg_temp': float, 'data_points': int},
        'difference': float,
        'trend': str  # 'warming', 'cooling', or 'stable'
    }
    """

    class Results:
        def __init__(self):
            self.data = {
                "city": city_name,
                "decade1": {
                    "period": decade1,
                    "avg_temp": avg_decade1,
                    "data_points": n_decade1,
                },
                "decade2": {
                    "period": decade2,
                    "avg_temp": avg_decade2,
                    "data_points": n_decade2,
                },
                "difference": temp_difference,
                "trend": trend,  # 'warming', 'cooling', or 'stable'
            }

    end_d1 = str(int(decade1) + 10)
    end_d2 = str(int(decade2) + 10)

    data_decade1 = get_city_temperatures(filename, city_name, decade1, end_d1)
    data_decade2 = get_city_temperatures(filename, city_name, decade2, end_d2)

    avg_decade1, n_decade1 = get_period_average(data_decade1)
    avg_decade2, n_decade2 = get_period_average(data_decade2)
    if avg_decade1 is None or avg_decade2 is None:
        print("missing_data")
        temp_difference = float("nan")
        trend = None

        results = Results()
        return results.data

    temp_difference = avg_decade2 - avg_decade1

    if int(decade2) < int(decade1):
        temp_difference = -1 * temp_difference
    if temp_difference > 0:
        trend = "warming"
    elif temp_difference < 0:
        trend = "cooling"
    else:
        trend = "stable"

    results = Results()
    return results.data


def find_similar_cities(filename, target_city, tolerance=2.0):
    """
    Find cities with similar average temperatures to the target city.

    Parameters:
    filename (str): Path to the CSV file
    target_city (str): Reference city name
    tolerance (float): Temperature difference threshold in °C

    Returns:
    dict: {
        'target_city': str,
        'target_avg_temp': float,
        'similar_cities': [
            {'city': str, 'country': str, 'avg_temp': float, 'difference': float}
        ],
        'tolerance': float
    }
    """
    result = {
        "target_city": target_city,
        "target_avg_temp": float("nan"),
        "similar_cities": [],
        "tolerance": tolerance,
    }

    cities_list = get_available_cities(filename)
    cities_avg = get_city_averages(
        filename, cities_list
    )  # Holds country, average_temp data

    try:
        target_temp = (
            cities_avg[target_city]["sum_temp"] / cities_avg[target_city]["points"]
        )
    except KeyError:
        return result
    except ZeroDivisionError:
        return result

    similar_cities = []
    for city in cities_avg:
        if city == target_city:
            continue
        if cities_avg[city]["average"] is None:
            continue

        diff = target_temp - cities_avg[city]["average"]
        if diff < tolerance and diff > -1 * tolerance:
            data = {
                "city": city,
                "country": cities_avg[city]["country"],
                "avg_temp": cities_avg[city]["average"],
                "difference": diff,
            }
            similar_cities.append(data)

    result["target_avg_temp"] = target_temp
    result["similar_cities"] = similar_cities
    return result


def get_temperature_trends(filename, city_name, window_size=5):
    """
    Calculate temperature trends using moving averages and identify patterns.

    Parameters:
    filename (str): Path to the CSV file
    city_name (str): Name of the city
    window_size (int): Number of years for moving average calculation

    'warming_periods': [{'start': year, 'end': year, 'rate': float}],
    """

    class Results:
        def __init__(
            self,
            annual_data_records={},
            moving_averages={},
            overall_slope=float("nan"),
            warming_periods=[],
            cooling_periods=[],
        ):
            self.data = {
                "city": city_name,
                "raw_annual_data": annual_data_records,  # Annual averages in dict form
                "moving_averages": moving_averages,  # Moving averages
                "trend_analysis": {
                    "overall_slope": overall_slope,  # °C per year
                    "warming_periods": warming_periods,
                    "cooling_periods": cooling_periods,
                },
            }
            annual_data_records, moving_averages, warming_periods, cooling_periods = (
                None,
                None,
                None,
                None,
            )

    records = get_city_temperatures(filename, city_name)
    if records == {}:
        results = Results()
        return results

    # Annual average
    annual_data_records = get_annual_average(records)
    # Centered moving average
    window_avg_records = get_window_average(window_size, annual_data_records)
    # Trends
    warming_periods, cooling_periods = build_trends(window_avg_records)
    # Best fit
    overall_slope, intercept = get_slope(annual_data_records)

    # Plotting
    # trend_plotter( warming_periods, cooling_periods, window_avg_records )

    results = Results(
        annual_data_records,
        window_avg_records,
        overall_slope,
        warming_periods,
        cooling_periods,
    )
    return results.data


# =============================================================================
# TESTING CODE
# =============================================================================


def test_api_functions():
    """
    Test all API functions with sample data.
    """
    filename = "temperature_data.csv"
    test_city = "Madras"

    print("Testing Temperature Data API")
    print("=" * 40)

    # Test basic function
    temps = get_city_temperatures(filename, test_city)
    print(f"Basic function: Found {len(temps)} temperature records")

    # Test extremes
    extremes = find_temperature_extremes(filename, test_city)
    print(f"Extremes: Hottest = {extremes['hottest']['temperature']}°C")

    # Test seasonal averages
    summer_avg = get_seasonal_averages(filename, test_city, "summer")
    print(f"Seasonal: Summer average = {summer_avg['average_temperature']:.1f}°C")

    # Test decade comparison
    comparison = compare_decades(filename, test_city, 1980, 2000)
    print(f"Decades: Temperature change = {comparison['difference']:.2f}°C")

    # Test similar cities
    similar = find_similar_cities(filename, test_city, tolerance=3.0)
    print(f"Similar cities: Found {len(similar['similar_cities'])} matches")

    # Test trends
    trends = get_temperature_trends(filename, test_city)
    print(
        f"Trends: Overall slope = {trends['trend_analysis']['overall_slope']:.4f}°C/year"
    )


if __name__ == "__main__":
    test_api_functions()

