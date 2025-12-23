import matplotlib.pyplot as plt   

def d_t_plotter( form="loose", invert=True, **kwargs ):
    '''
    Visualising data, if time windows are mismatched, place kwargs in ascending order

    Parameters: **kwargs -> name:plot ,  name is set as title
    Output: Each series plotted individually, last plot is superposition
    '''

    total_plots = len(kwargs) + 1
    fig, axs = plt.subplots( total_plots, 1 )
    lowest_temp = 1000 # magic numbers
    highest_temp = -1000

    index = 0
    for data in kwargs:
        plt.sca(axs[index])
        simple_plotter( data, kwargs[data], form=form, show="suppress", invert=invert, label="minimal" )
        axs[index].set_title(data)
        low, high = axs[index].get_ylim()

        if low<lowest_temp:
            lowest_temp = low
        if high>highest_temp:
            highest_temp = high

        index += 1

        plt.sca(axs[total_plots - 1])
        simple_plotter( data, kwargs[data], form=form, show="suppress", invert=invert, label="minimal")

    axs[total_plots-1].legend()
    axs[total_plots-1].set_ylim( lowest_temp, highest_temp )

    plt.tight_layout()
    plt.show()

def simple_plotter( title, series, form="tight", label="compact", show="present", invert=True):
    ''' 
    Plots temperature (y) vs time (x)
    Inputs: title -->  name of data
            series --> dict of form date:temp
                        Can be a list of tuples as well (date, temp)
                        Expected to be inverted (most code objects are structured that way)
            decades_span --> decade_code ('198','199' etc), index
    Options:
        form: tight --> fits the axes to series min/max
              loose --> fits the axes from 0 - max+10
        label: compact --> Shows decade labels
    '''
    x = []
    y = []

    min_y = 100000
    max_y = -100000

    try:
        start = list(series.keys())[0]
        end = list(series.keys())[-1]

        for year in range(int(end[0:4])+1, int(start[0:4])) :
            for month in range(0, 13):
                if month < 10:
                    date = f"{str(year)}-0{month}"
                else:
                    date = f"{str(year)}-{month}"

                x.append( date )
                try:
                    temp = series[str(date)]
                    if temp < min_y:
                        min_y = temp
                    elif temp > max_y:
                        max_y = temp
                    y.append( temp )
                except KeyError:
                    y.append(None)
    except:
        # date, temp format
        prev_year = int(series[0][0])
        for i in series:
            year, temp = i[0], i[1]
            while int(year) - prev_year >= 2 and int(year) != prev_year:
                prev_year += 1
                if x[-1] == str(prev_year):
                    continue
                else:
                    x.append(str(prev_year))
                    y.append(None)
            x.append(year)
            y.append(temp)


    if invert == True:
        x = x[::-1]
        y = y[::-1]

    plt.plot( x, y, label=title )
    ax = plt.gca()

    if label == "compact":
        x_axis = ax.axes.get_xticklabels()
        end_decade = max(x)[0:3]
        start_window = min(x)[0:3]
        decades = int(end_decade) - int(start_window)

        if len(str(x_axis[0].get_text())) == 7: #yyyy-mm
            for d in range(decades+1):
                plt.setp(x_axis[120*d+1: 120*d+120], visible=False)
        if len(str(x_axis[0].get_text())) == 4: #yyyy
            for d in range(decades+1):
                plt.setp(x_axis[10*d+1: 10*d+10], visible=False)
    elif label == "minimal":
        x_axis = ax.axes.get_xticklabels()
        plt.setp( x_axis[1:-2], visible=False )

    if form == "loose":
        if min_y < 0:
            plt.ylim( min_y-10, max_y + 10 )
        else:
            plt.ylim( 0, max_y + 10 )

    if show == "suppress":
        return plt

    if show == "present":
        plt.title( title )
        plt.show()

def trend_plotter(warming_periods, cooling_periods, base):

    base = list(base.items())[::-1]
    record = []
    for i in warming_periods:
        for _ in range(int(i["start"]), int(i["end"])):
            record.append((str(_), i["rate"]))
    for i in cooling_periods:
        for _ in range(int(i["start"]), int(i["end"])):
            record.append((str(_), i["rate"]))

    record.sort()

    fig, axs = plt.subplots(2, 1)
    plt.sca(axs[0])
    simple_plotter("RAW", base, show="suppress", invert=False)

    plt.sca(axs[1])
    simple_plotter("Warm_cool", record, show="suppress", invert=False)
    plt.axhline(y=0, color="red", linestyle="--")

    axs[1].legend()
    plt.show()

