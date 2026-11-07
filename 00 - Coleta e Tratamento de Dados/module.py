import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import obspy.core.utcdatetime as utc
import obspy.geodetics as geo
import pandas as pd
from obspy import taup
from obspy.clients.fdsn import Client
from obspy.core.event.catalog import Catalog

import matplotlib.pyplot as plt
import requests
import pandas as pd
import os

SIRIUS_LOCATION = (-22.807606, -47.052526)
POS_LAT_SIRIUS, POS_LON_SIRIUS = SIRIUS_LOCATION


def get_archiver_data(
    pv,
    from_time,
    to_time,
    optimize=-1,
    diff=True,
    base_url="https://ais-eng-srv-ca.cnpem.br/retrieval/data/getData.json",
):
    """
    Retrieve time-series data from the EPICS Archiver Appliance and return it
    as a pandas DataFrame.

    Parameters
    ----------
    pv : str
        Process Variable (PV) name to query.
    from_time : str
        Start timestamp of the query interval. Must be in a format accepted
        by the Archiver Appliance API.
    to_time : str
        End timestamp of the query interval. Must be in a format accepted
        by the Archiver Appliance API.
    optimize : int, optional
        Optimization level for server-side aggregation. If set to -1, raw
        data are retrieved. Otherwise, the PV is requested using the
        ``optimized_<optimize>(PV)`` syntax provided by the Archiver API.
        Default is -1.
    diff : bool, optional
        If ``True`` and an optimized query is used, the returned ``val``
        column is shifted so that its first value becomes zero. This is
        useful for analyzing cumulative quantities as relative variations
        over the selected time interval. Ignored when ``optimize = -1``.
        Default is ``True``.
    base_url : str, optional
        URL of the Archiver Appliance data retrieval endpoint.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the retrieved data.

        For non-optimized queries, the DataFrame contains the fields
        returned directly by the API.

        For optimized queries, the aggregated statistics are expanded into
        separate columns:

        - ``val``: aggregated value (or relative value if ``diff=True``)
        - ``std``: standard deviation
        - ``min``: minimum value
        - ``max``: maximum value
        - ``count``: number of samples used in the aggregation

    Raises
    ------
    requests.HTTPError
        If the HTTP request to the Archiver Appliance fails.

    Notes
    -----
    The function expects the response format of the Archiver Appliance
    ``getData.json`` endpoint, where the payload is available at
    ``response.json()[0]["data"]``.

    When optimization is enabled, the API returns each value as a list
    containing the aggregated value and additional statistics. These are
    automatically unpacked into separate DataFrame columns.
    """

    parameters = {
        "pv": f"optimized_{optimize}({pv})" if optimize != -1 else pv,
        "from": from_time,
        "to": to_time,
    }

    response = requests.get(base_url, params=parameters)

    if not response.ok:
        response.raise_for_status()

    json_data = response.json()[0]["data"]

    df = pd.json_normalize(json_data)

    if optimize != -1:
        df["std"] = df["val"].apply(lambda x: x[1])
        df["min"] = df["val"].apply(lambda x: x[2])
        df["max"] = df["val"].apply(lambda x: x[3])
        df["count"] = df["val"].apply(lambda x: int(x[4]))

        df["val"] = df["val"].apply(lambda x: x[0])
        if diff:
            df["val"] -= df["val"][0]

    return df


def arrival_list(
    latitude: float,
    longitude: float,
    depth: float,
    POS_LAT_SIRIUS=POS_LAT_SIRIUS,
    POS_LON_SIRIUS=POS_LON_SIRIUS,
):
    """
    Function that calculates the list of arrival of an earthquake

    Input:

    - latitude: the latitude of the event in degree

    - longitude: the longitude of the event in degree

    - depth: the estimated depth of the event in kilometers

    Output:

    - travel_times: the list of arrivals of each wave that reaches the target

    - epicenter_distance_degree: the epicenter distance between the origin of the event and the target in degree

    - model: the model used to calculate the travel_time

    - model_name: the name of the model used for register purposes
    """

    # geodesic_distance is in meters, azimuth_degree, back_azimuth_degree and epicenter_distance is in degree
    geodesic_distance, azimuth_degree, back_azimuth_degree = geo.gps2dist_azimuth(
        latitude, longitude, POS_LAT_SIRIUS, POS_LON_SIRIUS
    )
    epicenter_distance_degree = geo.locations2degrees(
        latitude, longitude, POS_LAT_SIRIUS, POS_LON_SIRIUS
    )

    model_name = "iasp91"
    model = taup.TauPyModel(model=model_name)

    travel_times = model.get_travel_times(
        source_depth_in_km=depth, distance_in_degree=epicenter_distance_degree
    )

    return travel_times, epicenter_distance_degree, model, model_name


def earth_plot(model: taup.TauPyModel, depth: float, epicenter_distance_degree: float):
    """
    A plot-function, it takes the epicenter distance, the depth and the model


    """

    paths = model.get_ray_paths(depth, epicenter_distance_degree)

    fig = plt.figure(figsize=(12, 5))

    ax_rays = fig.add_subplot(1, 2, 1, projection="polar")
    ax_times = fig.add_subplot(1, 2, 2)

    paths.plot_rays(legend=True, fig=fig, ax=ax_rays, show=False)
    paths.plot_times(legend=False, fig=fig, ax=ax_times, show=False)

    plt.tight_layout()
    plt.show()


def table_plot(id: str, event_data: pd.DataFrame, time_data: pd.DataFrame):
    """
    This function will get the specific event wanted and will plot the earthquake

    Input:

    - id: the name of the event

    - event_data: the dataframe of the events being used

    - time_data: the dataframe of the arrival times of the events being used
    """

    new_event_data = event_data[event_data["event_id"] == id]
    new_time_data = time_data[time_data["event_id"] == id]

    model_name = new_time_data["model_used"].tolist()[0]

    model = taup.TauPyModel(model=model_name)

    earth_plot(
        model,
        new_event_data["depth"].tolist()[0],
        new_event_data["epicenter_distance"].tolist()[0],
    )


def table_create(
    latitude: float,
    longitude: float,
    depth: float,
    magnitude: float,
    magnitude_type: str,
    event_id: str,
    arrivals: taup.tau.Arrivals,
    UTC_origin: utc.UTCDateTime,
    model_name: str,
    catalog: str,
    epicenter_distance: float,
    event_data: pd.DataFrame = None,
    time_data: pd.DataFrame = None,
):
    """
    Function that add or create a Data Frame in pandas format for the event and it's waves.

    Entry:

    - latitude: the latitude of the event in degree

    - longitude: the longitude of the event in degree

    - depth: the estimated depth of the event in kilometers

    - magnitude: how strong was the earthquake in magnitude type

    - magnitude_type: the type of earthquake used

    - event_id: the name of the event

    - arrivals: the arrival list calculated by ``get_travel_time``

    - UTC_origin: the time date of the event in the UTC scale

    - model_name: the name of the model used to calculate the ray pattern

    - catalog: the source of the earthquake that is being analysed

    - epicenter_distance: the epicenter distance between the source and the target in degree

    - event_data: the data frame of the event, `None` is the default value if none was created until now

    - time_data: the data frame of every single wave that reaches the target, `None` is the default value if none was created until now

    Output:

    - An updated version of event_data or event_data if none was given

    - An updated version of time_data or time_data if none was given
    """

    if event_data is None or time_data is None:
        # Creating data frame if it doesn't exist
        event_data = pd.DataFrame(
            {
                "event_id": [event_id + str(UTC_origin)],
                "latitude": [latitude],
                "longitude": [longitude],
                "epicenter_distance": [epicenter_distance],
                "depth": [depth],
                "magnitude": [magnitude],
                "magnitude_type": [magnitude_type],
                "Source": [catalog],
                "origin_time": [UTC_origin],
                "receptor_latitude": [POS_LAT_SIRIUS],
                "receptor_longitude": [POS_LON_SIRIUS],
            }
        )
        time_data = pd.DataFrame(
            {
                "event_id": [],
                "phase": [],
                "takeoff_angle": [],
                "frequency": [],
                "incidente_angle": [],
                "ray_parameter": [],
                "travel_time": [],
                "arrival_time": [],
                "model_used": [],
            }
        )

        for arrival in arrivals:
            time_data.loc[len(time_data)] = [
                event_id + str(UTC_origin),
                arrival.name,
                arrival.takeoff_angle,
                "Unknown",
                arrival.incident_angle,
                arrival.ray_param,
                arrival.time,
                UTC_origin + arrival.time,
                model_name,
            ]

    else:
        # Adding new data
        event_data.loc[len(event_data)] = [
            event_id + str(UTC_origin),
            latitude,
            longitude,
            epicenter_distance,
            depth,
            magnitude,
            magnitude_type,
            catalog,
            UTC_origin,
            POS_LAT_SIRIUS,
            POS_LON_SIRIUS,
        ]

        for arrival in arrivals:
            time_data.loc[len(time_data)] = [
                event_id + str(UTC_origin),
                arrival.name,
                arrival.takeoff_angle,
                "Unknown",
                arrival.incident_angle,
                arrival.ray_param,
                arrival.time,
                UTC_origin + arrival.time,
                model_name,
            ]

    return event_data, time_data


def analise_catalog(events: Catalog, catalog: str, plotting: bool = False):
    """
    With a given list of events of a catalog, this function will create a dataframe with all the events and each phase wave

    Input:

    - events: the catalog list of all the events tha will be put in the dataframe

    - catalog: the name of the catalog used to get the events

    - plotting: if `True`, all the events will be plotted. The default argument is `False`.

    Output:

    - An updated version of the event dataframe or event_data if none was given

    - An updated version of the time dataframe or time_data if none was given

    """

    event_data, time_data = load_dataframe()

    for event in events:
        pref_origin = event.preferred_origin()
        pref_mag = event.preferred_magnitude()

        if pref_origin is None:
            pref_origin = event.origins[0]

        if pref_mag is None:
            pref_mag = event.magnitudes[0]

        latitude = pref_origin.latitude
        longitude = pref_origin.longitude
        depth = pref_origin.depth / 1000
        magnitude = pref_mag.mag
        magnitude_type = pref_mag.magnitude_type
        event_id = event.event_descriptions[0].text
        origin_time = pref_origin.time

        travel_time, epicenter_distance_degree, model, model_name = arrival_list(
            latitude, longitude, depth
        )

        if plotting:
            earth_plot(model, depth, epicenter_distance_degree)

        event_data, time_data = table_create(
            latitude,
            longitude,
            depth,
            magnitude,
            magnitude_type,
            event_id,
            travel_time,
            origin_time,
            model_name,
            catalog,
            epicenter_distance_degree,
            event_data,
            time_data,
        )

    superficial_waves = []
    for idx, row in event_data.iterrows():
        name = row["event_id"]
        lat = row["latitude"]
        lon = row["longitude"]
        date = str(row["origin_time"])

        superficial_waves.append(
            (
                name,
                calculate_GDM52_arrival_times(
                    date, (lat, lon), (POS_LAT_SIRIUS, POS_LON_SIRIUS)
                ),
            )
        )

    for name, table in superficial_waves:
        for idx, row in table.iterrows():
            phase = row["type"]
            takeoff_angle = "270"
            incident_angle = "0"

            group_velocity = row["U"]
            earth_radius = 6371
            ray_param = earth_radius / group_velocity

            frequency = row["f(mHz)"]
            travel_time = row["travel_time_s"]
            arrival_time = row["arrival_time"]
            model_used = "GDM52"

            time_data.loc[len(time_data)] = [
                name,
                phase,
                takeoff_angle,
                frequency,
                incident_angle,
                ray_param,
                travel_time,
                arrival_time,
                model_used,
            ]

    time_data = time_data.sort_values("arrival_time")

    return event_data, time_data


def save_dataframe(timeframe: pd.DataFrame, eventframe: pd.DataFrame):
    """
    Saves two dataframes in .csv format in a data folder.

    Input:

    - eventframe: the data frame of the events

    - timeframe: the data frame of every single wave that reaches the target
    """

    timeframe.to_csv("Data/time_data.csv", sep=";", index=False)
    eventframe.to_csv("Data/event_data.csv", sep=";", index=False)


def load_dataframe():
    """
    Load the saved data in .csv format and output a dataframe with the earthquakes

    Output:

    - an existing dataframe of the events or an empty dataframe with the correct number of columns


    - an existing dataframe of the phase arrival or an empty dataframe with the correct number of columns
    """

    if os.path.exists("Data/time_data.csv") and os.path.exists("Data/event_data.csv"):
        timeframe = pd.read_csv("Data/time_data.csv", sep=";")
        eventframe = pd.read_csv("Data/event_data.csv", sep=";")
    else:
        eventframe = None
        timeframe = None

    return eventframe, timeframe


def calculate_GDM52_dispersion(
    model="Love", dispersion_type="path", origin_point=(0, 0), receiver_point=(0, 0)
):
    """
    Calculate surface-wave dispersion curves using the GDM52 model.

    Runs the GDM52 Fortran program to obtain phase and group velocities
    for Love or Rayleigh waves, either at a single geographic point or
    along a path between two geographic coordinates.

    Parameters
    ----------
    model : {"Love", "Rayleigh"}, default="Love"
        Surface-wave type to calculate. "Love" selects Love waves,
        while "Rayleigh" switches the GDM52 program to Rayleigh waves.

    dispersion_type : {"point", "path"}, default="path"
        Type of dispersion calculation:
        - "point": calculates dispersion at a single geographic point,
          specified by `origin_point`.
        - "path": calculates dispersion along the path between
          `origin_point` and `receiver_point`.

    origin_point : tuple of (float, float), default=(0, 0)
        Geographic coordinates of the calculation point or path origin,
        expressed as (latitude, longitude), in degrees.

    receiver_point : tuple of (float, float), default=(0, 0)
        Geographic coordinates of the path endpoint, expressed as
        (latitude, longitude), in degrees. Used only when
        `dispersion_type="path"`.

    Returns
    -------
    pandas.DataFrame
        Dispersion results with the following columns:
        - f(mHz): Frequency in millihertz.
        - c(PREM): Reference phase velocity from PREM.
        - c: Phase velocity predicted by GDM52.
        - dc(%): Percentage difference between GDM52 and PREM phase
          velocities.
        - U(PREM): Reference group velocity from PREM.
        - U: Group velocity predicted by GDM52.
        - dU(%): Percentage difference between GDM52 and PREM group
          velocities.

    Raises
    ------
    FileNotFoundError
        If the GDM52 output file cannot be found.
    ValueError
        If the output file contains rows that cannot be parsed as
        floating-point values.
    RuntimeError
        If the GDM52 program reports an error or fails to execute
        successfully.

    Notes
    -----
    This function requires the compiled `GDM52_dispersion.exe` executable
    and the GDM52 program files to be present in the expected directory.

    The function uses the program's generated output file,
    `GDM52_dispersion.out`, to construct the returned DataFrame.

    For path calculations, the resulting dispersion curve can be used
    to investigate frequency-dependent surface-wave propagation between
    the specified locations.

    """
    input_text = ""

    if model == "Rayleigh":
        input_text += f"77\n"

    if dispersion_type == "point":
        input_text += f"2\n"
        input_text += f"{origin_point[0]} {origin_point[1]}\n"
    elif dispersion_type == "path":
        input_text += f"6\n"
        input_text += f"{origin_point[0]} {origin_point[1]}\n"
        input_text += f"{receiver_point[0]} {receiver_point[1]}\n"

    input_text += f"99\n"

    directory = Path.cwd() / "GDM52_program"

    exe = directory / "GDM52_dispersion.exe"
    output = directory / "GDM52_dispersion.out"

    env = os.environ.copy()
    octave_bin = r"C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin"

    env["PATH"] = octave_bin + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        [str(exe)],
        input=input_text,
        capture_output=True,
        text=True,
        cwd=str(directory),
        env=env,
        check=False,
    )

    if result.stderr:
        print("ERRO -> ", result.stderr)
        raise

    with open(output) as f:
        out = f.read().split(f"\n")
        out_cols = ["f(mHz)", "c(PREM)", "c", "dc(%)", "U(PREM)", "U", "dU(%)"]
        out_rows = [[float(x) for x in s.strip().split()] for s in out[5:-1]]

        out_df = pd.DataFrame(columns=out_cols, data=out_rows)

    return out_df


def calculate_GDM52_arrival_times(origin_time, origin_point, receiver_point):
    rayleigh = calculate_GDM52_dispersion(
        model="Rayleigh",
        dispersion_type="path",
        origin_point=origin_point,
        receiver_point=receiver_point,
    )

    rayleigh["type"] = "rayleigh"

    love = calculate_GDM52_dispersion(
        model="Love",
        dispersion_type="path",
        origin_point=origin_point,
        receiver_point=receiver_point,
    )

    love["type"] = "love"

    dist_m, az, baz = geo.gps2dist_azimuth(
        origin_point[0], origin_point[1], receiver_point[0], receiver_point[1]
    )

    df = pd.concat([rayleigh, love]).reset_index(drop=True)

    df = df.reindex(["f(mHz)", "T(s)", "U", "type", "arrival_time"], axis=1)

    df["T(s)"] = 1000 / df["f(mHz)"]
    df["travel_time_s"] = (dist_m / 1000) / df["U"]
    df["arrival_time"] = (
        pd.Timestamp(origin_time) + pd.to_timedelta(df["travel_time_s"], unit="s")
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df
