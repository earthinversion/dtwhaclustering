"""
DTW HAC analysis support functions (`analysis_support`)

:author: Utpal Kumar, Institute of Earth Sciences, Academia Sinica
"""

import time
from datetime import datetime, timedelta


def dec2dt(start):
    """
    Convert the decimal type time array to the date-time type array

    :param start: list or numpy array of decimal year values e.g., [2020.001]
    :type start: list
    :return: date-time type array
    :rtype: list
    """
    results = []
    for st in start:
        year = int(st)
        rem = st - year
        base = datetime(year, 1, 1)
        result = base + timedelta(
            seconds=(base.replace(year=base.year + 1) -
                     base).total_seconds() * rem
        )
        results.append(result)
    return results


def dec2dt_scalar(st):
    """
    Convert the decimal type time value to the date-time type 

    :param st: scalar decimal year value e.g., 2020.001
    :return: time as datetime type
    :rtype: str
    """
    year = int(st)
    rem = st - year
    base = datetime(year, 1, 1)
    result = base + timedelta(
        seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
    )
    return result


def toYearFraction(date):
    """
    Convert the date-time type object to decimal year

    :param date: the date-time type object
    :return: decimal year
    :rtype: float
    """

    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction
