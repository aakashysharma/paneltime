#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
from datetime import date
from . import model_parser
import pandas as pd
import json
import csv
forbidden_names = ['tobit_low', 'tobit_high',
                   model_parser.DEFAULT_INTERCEPT_NAME, model_parser.CONST_NAME]


def load(filepath_or_buffer, sep=None, header="infer",
         names=None, index_col=None, usecols=None, squeeze=False, prefix=None,
         mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
         true_values=None, false_values=None, skipinitialspace=False, skiprows=None,
         skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True,
         verbose=False, skip_blank_lines=True, parse_dates=False, infer_datetime_format=False,
         keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True,
         iterator=False, chunksize=None, compression="infer", thousands=None, decimal=".",
         lineterminator=None, quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True,
         escapechar=None, comment=None, encoding=None, dialect=None, error_bad_lines=True,
         warn_bad_lines=True, delim_whitespace=False,
         memory_map=False, float_precision=None):
    print("opening file ...")
    sep = get_sep(filepath_or_buffer, sep)
    data = pd.read_csv(filepath_or_buffer, sep, sep, header,
                       names, index_col, usecols, squeeze, prefix,
                       mangle_dupe_cols, dtype, engine, converters,
                       true_values, false_values, skipinitialspace, skiprows,
                       skipfooter, nrows, na_values, keep_default_na, na_filter,
                       verbose, skip_blank_lines, parse_dates, infer_datetime_format,
                       keep_date_col, date_parser, dayfirst, cache_dates,
                       iterator, chunksize, compression, thousands, decimal,
                       lineterminator, quotechar, quoting, doublequote,
                       escapechar, comment, encoding, dialect, error_bad_lines,
                       warn_bad_lines, delim_whitespace,
                       memory_map=memory_map, float_precision=float_precision)
    print("... done")
    load_data_printout(data)
    return data


def load_json(path_or_buf=None, orient=None, typ="frame", dtype=None, convert_axes=None, convert_dates=True,
              keep_default_dates=True, numpy=False, precise_float=False, date_unit=None, encoding=None,
              lines=False, chunksize=None, compression="infer"):
    return pd.read_json(path_or_buf, orient, typ, dtype, convert_axes, convert_dates,
                        keep_default_dates, numpy, precise_float, date_unit,
                        encoding, lines, chunksize, compression)


def load_data_printout(data):
    print("The following variables were loaded:"+', '.join(data.keys()))


def append(d, key, i):
    if key in d:
        d[key].append(i)
    else:
        d[key] = [i]


def load_SQL(sql_string, conn, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None):
    return pd.read_sql(sql_string, conn, index_col, coerce_float, params, parse_dates, columns, chunksize)


def get_sep(fname, sep):
    f = open(fname, 'r')
    r = []
    sample_size = 20
    for i in range(sample_size):
        r.append(f.readline())
    f.close()
    d = {}
    for i in [sep, ';', ',', '\t', ' ']:  # checks whether the separator is consistent
        len0 = len(r[0].split(i))
        err = False
        for j in r[1:]:
            rlen = len(j.split(i))
            if rlen != len0:
                err = True
                break
        if not err and rlen > 1:
            d[i] = rlen
    maxlen = max([d[i] for i in d])
    if ',' in d:
        if d[','] == maxlen:
            if len(d) > 0:
                d.pop('.')
                maxlen = max([d[i] for i in d])
    for i in d:
        if d[i] == maxlen:
            return i


SQL_type_dict = {0: float,
                 1: int,
                 2: int,
                 3: float,
                 4: float,
                 5: float,
                 6: float,
                 8: int,
                 9: int,
                 16: int,
                 246: int
                 }
