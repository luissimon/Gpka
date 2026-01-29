#! /usr/bin/env python3.8
# -*- coding: utf-8 -*-

import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.string_) or isinstance(obj,np.str_):
            return str(obj)
        if isinstance(obj, dict):
            new_dict={}
            for k in obj.keys(): new_dict[k]=obj[k]
            return new_dict
        return super(NpEncoder, self).default(obj)


