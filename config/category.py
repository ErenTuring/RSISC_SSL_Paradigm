# -*- coding:utf-8 -*-
'''
Catrgory information.

Version 1.0  2019-09-19 20:21:14
'''


class NWPU_RESISC45(object):
    ''' Category details of NWPU_RESISC45 dataset. '''
    plan_name = 'NWPU_RESISC45'
    table = {
        "airplane": 0,
        "airport": 1,
        "baseball_diamond": 2,
        "basketball_court": 3,
        "beach": 4,
        "bridge": 5,
        "chaparral": 6,
        "church": 7,
        "circular_farmland": 8,
        "cloud": 9,
        "commercial_area": 10,
        "dense_residential": 11,
        "desert": 12,
        "forest": 13,
        "freeway": 14,
        "golf_course": 15,
        "ground_track_field": 16,
        "harbor": 17,
        "industrial_area": 18,
        "intersection": 19,
        "island": 20,
        "lake": 21,
        "meadow": 22,
        "medium_residential": 23,
        "mobile_home_park": 24,
        "mountain": 25,
        "overpass": 26,
        "palace": 27,
        "parking_lot": 28,
        "railway": 29,
        "railway_station": 30,
        "rectangular_farmland": 31,
        "river": 32,
        "roundabout": 33,
        "runway": 34,
        "sea_ice": 35,
        "ship": 36,
        "snowberg": 37,
        "sparse_residential": 38,
        "stadium": 39,
        "storage_tank": 40,
        "tennis_court": 41,
        "terrace": 42,
        "thermal_power_station": 43,
        "wetland": 44,
    }

    names = [name for (name, _) in table.items()]
    inds = table.values()

    num = len(table)

    color_table = None
    bgr_table = None  # BGR
    mapping = None


class AID(object):
    ''' Category details of AID dataset. '''
    plan_name = 'AID'
    table = {
        "Airport": 0,
        "BareLand": 1,
        "BaseballField": 2,
        "Beach": 3,
        "Bridge": 4,
        "Center": 5,
        "Church": 6,
        "Commercial": 7,
        "DenseResidential": 8,
        "Desert": 9,
        "Farmland": 10,
        "Forest": 11,
        "Industrial": 12,
        "Meadow": 13,
        "MediumResidential": 14,
        "Mountain": 15,
        "Park": 16,
        "Parking": 17,
        "Playground": 18,
        "Pond": 19,
        "Port": 20,
        "RailwayStation": 21,
        "Resort": 22,
        "River": 23,
        "School": 24,
        "SparseResidential": 25,
        "Square": 26,
        "Stadium": 27,
        "StorageTanks": 28,
        "Viaduct": 29,
    }

    names = [name for (name, _) in table.items()]
    inds = table.values()

    num = len(table)

    color_table = None
    bgr_table = None  # BGR
    mapping = None


class Unlabeled(object):
    ''' Unlabeled self-supervised dataset. '''
    plan_name = 'SSD'
    table = {
        "Unlabeled": 0,
    }

    names = [name for (name, _) in table.items()]
    inds = table.values()

    num = len(table)

    color_table = None
    bgr_table = None  # BGR
    mapping = None


class EuroSAT(object):
    plan_name = 'EuroSAT'

    table = {
        "Highway": 0,
        "Industrial": 1,
        "Pasture": 2,
        "PermanentCrop": 3,
        "Residential": 4,
        "River": 5,
        "SeaLake": 6,
        "AnnualCrop": 7,
        "Forest": 8,
        "HerbaceousVegetation": 9,
    }

    names = [name for (name, _) in table.items()]
    inds = table.values()
    num = len(table)
    color_table = None
    bgr_table = None  # BGR
    mapping = None


def get_category(ds):
    if ds == 'aid':
        return AID()
    elif ds == 'nr':
        return NWPU_RESISC45()
    elif ds in ['eurorgb', 'euroms', 'euromsrgb']:
        return EuroSAT()
