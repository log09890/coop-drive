'''
OpenStreetMap -> OpenDRIVE map for CARLA 0.9.10 (carla.Osm2Odr), e.g. Nga Tu So, Hanoi:

  python osm_map.py nga_tu_so.osm maps/NgaTuSo.xodr                 # bridges dropped, everything at grade
  python osm_map.py nga_tu_so.osm maps/NgaTuSoHigh.xodr --elevated  # flyover / elevated ring road 6 m per OSM layer
  python dashboard.py --town NgaTuSo --vn          # any maps/<Town>.xodr is loaded as a generated world

--elevated: CARLA 0.9.10 cannot build overlapping roads (tested: elevated -> server crash, at grade -> "failed to
connect to newly created map"); kept for newer CARLA versions. Motorbikes are banned from the Nga Tu So flyover
anyway, the chaos is on the ground.
Map data (c) OpenStreetMap contributors, ODbL 1.0.
'''
import os
import sys
import xml.etree.ElementTree as ET
from simulation.connection import carla


def drop_elevated(osm_text):
    '''Remove ways tagged bridge=* (not "no") or layer > 0; returns the new OSM XML and how many were dropped.'''
    root = ET.fromstring(osm_text)
    dropped = 0
    for way in root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
        try:
            layer = int(tags.get('layer', '0'))
        except ValueError:
            layer = 0
        if tags.get('bridge', 'no') != 'no' or layer > 0:
            root.remove(way)
            dropped += 1
    return ET.tostring(root, encoding='unicode'), dropped


def main(src, dst, flat=True, lane_width=3.2):
    with open(src, encoding='utf-8') as f:
        osm = f.read()
    dropped = 0
    if flat:
        osm, dropped = drop_elevated(osm)
    settings = carla.Osm2OdrSettings()
    settings.elevation_layer_height = 6.0  # OSM layer=1 (flyover, elevated ring road 2) -> 6 m above the street
    settings.default_lane_width = lane_width  # Hanoi lanes are narrower than the 4 m default
    dst, here = os.path.abspath(dst), os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps'))  # netconvert reads ../carla/data/xsd (copied there)
    try:
        xodr = carla.Osm2Odr.convert(osm, settings)
    finally:
        os.chdir(here)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(xodr)
    print(f'{dst}: {len(xodr) // 1024} KB OpenDRIVE, {dropped} elevated ways dropped')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], flat='--elevated' not in sys.argv)
