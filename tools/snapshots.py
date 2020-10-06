import json
import os
import time
from datetime import datetime
from multiprocessing import Pool
from os.path import join

import requests
from lxml import etree


def save_img2local(camera, folder_path):
    url = camera['url']
    img_name = datetime.now().strftime(
        '%Y-%m-%d-%H-%M-%S') + '-' + camera['key'] + '.jpg'
    save2img_path = join(folder_path, img_name)
    resp = requests.get(url)
    if resp.status_code != 200:
        pass
    else:
        img_data = resp.content
        f = open(save2img_path, 'wb')
        f.write(img_data)
        f.close()


def get_camera_list():
    camera_list = list()
    url = 'https://static.data.gov.hk/td/traffic-snapshot-images/code/' \
          'Traffic_Camera_Locations_En.xml'
    resp = requests.get(url)
    root = etree.fromstring(resp.content)
    for child in root:
        camera_list.append({
            str(_key).lower(): child.find('%s' % _key).text
            for _key in [
                'key', 'region', 'district', 'description', 'easting',
                'northing', 'latitude', 'longitude', 'url'
            ]
        })
    with open('./snapshots_info.json', 'w+') as fout:
        json.dump(camera_list, fout, indent=2, ensure_ascii=False)
    return camera_list


def collect_snapshots():
    if os.path.exists('./snapshots_info.json'):
        with open('./snapshots_info.json', 'r') as fin:
            camera_list = json.load(fin)
    else:
        camera_list = get_camera_list()
    folder_date = datetime.now().strftime('%Y%m%d')
    if not os.path.exists('data/snapshots'):
        os.mkdir('data/snapshots')
    folder_path = join('data/snapshots', folder_date)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    my_pool = Pool(5)
    for camera in camera_list:
        my_pool.apply_async(save_img2local, (camera, folder_path))
    my_pool.close()
    my_pool.join()


if __name__ == '__main__':
    while True:
        starting_time = time.time()
        print('Starting One Round at',
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        collect_snapshots()
        ending_time = time.time()
        print('Finish One Round at',
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        time_range = 120 - (ending_time - starting_time)
        print('Sleeping for ' + str(time_range) + 's')
        time.sleep(time_range)
