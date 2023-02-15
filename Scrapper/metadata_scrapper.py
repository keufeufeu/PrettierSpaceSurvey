import requests
import json
from re import compile
from time import sleep
from sys import argv
from pathlib import Path
from bs4 import BeautifulSoup


# Launch the script with the terminal with option as argument or import run_all and add parameter
# Option supported: hubble or webb
# It will generate a file hubble_galaxy.json or webb_galaxy.json
# The file will contain metadata of "Outreach" type image of the telescope such as size, coordinates
# Some filter are apply such as minimum size or observation type image only.
# A manual blacklist as been made with coordonate problem, other stuff i found not frequently

def telescope_option(telescope='hubble'):
    if telescope.lower() in ['hubble', 'hst']:
        return {
        'obs' : "hubble",
        'dns' : "https://esahubble.org/",
        'cdn_dns': "https://cdn.spacetelescope.org/",
        'search' : "?minimum_size=3&facility=2&type=Observation",
        'base_search' : "images/archive/search/",
        'base_img' : "images/",
        'original' : "media/archives/images/original/",
        'large' : "archives/images/large/",
        'publication' : "archives/images/publicationjpg/",
        'screen' : "archives/images/screen/"
        }
    elif telescope.lower() in ['webb', 'jwst']:
        return {
        'obs' : "webb",
        'dns' : "https://esawebb.org/",
        'cdn_dns': "https://cdn.esawebb.org/",
        'search' : "?minimum_size=3&facility=1&type=Observation",
        'base_search' : "images/archive/search/",
        'base_img' : "images/",
        'original' : "media/archives/images/original/",
        'large' : "archives/images/large/",
        'publication' : "archives/images/publicationjpg/",
        'screen' : "archives/images/screen/"
        }
    else:
        raise NameError('Wrong argument, only hubble and webb are supported')


FOLDER_PATH = Path(__file__).parent


def run_all(obs='hubble'):
    url = telescope_option(obs)
    # Scrap id from the page
    ids = get_all_ids(url)
    # Fill ids with the data
    dic_ids = get_galax_info(ids, url)
    # Save in json
    with open(FOLDER_PATH / (url['obs'] + '_galaxy.json'), 'w') as f:
        json.dump(dic_ids, f)


def get_all_ids(url):
    # Go to the search page with some option
    get_page = requests.get(url['dns'] + url['base_search'] + url['search'])
    soup = BeautifulSoup(get_page.text, "html.parser")
    # Get page number
    nombrepage = soup.find("ul", {"class": "pagination"}).text.split('.')[-1]
    # First extract ids on page 1
    print('Scrapping page: 1')
    ids = extract_ids(soup)
    nombreids = soup.find("div", {"class":"col-md-3"}).text.split()[-1]
    # Go to each page of the search and scrap the ids
    for p in range(2, int(nombrepage) + 1):
        sleep(1)
        print('Scrapping page:', p, '/', nombrepage, "|", len(ids), '/', nombreids)
        get_page = requests.get(url['dns'] + url['base_search'] + "page/" + str(p) + "/" + url['search'])
        soup = BeautifulSoup(get_page.text, "html.parser")
        ids = ids + extract_ids(soup)
        #clear_output(wait=True)
    ids = set(ids)
    # Remove blacklisted ids
    to_blacklist = json.load(open(FOLDER_PATH / 'blacklist.json'))
    for i in to_blacklist:
        if i in ids:
            ids.remove(i)
    print(len(ids), 'ids are found')
    return ids


def extract_ids(soup):
    ids = soup.find('div', id='content').find('script').text.split('{')
    ids.pop(0)
    list_ids = []
    term = 'id: '
    # get each id of galaxies
    for i in range(len(ids)):
        start = ids[i].find(term) + len(term) + 1
        end = ids[i].find(',', start) - 1
        list_ids.append(ids[i][start:end])
    return list_ids


def get_galax_info(ids, url):
    dic_ids = {}
    nombreid = len(ids)
    n = 0
    # For each id found, getting data from the page associated with
    for i in ids:
        n += 1
        print('getting:', i, '| progress:', n, '/', nombreid)
        info = get_info_obs(i, url)
        if info is not None:
            # Filter low fov out, because survey image are too low resolution.
            if info['FOV']['Width'] > 0.5 and info['FOV']['Height'] > 0.5:
                dic_ids[i] = info
        sleep(0.5)
    print('Data from', len(dic_ids), 'observations are scrapped')
    return dic_ids


def get_info_obs(iden, url):
    # Go to the page of one observation
    url_id = url['dns'] + url['base_img'] + iden + '/'
    get_page = requests.get(url_id)
    soup = BeautifulSoup(get_page.text, "html.parser")

    oritemp = soup.find(text="Orientation:")
    if oritemp is None:
        return None
    oritemp = oritemp.next_element.text.split()[2:4]
    orientation = float(oritemp[0][:-1])
    if oritemp[1] == "right":
        orientation *= -1
    
    exclude_category = ['Early Universe', 'Planet', 'Solar System']
    # Find elements in the hubble page
    if url['obs'] == 'hubble':
        typ = soup.find_all(text='Type:')
        if len(typ) > 1:
            typ = typ[1].next_element.text.split(':')
            for t in typ:
                if t.strip() in exclude_category:
                    return None

        date = soup.find(text="Release date:").next_element.text.split(',')[0]
        size = soup.find(text="Size:").next_element.text.split()
    
    # Find elements in the webb page
    if url['obs'] == 'webb':
        category = soup.find(text=compile('^\s*Category:\s*$')).next_element.find_all('a')
        for t in category:
            if t.text.strip() in exclude_category:
                return None

        date = soup.find(text=compile('^\s*Release date:\s*$')).next_element.next_element.text
        date = date.split(',')[0].replace('\n','')
        size = soup.find(text=compile('^\s*Size:\s*$')).next_element.next_element.text.split()

    name = soup.find(text=compile('^\s*Name:\s*$'))
    if name is not None:
        name = name.next_element.text

    distance = soup.find(text=compile('^\s*Distance:\s*$'))
    if distance is not None:
        distance = distance.next_element.text.replace('\n', '')
        
    # Common betweed hubble and webb site
    ra = soup.find(text="Position (RA):").next_element.text
    dec = soup.find(text="Position (Dec):").next_element.text

    ra_temp = ra.split()
    ra_deg = (int(ra_temp[0]) + int(ra_temp[1]) / 60 + float(ra_temp[2]) / 3600) * 15
    dec_temp = [_[:-1] for _ in dec.split()]
    if int(dec_temp[0]) < 0:
        dec_deg = int(dec_temp[0]) - int(dec_temp[1]) / 60 - float(dec_temp[2]) / 3600
    else:
        dec_deg = int(dec_temp[0]) + int(dec_temp[1]) / 60 + float(dec_temp[2]) / 3600

    fov = soup.find(text="Field of view:")
    if fov is None:
        return None
    fov = fov.next_element.text.replace('x', '').split()
    ffov = [float(fov[0]), float(fov[1])]

    credit = soup.find(text="Credit:").find_parent().find_next_sibling().text
    credit = credit.replace('\n', ' ').replace('\xa0', ' ')

    # Formating data for json storage
    obs_photo = {
        'Original' : url['dns'] + url['original'] + iden + '.tif',
        'Large' : url['cdn_dns'] + url['large'] + iden + '.jpg',
        'Publication' : url['cdn_dns'] + url['publication'] + iden + '.jpg',
        'Screensize' : url['cdn_dns'] + url['screen'] + iden + '.jpg'
    }
    
    info_galaxy = {
        'Name': name,
        'Date': date,
        'Size': {
            'Width' : int(size[0]),
            'Height' : int(size[2]),
            'Ratio': int(size[2]) / int(size[0])
        },
        'Position': {
            'RA': ra,
            'Dec': ' '.join(dec_temp),
            'RA_d' : ra_deg,
            'Dec_d' : dec_deg
        },
        'FOV': {
            'Width' : ffov[0],
            'Height' : ffov[1]
        },
        'Orientation': orientation,
        'Distance': distance,
        'Credit': credit,
        'Img_url': obs_photo,
    }
    return info_galaxy


def gen_credit():
    pass


if __name__ == '__main__':
    if len(argv) > 1:
        run_all(argv[1])
    else:
        run_all()