import json, os
import requests
import io
import base64
try :
    from PIL import Image
    import pandas as pd
except :
    requirements = ['pandas', 'pillow']
    for requirement in requirements :
        os.system(f'pip3 install {requirement}')
        print('Done!')

url = "http://127.0.0.1:7860"


def load_unprocessing(path) :
    df = pd.read_csv(path)
    descriptions = [desc.replace('"', '') for desc in df['description']]
    image_paths = [image_id for image_id in df['image_id']]
    return descriptions, image_paths


def implementation(lists, img_lists) :
    for index, item in enumerate(lists) :
        payload = {
            "prompt": item,
            "steps": 15
        }

        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

        r = response.json()
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
        image.save(f'./images/{img_lists[int(index)]}')

def main() :
    os.makedirs('./images', exist_ok=True)
    key = 'descriptions.csv'
    txts, imgs = load_unprocessing(key)
    implementation(txts, imgs)


if __name__ == '__main__' :
    main()