import os
import urllib.request

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

emojis = {
    "smiley": "1f600.png",
    "wink": "1f609.png",
    "cool": "1f60e.png"
}

BASE_URL = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/"

for name, filename in emojis.items():
    url = BASE_URL + filename
    filepath = os.path.join(ASSETS_DIR, f"{name}.png")
    print(f"Downloading {name} from {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved {filepath}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
