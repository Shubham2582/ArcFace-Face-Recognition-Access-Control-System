import os
import requests
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor
import time

# List of 50+ Indian celebrities (actors, sports, politics, etc.)
CELEBRITIES = [
    "Shah Rukh Khan", "Amitabh Bachchan", "Aamir Khan", "Salman Khan",
    "Deepika Padukone", "Priyanka Chopra", "Virat Kohli", "MS Dhoni",
    "Alia Bhatt", "Ranveer Singh", "Akshay Kumar", "Hrithik Roshan",
    "Kareena Kapoor", "Ranbir Kapoor", "Sachin Tendulkar", "Narendra Modi",
    "Sania Mirza", "PV Sindhu", "Saina Nehwal", "Aishwarya Rai",
    "Rajinikanth", "Kamal Haasan", "Vijay", "Allu Arjun", "Prabhas",
    "Anushka Sharma", "Katrina Kaif", "Sonam Kapoor", "Madhuri Dixit",
    "Sridevi", "Rekha", "Ajay Devgn", "Rani Mukerji", "Vidya Balan",
    "Sushmita Sen", "Sunil Chhetri", "Rahul Dravid", "Sourav Ganguly",
    "Rohit Sharma", "Hardik Pandya", "Jasprit Bumrah", "Neeraj Chopra",
    "Mary Kom", "Rahul Gandhi", "Arvind Kejriwal", "Amit Shah",
    "Saina Nehwal", "Dhanush", "Rajkummar Rao", "Ayushmann Khurrana"
]

def download_image(url, save_path):
    """Download and save an image from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def download_celeb_images(celeb_name, max_images=3):
    """Download images for a celebrity and save to a folder."""
    folder_path = os.path.join("celeb_images", celeb_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Skip if already downloaded
    existing = len(os.listdir(folder_path))
    if existing >= max_images:
        return
    
    print(f"Downloading images for {celeb_name}...")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.images(celeb_name, max_results=max_images)]
            for i, result in enumerate(results[:max_images]):
                ext = os.path.splitext(result['image'])[1].split('?')[0]
                if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                    ext = '.jpg'
                save_path = os.path.join(folder_path, f"{celeb_name}_{i+1}{ext}")
                if not os.path.exists(save_path):
                    download_image(result['image'], save_path)
                    time.sleep(1)  # Avoid rate-limiting
    except Exception as e:
        print(f"Error searching for {celeb_name}: {e}")

def main():
    os.makedirs("celeb_images", exist_ok=True)
    # Use threading for faster downloads
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_celeb_images, CELEBRITIES)

if __name__ == "__main__":
    main()