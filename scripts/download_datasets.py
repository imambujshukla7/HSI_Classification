import requests
import os

def download_file(url, dest_folder, file_name):
    """
    Download a file from a URL and save it to a destination folder.

    Parameters:
    url (str): URL of the file to be downloaded.
    dest_folder (str): Destination folder where the file will be saved.
    file_name (str): Name of the file to be saved.
    """
    try:
        # Create destination folder if it does not exist
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        file_path = os.path.join(dest_folder, file_name)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded {file_name} to {dest_folder}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_name}: {e}")
    except Exception as e:
        print(f"Error saving {file_name} to {dest_folder}: {e}")

def download_datasets():
    """
    Download all datasets specified in the URLs dictionary.
    """
    urls = {
        'IP': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'PU': 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'KSC': 'http://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat',
        'SV': 'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat'
    }

    for key, url in urls.items():
        download_file(url, f'data/{key}', f'{key}.mat')

if __name__ == "__main__":
    # Ensure Git LFS is initialized and tracking .mat files
    # Run the following commands in your terminal:
    # cd HSI_Classification
    # git lfs install
    # git lfs track "*.mat"
    
    download_datasets()
