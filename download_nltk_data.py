# download_nltk_data.py
import nltk
import sys
import os

def download_nltk_resources(download_dir_base):
    print(f"NLTK: Attempting to download resources to base directory: {download_dir_base}")
    
    # NLTK expects data to be in subdirectories like 'corpora', 'tokenizers' etc.
    # nltk.download() manages these subdirectories automatically when download_dir is a top-level NLTK path.
    # Ensure the base download_dir itself is known to NLTK for this script's execution.
    if download_dir_base not in nltk.data.path:
        nltk.data.path.append(download_dir_base)
    print(f"NLTK: nltk.data.path for download process: {nltk.data.path}")

    datasets_to_download = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    all_successful = True

    for dataset_name in datasets_to_download:
        try:
            print(f"NLTK: Downloading '{dataset_name}' to '{download_dir_base}'...")
            # nltk.download will place it into the correct subdirectory (e.g. corpora/wordnet.zip)
            # within download_dir_base.
            if not nltk.download(dataset_name, download_dir=download_dir_base, quiet=False, raise_on_error=True):
                print(f"NLTK: nltk.download call returned False for '{dataset_name}'. This usually indicates an issue.")
                all_successful = False
            else:
                print(f"NLTK: Successfully downloaded or verified '{dataset_name}'.")
                
                # Verification step:
                # Construct the resource name NLTK uses for find()
                resource_path_in_nltk = ""
                if dataset_name == "punkt":
                    resource_path_in_nltk = f"tokenizers/{dataset_name}"
                elif dataset_name in ["stopwords", "wordnet", "omw-1.4"]:
                    resource_path_in_nltk = f"corpora/{dataset_name}"
                else:
                    print(f"NLTK: Unknown dataset type for '{dataset_name}', skipping find verification.")
                    continue
                
                try:
                    # Use nltk.data.find(), ensuring it searches within the specified download_dir_base
                    found_path = nltk.data.find(resource_path_in_nltk, paths=[download_dir_base])
                    print(f"NLTK: Verified '{dataset_name}' exists at '{found_path}' within '{download_dir_base}'.")
                except LookupError:
                    print(f"NLTK: CRITICAL VERIFICATION FAILED for '{dataset_name}'. Could not find '{resource_path_in_nltk}' within '{download_dir_base}'.")
                    all_successful = False
        except Exception as e:
            print(f"NLTK: Error downloading or verifying '{dataset_name}': {e}")
            all_successful = False
            
    if not all_successful:
        print("NLTK: One or more resources failed to download/verify. Exiting with error.")
        sys.exit(1) # Fail the script, which should fail the Docker build step
    
    print(f"NLTK: All specified resources downloaded and verified successfully in '{download_dir_base}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_nltk_data.py <target_directory_for_nltk_data>")
        print("This directory will be used as a root for NLTK data (e.g., corpora, tokenizers subdirs will be created inside it).")
        sys.exit(1)
    
    nltk_data_target_directory = sys.argv[1]
    # Create the directory if it doesn't exist. nltk.download might do this, but being explicit is safer.
    os.makedirs(nltk_data_target_directory, exist_ok=True)
    
    download_nltk_resources(nltk_data_target_directory)