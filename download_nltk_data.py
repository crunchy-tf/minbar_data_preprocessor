# download_nltk_data.py
import nltk
import sys
import os
import zipfile
import shutil # For removing files/dirs

def list_directory_contents(path, prefix="NLTK_DEBUG:"):
    print(f"{prefix} Listing contents of '{path}':")
    if os.path.exists(path) and os.path.isdir(path):
        for item in os.listdir(path):
            print(f"{prefix}   - {item}")
    elif os.path.exists(path):
        print(f"{prefix}   '{path}' exists but is not a directory (it's a file).")
    else:
        print(f"{prefix}   Path '{path}' does not exist.")

def download_nltk_resources(download_dir_base):
    print(f"NLTK: Attempting to download resources to base directory: {download_dir_base}")
    
    if download_dir_base not in nltk.data.path:
        nltk.data.path.append(download_dir_base)
    print(f"NLTK: nltk.data.path for download process: {nltk.data.path}")

    datasets_to_download = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    all_successful = True

    for dataset_name in datasets_to_download:
        print(f"\nNLTK: --- Processing dataset: {dataset_name} ---")
        resource_path_in_nltk = "" # e.g. "corpora/wordnet" for nltk.data.find()
        expected_zip_name = f"{dataset_name}.zip" # e.g. "wordnet.zip"
        # expected_unzipped_name_in_subdir is the dir name inside corpora/tokenizers, e.g. "wordnet"
        expected_unzipped_name_in_subdir = dataset_name 

        if dataset_name == "punkt":
            resource_path_in_nltk = f"tokenizers/{dataset_name}"
            # subdir where zip/unzipped content is expected, relative to download_dir_base
            expected_item_subdir_rel = "tokenizers" 
        elif dataset_name in ["stopwords", "wordnet", "omw-1.4"]:
            resource_path_in_nltk = f"corpora/{dataset_name}"
            expected_item_subdir_rel = "corpora"
        else:
            print(f"NLTK: Unknown dataset type for '{dataset_name}', skipping.")
            continue
        
        # Absolute path to the subdirectory (e.g. /usr/local/share/nltk_data/corpora)
        full_expected_item_subdir = os.path.join(download_dir_base, expected_item_subdir_rel)
        os.makedirs(full_expected_item_subdir, exist_ok=True)

        # Absolute path to where the unzipped directory should be (e.g. .../corpora/wordnet/)
        full_expected_unzipped_dir_path = os.path.join(full_expected_item_subdir, expected_unzipped_name_in_subdir)
        # Absolute path to where the zip file should be (e.g. .../corpora/wordnet.zip)
        full_expected_zip_file_path = os.path.join(full_expected_item_subdir, expected_zip_name)

        # Clean slate: if an (empty/incomplete) unzipped dir exists from a previous attempt, remove it.
        if os.path.isdir(full_expected_unzipped_dir_path):
            print(f"NLTK_DEBUG: Pre-existing directory found at '{full_expected_unzipped_dir_path}'. Removing it for a clean download/unzip attempt.")
            try:
                shutil.rmtree(full_expected_unzipped_dir_path)
                print(f"NLTK_DEBUG: Successfully removed pre-existing directory '{full_expected_unzipped_dir_path}'.")
            except Exception as e_rm:
                print(f"NLTK_DEBUG: Failed to remove pre-existing directory '{full_expected_unzipped_dir_path}': {e_rm}. Proceeding cautiously.")
        elif os.path.exists(full_expected_unzipped_dir_path): # Exists but not a dir (e.g. a file)
             print(f"NLTK_DEBUG: Pre-existing non-directory item found at '{full_expected_unzipped_dir_path}'. Removing it.")
             try:
                os.remove(full_expected_unzipped_dir_path)
             except Exception as e_rm_file:
                print(f"NLTK_DEBUG: Failed to remove pre-existing file '{full_expected_unzipped_dir_path}': {e_rm_file}")


        try:
            print(f"NLTK: Calling nltk.download for '{dataset_name}' to '{download_dir_base}' (raise_on_error=True)...")
            # NLTK's downloader should place the zip into the correct subdir (corpora/tokenizers)
            nltk.download(dataset_name, download_dir=download_dir_base, quiet=False, raise_on_error=True)
            print(f"NLTK: nltk.download for '{dataset_name}' call completed without raising an error.")

            # --- Post-download verification and manual unzip if needed ---
            list_directory_contents(full_expected_item_subdir, prefix="NLTK_POST_DOWNLOAD_LS:")
            
            # Check if the unzipped directory now exists and is valid (not empty)
            unzipped_dir_is_valid = (
                os.path.exists(full_expected_unzipped_dir_path) and 
                os.path.isdir(full_expected_unzipped_dir_path) and 
                bool(os.listdir(full_expected_unzipped_dir_path)) # Check if dir is not empty
            )

            if not unzipped_dir_is_valid:
                print(f"NLTK_DEBUG: Unzipped directory '{full_expected_unzipped_dir_path}' not found, not a dir, or is empty after nltk.download.")
                
                if os.path.exists(full_expected_zip_file_path):
                    print(f"NLTK_DEBUG: Zip file '{full_expected_zip_file_path}' found. Attempting manual unzip.")
                    try:
                        # Ensure target unzipped dir path is clear if it exists as an empty dir or wrong file type
                        if os.path.isdir(full_expected_unzipped_dir_path): # If it's an empty dir
                             shutil.rmtree(full_expected_unzipped_dir_path)
                        elif os.path.exists(full_expected_unzipped_dir_path): # If it's a file
                            os.remove(full_expected_unzipped_dir_path)

                        with zipfile.ZipFile(full_expected_zip_file_path, 'r') as zip_ref:
                            zip_ref.extractall(full_expected_item_subdir) 
                        print(f"NLTK_DEBUG: Successfully unzipped '{full_expected_zip_file_path}' to '{full_expected_item_subdir}'.")
                        list_directory_contents(full_expected_item_subdir, prefix="NLTK_POST_MANUAL_UNZIP_LS:")
                        
                        # Verify again that the unzipped directory exists and is not empty
                        unzipped_dir_is_valid_after_manual_unzip = (
                            os.path.exists(full_expected_unzipped_dir_path) and 
                            os.path.isdir(full_expected_unzipped_dir_path) and 
                            bool(os.listdir(full_expected_unzipped_dir_path))
                        )
                        if not unzipped_dir_is_valid_after_manual_unzip:
                             print(f"NLTK_ERROR: Manual unzip attempted, but '{full_expected_unzipped_dir_path}' is still not a valid, non-empty directory.")
                             all_successful = False; continue # to next dataset
                        else:
                             print(f"NLTK_DEBUG: Manual unzip successful, '{full_expected_unzipped_dir_path}' is now valid.")

                    except Exception as e_unzip:
                        print(f"NLTK_ERROR: Manual unzipping failed for '{full_expected_zip_file_path}': {e_unzip}")
                        all_successful = False; continue # to next dataset
                else: # Zip file not found
                    print(f"NLTK_ERROR: Zip file '{full_expected_zip_file_path}' also not found. nltk.download failed to place the zip correctly or it was already unzipped and removed by NLTK (unlikely for wordnet/omw based on logs).")
                    # This case could happen if NLTK *did* unzip and remove the zip, but the unzipped dir is somehow invalid.
                    # However, the initial logs suggest NLTK isn't even *attempting* to unzip wordnet/omw.
                    all_successful = False; continue # to next dataset
            else: # Unzipped directory was valid after nltk.download call
                print(f"NLTK_DEBUG: Unzipped directory '{full_expected_unzipped_dir_path}' found valid after nltk.download call (likely unzipped by NLTK).")


            # --- Final verification using nltk.data.find() ---
            print(f"NLTK: Final verification for '{dataset_name}' (searched as '{resource_path_in_nltk}') using nltk.data.find()...")
            try:
                # nltk.data.path already includes download_dir_base
                found_path = nltk.data.find(resource_path_in_nltk) 
                print(f"NLTK: VERIFIED '{dataset_name}' (found as '{resource_path_in_nltk}') at '{found_path}'.")
            except LookupError:
                print(f"NLTK_ERROR: CRITICAL VERIFICATION FAILED for '{dataset_name}'. Could not find '{resource_path_in_nltk}'.")
                list_directory_contents(full_expected_item_subdir, prefix="NLTK_FIND_FAIL_LS:")
                all_successful = False
        
        except Exception as e: 
            print(f"NLTK_ERROR: General error during processing of '{dataset_name}': {e}")
            all_successful = False
            
    if not all_successful:
        print("\nNLTK_ERROR: One or more resources failed to download/verify. Exiting with error.")
        sys.exit(1)
    
    print(f"\nNLTK: All specified resources downloaded and verified successfully in '{download_dir_base}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_nltk_data.py <target_directory_for_nltk_data>")
        print("This directory will be used as a root for NLTK data (e.g., corpora, tokenizers subdirs will be created inside it).")
        sys.exit(1)
    
    nltk_data_target_dir = sys.argv[1]
    os.makedirs(nltk_data_target_dir, exist_ok=True) # Ensure NLTK_DATA base dir exists
    
    download_nltk_resources(nltk_data_target_dir)