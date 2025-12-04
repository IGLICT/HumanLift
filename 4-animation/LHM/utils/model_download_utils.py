# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-20 14:38:28
# @Function      : auto download class (Modified logic)

import os
import subprocess
import sys

# Need to import FileNotFoundError explicitly if we catch it specifically
from LHM.utils.model_card import HuggingFace_MODEL_CARD, ModelScope_MODEL_CARD

# --- Hugging Face Hub Import ---
package_name='huggingface_hub'
hf_snapshot = None # Initialize
try:
    from huggingface_hub import snapshot_download as hf_snapshot_import
    hf_snapshot = hf_snapshot_import
    print(f"{package_name} imported successfully.")
except ImportError:
    print(f"{package_name} is not installed. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"{package_name} has been installed.")
        from huggingface_hub import snapshot_download as hf_snapshot_import
        hf_snapshot = hf_snapshot_import
    except Exception as e:
        print(f"Failed to install or import {package_name}: {e}")
except Exception as e:
    print(f"An unexpected error occurred during {package_name} import: {e}")

# --- ModelScope Import ---
package_name = "modelscope"
ms_snapshot = None # Initialize
try:
    from modelscope import snapshot_download as ms_snapshot_import
    ms_snapshot = ms_snapshot_import
    print(f"{package_name} imported successfully.")
except ImportError:
    print(f"{package_name} is not installed. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"{package_name} has been installed.")
        from modelscope import snapshot_download as ms_snapshot_import
        ms_snapshot = ms_snapshot_import
    except Exception as e:
        print(f"Failed to install or import {package_name}: {e}")
except Exception as e:
     print(f"An unexpected error occurred during {package_name} import: {e}")


class AutoModelQuery:
    def __init__(self, save_dir='./pretrained_models', hf_kwargs=None, ms_kwargs=None):
        """
        :param save_dir: Base directory to store models.
        """
        # Define base directories clearly
        self.base_save_dir = os.path.abspath(save_dir) # Use absolute path
        self.hf_save_dir = os.path.join(self.base_save_dir, 'huggingface')
        # ModelScope downloads directly into save_dir/repo_id based on original code
        self.ms_save_dir = self.base_save_dir

        # Ensure directories exist
        os.makedirs(self.hf_save_dir, exist_ok=True)
        os.makedirs(self.ms_save_dir, exist_ok=True)

        self.logger = lambda x: "\033[31m{}\033[0m".format(x)

    def _get_max_step_folder(self, current_path):
        """ Helper to find the deepest 'step_*' folder or return current_path if none exist. """
        if not os.path.isdir(current_path):
            return None # Path doesn't exist or isn't a directory

        step_folders = [f for f in os.listdir(current_path) if f.startswith('step_') and os.path.isdir(os.path.join(current_path, f))]
        if not step_folders:
            # Check if the current_path itself contains model files
            # A simple check could be looking for a common config file or pytorch_model.bin
            if os.path.exists(os.path.join(current_path, 'configuration.json')) or \
               os.path.exists(os.path.join(current_path, 'config.json')) or \
               os.path.exists(os.path.join(current_path, 'pytorch_model.bin')):
                 return current_path
            else:
                 # print(f"No step_ folders and no common model files found in {current_path}")
                 return None # No step folders and doesn't look like a model dir
        else:
            # Find the step folder with the highest number
            max_folder_name = max(step_folders, key=lambda x: int(x.split('_')[1]), default=None)
            max_step_path = os.path.join(current_path, max_folder_name) if max_folder_name else None
            # print(f"Found max step folder: {max_step_path}")
            return max_step_path

    def query_huggingface_model(self, model_name, local_only=False):
        """ Queries Hugging Face, optionally checking local cache only. """
        if hf_snapshot is None:
             print(self.logger("Hugging Face Hub library not available."))
             raise ImportError("huggingface_hub not imported")

        if model_name not in HuggingFace_MODEL_CARD:
             raise ValueError(f"Model '{model_name}' not found in HuggingFace_MODEL_CARD.")

        model_repo_id = HuggingFace_MODEL_CARD[model_name]
        action = "Checking cache for" if local_only else "Querying/Downloading"
        print(f"{action} Hugging Face model: {model_repo_id}")

        try:
            # Use local_files_only flag for local check
            # ignore_patterns might be useful to avoid downloading large files during check? No, local_files_only handles it.
            model_path = hf_snapshot(repo_id=model_repo_id, cache_dir=self.hf_save_dir, local_files_only=local_only)
            print(f"Hugging Face model path {'found locally' if local_only else 'obtained'}: {model_path}")
            return model_path
        except FileNotFoundError as e:
             # This specific exception is expected when local_files_only=True and not found
             if local_only:
                 print(f"Hugging Face model {model_repo_id} not found in local cache {self.hf_save_dir}.")
                 raise e # Re-raise specifically for local check failure
             else:
                 print(self.logger(f"Cannot download {model_repo_id} from Hugging Face: {e}"))
                 raise e # Re-raise for download failure
        except Exception as e:
            # Catch other potential errors (network, permissions, etc.)
            log_prefix = "Local check for" if local_only else "Download attempt for"
            print(self.logger(f"{log_prefix} Hugging Face model {model_repo_id} failed: {e}"))
            raise e # Re-raise other exceptions

    def query_modelscope_model(self, model_name, local_only=False):
        """ Queries ModelScope, optionally checking local cache only. """
        if ms_snapshot is None:
             print(self.logger("ModelScope library not available."))
             raise ImportError("modelscope not imported")

        if model_name not in ModelScope_MODEL_CARD:
             raise ValueError(f"Model '{model_name}' not found in ModelScope_MODEL_CARD.")

        model_repo_id = ModelScope_MODEL_CARD[model_name]
        # ModelScope snapshot_download expects repo_id like 'damo/xxx', which might be what's in the card.
        # The directory structure it creates is often based on this repo_id.
        # Let's assume the directory name matches the repo_id directly under ms_save_dir.
        expected_ms_base_path = os.path.join(self.ms_save_dir, model_repo_id)

        action = "Checking cache for" if local_only else "Querying/Downloading"
        print(f"{action} ModelScope model: {model_repo_id}")

        if local_only:
            if os.path.isdir(expected_ms_base_path): # Check if the base directory exists
                model_path = self._get_max_step_folder(expected_ms_base_path)
                if model_path and os.path.isdir(model_path): # Check if path is valid directory
                    print(f"Found local ModelScope model at: {model_path}")
                    return model_path
                else:
                     print(f"ModelScope base path {expected_ms_base_path} exists, but couldn't find valid step folder or model files.")
                     raise FileNotFoundError(f"Local ModelScope model incomplete or invalid structure for {model_repo_id}")
            else:
                print(f"ModelScope model base path {expected_ms_base_path} not found in local cache.")
                raise FileNotFoundError(f"Local ModelScope model not found for {model_repo_id}")
        else:
            # Attempt download if not local_only
            try:
                # ms_snapshot downloads to cache_dir/repo_id structure
                # It returns the path it downloaded to, which should be expected_ms_base_path
                downloaded_path = ms_snapshot(model_repo_id, cache_dir=self.ms_save_dir)
                # downloaded_path is the base path, need to find the step folder
                model_path = self._get_max_step_folder(downloaded_path)
                if model_path and os.path.isdir(model_path):
                     print(f"ModelScope model path obtained: {model_path}")
                     return model_path
                else:
                     print(self.logger(f"ModelScope download for {model_repo_id} seemed successful ({downloaded_path}), but failed to find valid step folder or model files."))
                     raise FileNotFoundError(f"Failed to resolve model path within ModelScope download for {model_repo_id}")
            except Exception as e:
                print(self.logger(f"Download attempt for ModelScope model {model_repo_id} failed: {e}"))
                raise e # Re-raise the caught exception

    def query(self, model_name):
        """
        Query model path with prioritized local checks:
        1. Check local ModelScope cache.
        2. Check local Hugging Face cache.
        3. Try downloading from Hugging Face.
        4. Try downloading from ModelScope.
        """
        print(f"\n--- Querying model: {model_name} ---")
        # Ensure model name is valid in at least one card
        is_in_hf = model_name in HuggingFace_MODEL_CARD
        is_in_ms = model_name in ModelScope_MODEL_CARD
        if not is_in_hf and not is_in_ms:
             raise ValueError(f"Model name '{model_name}' not found in either HuggingFace or ModelScope cards.")
            

        model_path = None

        # 1. Check local ModelScope cache
        if is_in_ms:
            try:
                print("Step 1: Checking local ModelScope cache...")
                model_path = self.query_modelscope_model(model_name, local_only=True)
                if model_path:
                     model_path = model_path +'/' if model_path[-1] !='/' else model_path
                     print(f"Success: Found in local ModelScope cache: {model_path}")
                     return model_path
            except FileNotFoundError:
                 print("Info: Not found in local ModelScope cache.")
            except ImportError:
                 print(self.logger("Warning: ModelScope library not available for local check."))
            except Exception as e:
                 print(self.logger(f"Warning: Error checking local ModelScope cache: {e}"))
        else:
             print("Step 1: Skipping local ModelScope check (not in ModelScope card).")
        

        # 2. Check local Hugging Face cache
        if is_in_hf:
            try:
                print("Step 2: Checking local Hugging Face cache...")
                model_path = self.query_huggingface_model(model_name, local_only=True)
                if model_path:
                     model_path = model_path +'/' if model_path[-1] !='/' else model_path
                     print(f"Success: Found in local Hugging Face cache: {model_path}")
                     return model_path
            except FileNotFoundError:
                 print("Info: Not found in local Hugging Face cache.")
            except ImportError:
                 print(self.logger("Warning: Hugging Face library not available for local check."))
            except Exception as e:
                 print(self.logger(f"Warning: Error checking local Hugging Face cache: {e}"))
        else:
             print("Step 2: Skipping local Hugging Face check (not in HuggingFace card).")

        # If we reach here, the model was not found locally. Try downloading.
        print("Info: Model not found in local caches. Attempting downloads...")

        # 3. Try downloading from Hugging Face
        if is_in_hf:
            try:
                print("Step 3: Attempting download from Hugging Face...")
                model_path = self.query_huggingface_model(model_name, local_only=False)
                if model_path:
                     model_path = model_path +'/' if model_path[-1] !='/' else model_path
                     print(f"Success: Downloaded from Hugging Face: {model_path}")
                     return model_path
            except ImportError:
                 print(self.logger("Warning: Hugging Face library not available, cannot download."))
            except Exception as e:
                 print(f"Info: Hugging Face download failed: {e}. Trying ModelScope next.")
        else:
             print("Step 3: Skipping Hugging Face download (not in HuggingFace card).")

        # 4. Try downloading from ModelScope
        if is_in_ms:
            try:
                print("Step 4: Attempting download from ModelScope...")
                model_path = self.query_modelscope_model(model_name, local_only=False)
                if model_path:
                     model_path = model_path +'/' if model_path[-1] !='/' else model_path
                     print(f"Success: Downloaded from ModelScope: {model_path}")
                     return model_path
            except ImportError:
                 print(self.logger("Warning: ModelScope library not available, cannot download."))
            except Exception as e:
                 print(self.logger(f"Error: ModelScope download failed: {e}."))
        else:
             print("Step 4: Skipping ModelScope download (not in ModelScope card).")

        # If model_path is still None after all attempts
        if model_path is None:
             error_msg = f"Failed to find or download model '{model_name}' from any source (local or remote)."
             print(self.logger(error_msg))
             raise FileNotFoundError(error_msg)

        # Should be unreachable if logic is correct
        return model_path # Should already have trailing slash if needed

# Keep the __main__ block for testing if needed
if __name__ == '__main__':
    # Example usage:
    # Set save_dir to where your models actually are or where you want them
    test_save_dir = '../pretrained_models' # Adjust if needed relative to this script's location
    print(f"Initializing AutoModelQuery with save_dir: {os.path.abspath(test_save_dir)}")
    automodel = AutoModelQuery(save_dir=test_save_dir)

    # Test cases (assuming these models exist in cards)
    test_models = ['LHM-MINI'] # Add other valid model names

    for model_to_test in test_models:

        print(f"\n--- Testing {model_to_test} ---")
        model_path_test = automodel.query(model_to_test)
        print(f"===> Final path for {model_to_test}: {model_path_test}")
        print(f"     Does path exist? {os.path.exists(model_path_test)}")