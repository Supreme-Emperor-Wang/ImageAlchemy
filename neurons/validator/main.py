from neurons.shared import add_repo_to_path
from validator import StableValidator

if __name__ == "__main__":
    ### Add the base repository to the path so the validator can access it
    add_repo_to_path()

    StableValidator().run()
