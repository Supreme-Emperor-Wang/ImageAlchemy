from neurons.shared import add_repo_to_path

if __name__ == "__main__":
    ### Add the base repository to the path so the validator can access it
    add_repo_to_path()

    ### Import StableValidator after fixing paths
    from validator import StableValidator

    StableValidator().run()
