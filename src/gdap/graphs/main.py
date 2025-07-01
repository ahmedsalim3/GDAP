import argparse

from graph_composer import GraphComposer


def main(disease_id, disease_name):
    composer = GraphComposer(disease_id, disease_name)
    composer.process_all(plot=True)


if __name__ == "__main__":
    hardcode_values = True  # Set to True for hardcoded values, False for command-line arguments

    if hardcode_values:
        disease_id = "EFO_0005741"
        disease_name = "Infectious"
    else:
        parser = argparse.ArgumentParser(description="Process some diseases.")
        parser.add_argument("disease_id", type=str, help="The ID of the disease")
        parser.add_argument("disease_name", type=str, help="The name of the disease")
        args = parser.parse_args()
        disease_id = args.disease_id
        disease_name = args.disease_name

    main(disease_id, disease_name)
