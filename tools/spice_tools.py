import os
import subprocess

from langchain_core.tools import tool


@tool
def spice_tool(circuit_description: str) -> str:
    """Run circuit simulation using ngspice."""
    netlist_file: str = "generated_circuit.cir"

    with open(netlist_file, 'w') as file:
        file.write(circuit_description)

    command = ["ngspice", "-b", netlist_file]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return f"Simulation failed: {e}"

    raw_file = netlist_file.replace('.cir', '_output.csv')

    if os.path.exists(raw_file):
        with open(raw_file, 'r') as file:
            table_data = file.read()
        os.remove(raw_file)  # Удаляем временный файл после чтения
        return table_data
    else:
        return "Simulation completed, but output data was not found."