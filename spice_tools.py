from langchain_core.tools import tool
from spicelib import SimRunner, SpiceEditor, AscEditor
from spicelib.simulators.ltspice_simulator import LTspice


@tool
def spice_tool(asc: str) -> str:
    """Run circuit simulation with LTspice using .asc file contents."""

    asc_file: str = "temp/circuit.asc"

    with open(asc_file, 'w') as file:
        file.write(asc)

    runner = SimRunner(simulator=LTspice, output_folder='temp')
    netlist = AscEditor(asc_file)

    runner.run(netlist, exe_log=True)

    raws = []

    for raw, log in runner:
        #print(raw)
        print("Raw file: %s, Log file: %s" % (raw, log))
        raws.append(raw)


    return open(raws[0], "r").read()

    """

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
        return "Simulation completed, but output data was not found."""