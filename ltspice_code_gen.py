import os
import subprocess
from spicelib.simulators.ltspice_simulator import LTspice
from langchain_groq import ChatGroq



def generate_netlist(description: str, llm) -> str:
    """
    Преобразует текстовое описание схемы в netlist (.cir) для LTspice с помощью LLM.
    """
    prompt = f"""
    Создай netlist LTspice на основе следующего описания схемы:
    "{description}"
    """
    response = llm.invoke([{"role": "system", "content": "Ты инженер LTspice. Создавай netlist схемы по описанию."},
                           {"role": "user", "content": prompt}])

    return response.content

def save_netlist_to_file(netlist: str, filename: str) -> str:
    """
    Сохраняет netlist в файл .cir
    """
    filepath = f"{filename}.cir"
    with open(filepath, "w") as file:
        file.write(netlist)
    return filepath

def convert_netlist_to_asc(netlist_file: str) -> str:
    # TODO: сделать реальную конвертацию
    """
    Конвертирует netlist в .asc (использует LTspice, если установлен).
    """
    #asc_file = netlist_file.replace(".cir", ".asc")
    #LTspice.convert_to_asc(netlist_file, asc_file)  # Функция конвертации
    return netlist_file

def run_ltspice_simulation(netlist_file: str) -> str:
    """
    Запускает симуляцию схемы в LTspice.
    """
    command = ["ngspice", "-b", netlist_file]
    try:
        subprocess.run(command, check=True)
        return "Simulation completed successfully."
    except subprocess.CalledProcessError as e:
        return f"Simulation failed: {e}"

def process_circuit_description(description: str, llm):
    netlist = generate_netlist(description, llm)
    netlist_file = save_netlist_to_file(netlist, "generated_circuit")
    asc_file = convert_netlist_to_asc(netlist_file)
    simulation_result = run_ltspice_simulation(netlist_file)
    return asc_file, simulation_result

