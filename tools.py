import io
import math
import uuid

import numpy as np
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from telebot import TeleBot


def combine_netlists_tool(llm):
    def combine_netlists(netlists: list[str]) -> str:
        if len(netlists) == 0:
            return "You did not provide any netlist. Please before using this tool wait until you get proper netlists."

        if any(list(map(lambda x: x == "", netlists))):
            return "Some of your netlists are empty. Please before using this tool wait until you get proper netlists."

        netlists_str = ""

        for i, netlist in enumerate(netlists, start=1):
            netlists_str += f"\n* Netlist {i}:\n{netlist}\n"

        messages = [HumanMessage(
            "Соедини netlist'ы в один. У каждого netlist'а есть INPUT_NODE и OUTPUT_NODE, соединяй нетлисты именно в этих нодах. У первого нетлиста должен остаться INPUT_NODE, у последнего должен остаться OUTPUT_NODE. Названия нод, которые будут соединять нетлисты, может быть, например, CONNECTION_NODE_1, CONNECTION_NODE_2 и так далее.\n\n"
            f"Netlist'ы: {netlists_str}\n")]

        response = llm.invoke(messages)

        messages.append(AIMessage(response.content))

        messages.append(HumanMessage("Отправь мне ТОЛЬКО готовый netlist. Мне важно иметь ничего лишнего."))

        response = llm.invoke(messages)

        return response.content

    class CombineNetlistsInput(BaseModel):
        netlists: list[str] = Field(description="A list of netlists. Strings cannot be empty.")

    return StructuredTool.from_function(
        func=combine_netlists,
        args_schema=CombineNetlistsInput,
        name="combine_netlists",
        description="Объединяет несколько netlist'ов в один netlist. Этот инструмент не может вызываться вместе с другими инструментами."
    )


def get_netlist_for_butterworth_lowpass_filter_tool():
    def butterworth_low_pass(f, Z, n):
        id_of_scheme = str(uuid.uuid4())[:4]

        netlist = []

        num_of_Ls = 0
        num_of_Cs = 0

        # Вычисляем коэффициенты Баттерворта g_k
        g = [2 * math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]

        # Вычисляем значения L и C
        for k in range(n):
            if k % 2 == 0:
                # Расчет конденсатора
                C = g[k] / (2 * math.pi * f * Z)
                node = 2 + num_of_Cs
                netlist.append(f"C N_{id_of_scheme}_{node:03} 0 {C}")
                num_of_Cs += 1
            else:
                # Расчет катушки индуктивности
                L = (Z * g[k]) / (2 * math.pi * f)
                node = 2 + num_of_Ls
                nodeplus1 = node + 1
                netlist.append(f"L N_{id_of_scheme}_{node:03} N_{id_of_scheme}_{nodeplus1:03} {L}")
                num_of_Ls += 1

        netlist.extend([
            f"R N_{id_of_scheme}_002 N_{id_of_scheme}_001 {Z}",
            f"R N_{id_of_scheme}_{(num_of_Cs+1):03} N_{id_of_scheme}_{(num_of_Cs+2):03} {Z}",
        ])

        for i in range(len(netlist)):
            netlist[i] = netlist[i].replace(f"N_{id_of_scheme}_001", "INPUT_NODE")
            netlist[i] = netlist[i].replace(f"N_{id_of_scheme}_{(num_of_Cs+2):03}", "OUTPUT_NODE")

        return "\n".join(netlist)

    class GetNetlistInput(BaseModel):
        f: float = Field(description="Частота среза, Гц")
        Z: float = Field(description="Характеристическое сопротивление, Ом")
        n: int = Field(description="Количество реактивных элементов (порядок фильтра)")

    return StructuredTool.from_function(
        name="get_netlist_for_butterworth_lowpass_filter",
        description="Возвращает netlist для фильтра низких частот Баттерворта.",
        args_schema=GetNetlistInput,
        func=butterworth_low_pass
    )


def get_netlist_for_diode_bridge_tool():
    def diode_bridge():
        netlist = [
            f"D INPUT_NODE OUTPUT_NODE D",
            f"D 0 V_MINUS_NODE D",
            f"D V_MINUS_NODE OUTPUT_NODE D",
            f"D 0 INPUT_NODE D",
            f"C OUTPUT_NODE 0 100u",
            ".model D D",
            ".lib C:\\users\\vyacheslav\\AppData\\Local\\LTspice\\lib\\cmp\\standard.dio"
        ]

        return "\n".join(netlist)

    return Tool(
        name="get_netlist_for_diode_bridge",
        description="Возвращает netlist диодного моста.",
        func=lambda _: diode_bridge()
    )


def get_netlist_for_bessel_lowpass_filter_tool():
    def bessel_polynomial_coeffs(n):
        coeffs = np.zeros(n + 1)
        for k in range(n + 1):
            coeffs[k] = math.factorial(2 * n - k) / (2 ** (n - k) * math.factorial(k) * math.factorial(n - k))
        return coeffs

    def bessel_low_pass(order, cutoff_freq, impedance):
        if order < 1:
            raise ValueError("Порядок фильтра должен быть больше или равен 1")

        # Вычисляем коэффициенты полиномов Бесселя для текущего и предыдущего порядков
        a_n = bessel_polynomial_coeffs(order)

        # Вычисляем g-значения по рекуррентной формуле
        g_values = []
        for k in range(1, order + 1):
            if k == 1:
                g = a_n[0] / a_n[1]
            else:
                numerator = 4 * (2 * (order - k) + 1) * a_n[k - 2] * a_n[k]
                denominator = a_n[k - 1] ** 2
                g = numerator / denominator
            g_values.append(g)

        id_of_scheme = str(uuid.uuid4())[:4]
        netlist = []

        num_of_Ls = 0
        num_of_Cs = 0

        # Денормировка элементов под заданные частоту и импеданс
        for i, g in enumerate(g_values):
            if (i) % 2 == 1:  # Нечетные элементы - индуктивности
                num_of_Ls += 1
                L = (g * impedance) / (2 * np.pi * cutoff_freq)
                node = 1 + num_of_Ls
                nodeplus1 = node + 1
                netlist.append(f"L N_{id_of_scheme}_{node:03} N_{id_of_scheme}_{nodeplus1:03} {L}")
                # L_values.append(float(L))
            else:  # Четные элементы - емкости
                num_of_Cs += 1
                C = g / (impedance * 2 * np.pi * cutoff_freq)
                node = 1 + num_of_Cs
                netlist.append(f"C N_{id_of_scheme}_{node:03} 0 {C}")
                # C_values.append(float(C))

        netlist.extend([
            f"R N_{id_of_scheme}_002 N_{id_of_scheme}_001 {impedance}",
            f"R N_{id_of_scheme}_{(num_of_Cs + 1):03} N_{id_of_scheme}_{(num_of_Cs + 2):03} {impedance}",
        ])

        for i in range(len(netlist)):
            netlist[i] = netlist[i].replace(f"N_{id_of_scheme}_001", "INPUT_NODE")
            netlist[i] = netlist[i].replace(f"N_{id_of_scheme}_{(num_of_Cs + 2):03}", "OUTPUT_NODE")

        return "\n".join(netlist)

    class GetNetlistInput(BaseModel):
        order: int = Field(description="Количество реактивных элементов (порядок фильтра)")
        cutoff_freq: float = Field(description="Частота среза, Гц")
        impedance: float = Field(description="Характеристическое сопротивление, Ом")

    return StructuredTool.from_function(
        name="get_netlist_for_bessel_lowpass_filter",
        description="Возвращает netlist для фильтра низких частот Бесселя.",
        args_schema=GetNetlistInput,
        func=bessel_low_pass
    )


def get_netlist_for_dc_dc_boost_converter_tool():
    def dc_dc_boost_converter(Fs, D):
        id_of_scheme = str(uuid.uuid4())[:4]

        netlist = [
            f"L INPUT_NODE N_{id_of_scheme}_001 10m",
            f"S 0 N_{id_of_scheme}_001 N_{id_of_scheme}_002 N_{id_of_scheme}_003 MOSFET",
            f".model MOSFET SW(Ron=1u Roff=100Meg Vt=90)",
            f"V N_{id_of_scheme}_002 N_{id_of_scheme}_003 PULSE(0 100 0 1n 1n {D/Fs} {1/Fs} 1Meg)",
            f"D N_{id_of_scheme}_001 OUTPUT_NODE D",
            ".model D D",
            ".lib C:\\users\\vyacheslav\\AppData\\Local\\LTspice\\lib\\cmp\\standard.dio",
            f"C OUTPUT_NODE 0 10u",
        ]

        return "\n".join(netlist)

    class GetNetlistInput(BaseModel):
        Fs: float = Field(description="Switching frequency, Hz", default=16000.0)
        D: float = Field(description="Скважность", default=0.5)

    return StructuredTool.from_function(
        name="get_netlist_for_dc_dc_boost_converter",
        description="Возвращает netlist для преобразователя повышающего типа (boost converter).",
        args_schema=GetNetlistInput,
        func=dc_dc_boost_converter
    )


def get_netlist_for_transmission_line_tool():
    def transmission_line(R1, L, C, R2):
        id_of_scheme = str(uuid.uuid4())[:4]

        netlist = [
            f"R N_{id_of_scheme}_1 INPUT_NODE {R1}",
            f"L N_{id_of_scheme}_001 N_{id_of_scheme}_002 {L}",
            f"C N_{id_of_scheme}_002 OUTPUT_NODE {C}",
            f"R N_{id_of_scheme}_002 OUTPUT_NODE {R2}"
        ]

        return "\n".join(netlist)

    class GetNetlistInput(BaseModel):
        R1: float = Field(description="Сопротивление первого резистора, Ом")
        L: float = Field(description="Индуктивность катушки, Гн")
        C: float = Field(description="Ёмкость конденсатора, Ф")
        R2: float = Field(description="Проводимость, Ом")

    return StructuredTool.from_function(
        name="get_netlist_transmission_line",
        description="Возвращает netlist для одной секции линии передач.",
        args_schema=GetNetlistInput,
        func=transmission_line
    )


def finalize_netlist_tool():
    def finalize_netlist(netlist, v, analysis):
        if netlist == "":
            return "Provided netlist is empty."

        old_netlist = netlist.split("\n")
        new_netlist = [
            "* Generated by @LTSpice_bot"
        ]

        numbered_components = list("RCLVDS")
        nums = {"V": 0, "R": 0}

        flag_for_v_minus = False

        for line in old_netlist:
            if line == "":
                continue
            if "V_MINUS_NODE" in line:
                flag_for_v_minus = True
            if line[0] in numbered_components:
                if line[0] not in nums:
                    nums[line[0]] = 0
                nums[line[0]] = nums[line[0]] + 1
                newline = f"{line[0]}{nums[line[0]]} {line[2:]}"
            else:
                newline = line
            #newline = newline.replace("OUTPUT_NODE", "0")
            new_netlist.append(newline)

        if v == "SINE":
            new_netlist.append(f"V{nums["V"]+1} INPUT_NODE {"V_MINUS_NODE" if flag_for_v_minus else "0"} SINE(1 1 100000) AC 1")
        else:
            new_netlist.append(f"V{nums["V"]+1} INPUT_NODE {"V_MINUS_NODE" if flag_for_v_minus else "0"} 1")

        new_netlist.append(f"R{nums["R"] + 1} OUTPUT_NODE 0 10")

        if analysis == "ac":
            new_netlist.append(f".ac dec 1000 10 100000")
        else:
            new_netlist.append(f".tran 10m")

        new_netlist.extend([
            f".backanno",
            f".end"
        ])

        return "\n".join(new_netlist)

    class GetNetlistInput(BaseModel):
        netlist: str = Field(description="Netlist, который необходимо довести до конца.")
        v: str = Field(description="Тип источника напряжения. Один из ['DC', 'SINE']")
        analysis: str = Field(description="Применяемый анализ. Один из ['transient', 'ac']")

    return StructuredTool.from_function(
        name="finalize_netlist",
        description="Поскольку все инструменты получения или объединения нетлистов возвращают netlistы, в которых у элементов не прописан порядковый номер, этот инструмент решает данную проблему и предоставляет netlist, полностью готовый к работе.",
        args_schema=GetNetlistInput,
        func=finalize_netlist
    )


def send_netlist_to_user_tool(chat_id: int, bot: TeleBot):

    def send_netlist_to_user(chat_id: int, bot: TeleBot, netlist: str):
        netlist_str = netlist
        file_obj = io.BytesIO(netlist_str.encode("utf-8"))
        file_obj.name = "circuit.net"
        bot.send_document(chat_id, file_obj)

        return "Successfully sent netlist contents to the user!"

    return Tool(
        name="send_netlist_to_user",
        description="Sends netlist to the user.",
        func=lambda asc_file_content: send_netlist_to_user(chat_id, bot, asc_file_content)
    )