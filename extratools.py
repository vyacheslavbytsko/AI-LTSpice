import base64
import io
import math
import uuid
import scipy.signal as signal
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from telebot import TeleBot

from misc import get_default_mapping, round16, get_pin_positions, text_to_base64, base64_to_text


def get_netlist_b64_for_butterworth_lowpass_filter_tool():
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

        return text_to_base64("\n".join(netlist))

    class GetNetlistInput(BaseModel):
        f: float = Field(description="Частота среза, Гц")
        Z: float = Field(description="Характеристическое сопротивление, Ом")
        n: int = Field(description="Количество реактивных элементов (порядок фильтра)")

    return StructuredTool.from_function(
        name="get_netlist_b64_for_butterworth_lowpass_filter",
        description="Возвращает base64 репрезентацию netlist для фильтра низких частот Баттерворта.",
        args_schema=GetNetlistInput,
        func=butterworth_low_pass
    )

def get_netlist_b64_for_bessel_lowpass_filter_tool():
    def bessel_low_pass(f, Z, n):
        id_of_scheme = str(uuid.uuid4())[:4]

        netlist = []

        num_of_Ls = 0
        num_of_Cs = 0

        # Вычисляем коэффициенты Бесселя g_k
        z, p, k = signal.besselap(n, norm='delay')
        b, a = signal.zpk2tf(z, p, k)

        g = [a[i] / b[i] for i in range(n)]

        # Вычисляем значения L и C
        for k in range(n):
            if k % 2 == 0:
                # Расчет конденсатора
                C = g[k] / (2 * np.pi * f * Z)
                node = 2 + num_of_Cs
                netlist.append(f"C N_{id_of_scheme}_{node:03} 0 {C}")
                num_of_Cs += 1
            else:
                # Расчет катушки индуктивности
                L = (Z * g[k]) / (2 * np.pi * f)
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

        return text_to_base64("\n".join(netlist))

    class GetNetlistInput(BaseModel):
        f: float = Field(description="Частота среза, Гц")
        Z: float = Field(description="Характеристическое сопротивление, Ом")
        n: int = Field(description="Количество реактивных элементов (порядок фильтра)")

    return StructuredTool.from_function(
        name="get_netlist_b64_for_butterworth_lowpass_filter",
        description="Возвращает base64 репрезентацию netlist для фильтра низких частот Бесселя.",
        args_schema=GetNetlistInput,
        func=bessel_low_pass
    )
