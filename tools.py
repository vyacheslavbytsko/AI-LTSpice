import base64
import io
import math

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from telebot import TeleBot

from misc import get_default_mapping, round16, get_pin_positions, text_to_base64


def description_to_simple_circuits_descriptions_tool(llm: ChatGroq, known_circuits: str):
    def process_description(query, known_circuits):
        messages = [
            HumanMessage(
                f"В моей библиотеке схем есть такие схемы:"
                f"\n{known_circuits}\nМой друг же хочет создать "
                f"сложную схему на основе следующего описания: "
                f"\"{query}\". Помоги мне выбрать минимальное "
                f"количество схем, при помощи которых можно будет "
                f"составить схему, которую хочет мой друг. Это "
                f"количество даже может быть равно единице! Мне "
                f"нужно реально минимальное количество схем из "
                f"библиотеки, чтобы можно было составить схему "
                f"по описанию."
            )
        ]

        response = llm.invoke(messages)

        messages.append(AIMessage(response.content))

        messages.append(HumanMessage("Выведи мне ТОЛЬКО список нужных схем. Ничего лишнего, даже не надо писать \"Вот список:\""))

        response = llm.invoke(messages)

        return f"Here's the list of simple circuits:\n{response.content}\nUse them to gather descriptions about all of them. They are NOT filenames."

    return Tool(
        name="description_to_simple_circuits",
        description="Разбивает описание сложной схемы на несколько простых описаний схем для последующего поиска.",
        func=lambda query: process_description(query, known_circuits)
    )


def simple_circuit_description_to_descriptions_and_filenames_tool(vector_store: FAISS):
    def get_relevant_filenames_and_descriptions(query):
        retriever = vector_store.as_retriever(
            search_type="similarity",
            k=3,
            score_threshold=None,
        )

        relevant_descriptions = retriever.invoke(query)

        relevant_descriptions_text = ""
        for relevant_description in relevant_descriptions:
            relevant_descriptions_text += f"Filename \"{relevant_description.metadata["description_filename"]}\": \"{relevant_description.page_content}\"\n\n"

        return f"Here are three filenames and descriptions which fit the best to the description you provided:\n\n{relevant_descriptions_text}Use them to choose filename to get netlist of the circuit. Be aware - those parts were found by using vector store, so if you don't like my answer, reask user for more specific description and run this tool again. DO NOT make user choose filename. Choose by yourself. Choose by yourself."

    return Tool(
        name="simple_circuit_description_to_descriptions_and_filenames",
        description="RAG tool: Given the simple circuit description, searches and returns three partial descriptions (and their filenames) that fit best to the description of simple circuit. Filenames can be used to gather full descriptions.",
        func=lambda query: get_relevant_filenames_and_descriptions(query)
    )


def filename_to_full_circuit_description_tool():
    def get_content(query):
        return f"Here is the full description of circuit:\n\n\"{open(query).read()}\""

    return Tool(
        name="filename_to_full_circuit_description",
        description="Returns the full description of circuit based on the filename of description.",
        func=lambda query: get_content(query)
    )


def filename_to_netlist_b64_tool():
    def filename_to_netlist_b64(query):
        try:
            return (f"Here's the base64 representation of netlist: \""
                    f"{base64.b64encode(
                        open(query.removesuffix(".desc.txt") + ".net")
                        .read()
                        .encode("utf-8")
                    ).decode("utf-8")}\"")
        except FileNotFoundError:
            return "We didn't find this file. Maybe you should try to use this filename to find description, and then gather real filename?"

    return Tool(
        name="filename_to_netlist_b64",
        description="Returns the base64 representation of netlist of circuit based on the filename of description.",
        func=lambda query: filename_to_netlist_b64(query)
    )

def combine_netlists_b64s_tool(llm):
    def combine_netlists(description: str, netlists: list[str]) -> str:
        netlists_str = ""

        for i, netlist in enumerate(netlists, start=1):
            netlists_str += f"\n* Netlist {i}:\n{base64.b64decode(netlist.encode("utf-8")).decode("utf-8")}\n"

        messages = [HumanMessage(
            "Объедини следующие netlist'ы в один, основываясь на их содержимом и общем описании. "
            "Убедись, что все компоненты подключены корректно и схема соответствует описанию.\n\n"
            f"Описание схемы: {description}\n\nNetlist'ы: {netlists_str}\n")]

        response = llm.invoke(messages)

        messages.append(AIMessage(response.content))

        messages.append(HumanMessage("Отправь мне ТОЛЬКО готовый netlist. Мне важно иметь ничего лишнего."))

        response = llm.invoke(messages)

        print("ГОТОВЫЙ NETLIST:", response.content)

        return base64.b64encode(response.content.encode("utf-8")).decode("utf-8")

    class CombineNetlistsInput(BaseModel):
        description: str = Field(description="Initial description of circuit user wants.")
        netlists: list[str] = Field(description="A list of base64 representations of netlists' contents (full contents!!!)")

    return StructuredTool.from_function(
        func=combine_netlists,
        args_schema=CombineNetlistsInput,
        name="combine_netlists",
        description="Объединяет несколько base64 репрезентаций netlist'ов в один combined netlist (его base64 репрезентацию) на основе их содержимого и описания схемы.\n"
    )


def apply_parameters_to_netlist_b64_tool(llm):
    def apply_parameters_to_netlist_b64(netlist, description):
        netlist_str = base64.b64decode(netlist.encode("utf-8")).decode("utf-8")

        messages = [
            HumanMessage(
                f"Мой друг составил netlist:\n\n"
                f"{netlist_str}\n\nПожалуйста, проверь, "
                f"что он соответствует описанию: \"{description}\". "
                f"Если он не соответствует описанию, измени netlist "
                f"и выведи новый. Если соответствует - просто "
                f"заново выведи netlist."
            )
        ]

        response = llm.invoke(messages)

        messages.append(AIMessage(response.content))

        messages.append(
            HumanMessage("Выведи ТОЛЬКО готовый netlist. Не должно быть ничего лишнего."))

        response = llm.invoke(messages)

        return base64.b64encode(response.content.encode("utf-8")).decode("utf-8")

    class ApplyParametersInput(BaseModel):
        netlist: str = Field(description="A base64 representation of combined netlist")
        description: str = Field(description="Initial description of circuit user wants")

    return StructuredTool.from_function(
        func=apply_parameters_to_netlist_b64,
        args_schema=ApplyParametersInput,
        name="apply_parameters_to_netlist_b64",
        description="Given the base64 representation of combined netlist (only one!), applies parameters from initial description and returns base64 representation of final netlist."
    )


def netlist_b64_to_asc_tool(sheet_size: str = "SHEET 1 800 600",
                        directive_text_coords: str = "TEXT 450 300 Left 2",
                        comment_text_default: str = "TEXT 80 -250 Left 4"):
    def net_to_asc(net_content: str,
                   sheet_size: str = "SHEET 1 800 600",
                   directive_text_coords: str = "TEXT 450 300 Left 2",
                   comment_text_default: str = "TEXT 80 -250 Left 4") -> str:
        """
        Универсально преобразует содержимое .net файла в .asc файл.

        Для каждого описания компонента:
          • Определяется mapping по имени (см. get_default_mapping).
          • Компоненты располагаются на сетке с начальным положением (comp_start_x, comp_start_y) и шагом, кратным 16.
          • Пины компонента вычисляются относительно его центра, координаты округляются до кратных 16.
          • Узлы (флаги) собираются из соединений, располагаются на отдельной сетке.
          • Для каждого пина компонента генерируется провод (WIRE), соединяющий его с соответствующим флагом.

        Возвращается содержимое .asc файла без дополнительных строк (.backanno, .end).
        """
        # Настройки сетки для компонентов (все значения кратны 16)
        comp_start_x = 48
        comp_start_y = 48
        comp_spacing_x = 144
        comp_spacing_y = 96
        comp_columns = 5

        # Настройки сетки для флагов (узлов)
        flag_start_x = -320
        flag_start_y = 0
        flag_spacing_x = 96
        flag_spacing_y = 48
        flag_columns = 4

        asc_lines = []
        asc_lines.append("Version 4")
        asc_lines.append(sheet_size)

        # Собираем уникальные узлы и список соединений (для проводов)
        node_set = set()
        connection_list = []  # Каждый элемент: (node, inst_name, pin_index, pin_x, pin_y)
        comp_counter = 0

        comp_blocks = []
        text_blocks = []

        for line in net_content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Пропускаем заголовки (например, путь, версия)
            if line.startswith("*") and ("LTspice" in line or ".asc" in line):
                continue
            # Обработка комментариев – преобразуем в текстовый блок
            if line.startswith("*"):
                comment = line[1:].strip()
                text_blocks.append(f"{comment_text_default} ;{comment}")
                continue
            # Обработка директив (.ac, .tran и т.п.) – в текстовый блок
            if line.startswith(".") and not (line.startswith(".backanno") or line.startswith(".end")):
                text_blocks.append(f"{directive_text_coords} !{line}")
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            inst_name = parts[0]
            # Нормализуем имя для инстанций, начинающихся с "X"
            if inst_name.startswith("X"):
                inst_name = inst_name.replace("§", "")

            mapping = get_default_mapping(inst_name, line)
            pin_count = mapping.get("pin_count", 2)

            # Предполагаем, что первые pin_count токенов после имени – это узлы
            node_list = parts[1:1 + pin_count]
            for node in node_list:
                if node != "0":
                    node_set.add(node)

            # Определяем позицию компонента по сетке
            col = comp_counter % comp_columns
            row = comp_counter // comp_columns
            cx = round16(comp_start_x + col * comp_spacing_x)
            cy = round16(comp_start_y + row * comp_spacing_y)
            comp_counter += 1

            coords = f"{cx} {cy} R0"
            comp_blocks.append(f"SYMBOL {mapping['symbol']} {coords}")
            for win in mapping.get("windows", []):
                comp_blocks.append(win)
            comp_blocks.append(f"SYMATTR InstName {inst_name}")
            comp_value = " ".join(parts[1 + pin_count:])
            comp_blocks.append(f"SYMATTR Value {comp_value}")
            if "extra" in mapping:
                comp_blocks.append(mapping["extra"])

            # Вычисляем позиции пинов компонента и сохраняем соединения
            pin_positions = get_pin_positions((cx, cy), pin_count)
            for i, node in enumerate(node_list):
                if i < len(pin_positions):
                    pin_x, pin_y = pin_positions[i]
                    connection_list.append((node, inst_name, i, pin_x, pin_y))

        # Генерация флагов для каждого уникального узла
        auto_flag_blocks = []
        flag_coord = {}
        for idx, node in enumerate(sorted(node_set)):
            col = idx % flag_columns
            row = idx // flag_columns
            fx = round16(flag_start_x + col * flag_spacing_x)
            fy = round16(flag_start_y + row * flag_spacing_y)
            flag_coord[node] = (fx, fy)
            auto_flag_blocks.append(f"FLAG {fx} {fy} {node}")

        # Генерация проводов, соединяющих пины с флагами
        auto_wire_blocks = []
        for (node, inst_name, pin_index, pin_x, pin_y) in connection_list:
            if node == "0":
                fx, fy = 0, 0
            elif node in flag_coord:
                fx, fy = flag_coord[node]
            else:
                continue
            auto_wire_blocks.append(f"WIRE {int(pin_x)} {int(pin_y)} {int(fx)} {int(fy)}")

        asc_lines.extend(auto_flag_blocks)
        asc_lines.extend(auto_wire_blocks)
        asc_lines.extend(comp_blocks)
        asc_lines.extend(list(map(lambda x: x.replace("\\n", "\\\\n"), text_blocks)))

        return "\n".join(asc_lines)

    def netlist_b64_to_asc_with_description(netlist: str) -> str:
        return f"Here's the .asc file contents:\n\n{net_to_asc(base64.b64decode(netlist.encode("utf-8")).decode("utf-8"))}"

    return Tool(
        name="netlist_to_asc",
        description="Converts base64 representation of netlist to .asc file content.",
        func=lambda query: netlist_b64_to_asc_with_description(query)
    )


def send_asc_to_user_tool(chat_id: int, bot: TeleBot):

    def send_asc_to_user(chat_id: int, bot: TeleBot, asc_file_content: str):
        file_obj = io.BytesIO(asc_file_content.encode("iso-8859-1"))
        file_obj.name = "circuit.asc"
        bot.send_document(chat_id, file_obj)

        return "Successfully sent .asc file contents to the user!"

    return Tool(
        name="send_asc_to_user",
        description="Sends .asc file content to the user.",
        func=lambda asc_file_content: send_asc_to_user(chat_id, bot, asc_file_content)
    )


def send_netlist_b64_to_user_tool(chat_id: int, bot: TeleBot):

    def send_netlist_b64_to_user(chat_id: int, bot: TeleBot, netlist: str):
        netlist_str = base64.b64decode(netlist.encode("utf-8")).decode("utf-8")
        file_obj = io.BytesIO(netlist_str.encode("utf-8"))
        file_obj.name = "circuit.net"
        bot.send_document(chat_id, file_obj)

        return "Successfully sent netlist contents to the user!"

    return Tool(
        name="send_netlist_to_user",
        description="Sends base64 representation of netlist to the user.",
        func=lambda asc_file_content: send_netlist_b64_to_user(chat_id, bot, asc_file_content)
    )


def get_netlist_b64_for_butterworth_lowpass_filter_tool():
    def butterworth_low_pass(f, Z, n):
        netlist = [
            f"V1 N001 0 SINE(1 1 {f}) AC 1",
            f"R1 N002 N001 {Z}",
        ]

        num_of_Ls = 0
        num_of_Cs = 0

        # Вычисляем коэффициенты Баттерворта g_k
        g = [2 * math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]

        # Вычисляем значения L и C
        for k in range(n):
            if k % 2 == 0:
                # Расчет конденсатора
                C = g[k] / (2 * math.pi * f * Z)
                node = 1 + num_of_Cs + 1
                netlist.append(f"C{num_of_Cs + 1} N{node:03} 0 {C}")
                num_of_Cs += 1
            else:
                # Расчет катушки индуктивности
                L = (Z * g[k]) / (2 * math.pi * f)
                node = 1 + num_of_Ls + 1
                nodeplus1 = 1 + num_of_Ls + 1 + 1
                netlist.append(f"L{num_of_Ls + 1} N{node:03} N{nodeplus1:03} {L}")
                num_of_Ls += 1

        netlist.extend([
            f"R2 0 N{(2 + num_of_Ls):03} {Z}",
            f".ac dec 1000 10 {f}",
            f".backanno",
            f".end"
        ])

        return text_to_base64("\n".join(netlist))

    class GetNetlistInput(BaseModel):
        f: float = Field(description="Частота среза, Гц")
        Z: float = Field(description="Характеристическое сопротивление, Ом")
        n: int = Field(description="Количество реактивных элементов")

    return StructuredTool.from_function(
        name="get_netlist_b64_for_butterworth_lowpass_filter",
        description="Возвращает base64 репрезентацию netlist для фильтра низких частот Баттерворта.",
        args_schema=GetNetlistInput,
        func=butterworth_low_pass
    )

